"""
Call media over LiveKit (how Chert's FaceTime bridge carries audio and video).

Chert joins a LiveKit room as the "bridge" participant and publishes the caller's
camera as the track `facetime-remote-video` and their mic as `facetime-remote-audio`.
Pinpoint joins the same room as the agent participant and:
  * reads caller video -> latest frame for the session loop
  * reads caller audio -> faster-whisper -> commands/questions
  * publishes its voice (TTS) as an audio track
  * publishes an annotated video track: target port highlighted + big burned-in captions.
    Chert has no captions API, so this video is how Deaf-mode captions reach the caller.

Docs: https://docs.trychert.com/facetime/audio-video
"""
from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import time
import wave

import cv2
import numpy as np

from backend.transport import CallTransport

log = logging.getLogger(__name__)

REMOTE_VIDEO_TRACK = "facetime-remote-video"
REMOTE_AUDIO_TRACK = "facetime-remote-audio"
OUT_RATE = 48000
OUT_W, OUT_H = 1280, 720


def wav_to_pcm(wav: bytes, rate: int = OUT_RATE) -> np.ndarray:
    """Any 16-bit PCM WAV -> mono int16 at `rate`."""
    with wave.open(io.BytesIO(wav)) as w:
        ch, sr, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit WAV")
    x = np.frombuffer(raw, "<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate and len(x):
        n = int(len(x) * rate / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def letterbox(frame: np.ndarray, w: int = OUT_W, h: int = OUT_H) -> np.ndarray:
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


class LiveKitTransport(CallTransport):
    local_audio = False
    wants_video = True

    def __init__(self, url: str, token: str, *, stt_model: str = "base.en", messaging=None,
                 bridge_identity: str | None = None, agent_identity: str = "pinpoint-agent"):
        self.url, self.token = url, token
        self.stt_model = stt_model
        self.messaging = messaging
        self.bridge_identity = bridge_identity
        self.agent_identity = agent_identity
        self.call_ended = False
        self.connected = threading.Event()
        self.transcripts: queue.Queue[str] = queue.Queue()

        self._latest: tuple[float, np.ndarray] | None = None
        self._seq = self._read_seq = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._room = None
        self._asrc = self._vsrc = None
        self._play_q: asyncio.Queue | None = None
        self._playing = False
        self._interrupt = False
        self._speech_starts = self._speech_starts_read = 0
        self._stt = None
        self._last_video_out = 0.0
        self._stop = threading.Event()
        self._attached = False
        self._seen_tracks: set[str] = set()
        self._tasks: list[asyncio.Task] = []

    @classmethod
    def for_room(cls, room, *, stt_model: str = "base.en", messaging=None) -> "LiveKitTransport":
        """Use a room someone else already joined (a LiveKit Agents job dispatched by `chert dev`)."""
        t = cls("", "", stt_model=stt_model, messaging=messaging)
        t._room = room
        return t

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._attached or self._room is not None:
            return  # job mode: the agent entrypoint awaits attach() itself
        threading.Thread(target=self._thread_main, name="livekit", daemon=True).start()

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception:
            log.exception("LiveKit transport crashed")
        finally:
            self.call_ended = True

    async def _main(self) -> None:
        from livekit import rtc

        room = rtc.Room()
        await room.connect(self.url, self.token)
        log.info("Joined LiveKit room %s as %s", room.name, self.agent_identity)
        await self.attach(room)
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
        for task in self._tasks:
            task.cancel()
        await room.disconnect()

    async def attach(self, room=None) -> None:
        """Publish Pinpoint's voice + video into an already-connected room and read the caller's tracks."""
        from livekit import rtc

        if self._attached:
            return
        self._attached = True
        self._rtc = rtc
        room = self._room = room or self._room
        self._loop = asyncio.get_running_loop()
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("participant_disconnected", self._on_participant_left)
        room.on("disconnected", lambda *a: self._end("room closed"))

        # The Mac takes audio from the ONE participant publishing an unmuted audio track and shows
        # video only from that same participant, so publish both from here for the whole call.
        self._asrc = rtc.AudioSource(OUT_RATE, 1)
        atrack = rtc.LocalAudioTrack.create_audio_track("pinpoint-voice", self._asrc)
        opts = rtc.TrackPublishOptions()
        opts.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(atrack, opts)

        self._vsrc = rtc.VideoSource(OUT_W, OUT_H)
        vtrack = rtc.LocalVideoTrack.create_video_track("pinpoint-video", self._vsrc)
        vopts = rtc.TrackPublishOptions()
        vopts.source = rtc.TrackSource.SOURCE_CAMERA
        await room.local_participant.publish_track(vtrack, vopts)

        self._play_q = asyncio.Queue()
        self._tasks.append(asyncio.create_task(self._player()))
        self.connected.set()
        # Tracks subscribed before we attached (ctx.connect auto-subscribes) never fire the event again.
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    self._on_track_subscribed(pub.track, pub, participant)

    def _end(self, why: str) -> None:
        if not self.call_ended:
            log.info("Call ended (%s)", why)
        self.call_ended = True
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        if self._loop is not None and self._attached:
            for task in self._tasks:
                self._loop.call_soon_threadsafe(task.cancel)

    # ------------------------------------------------------------ inbound media
    def _is_bridge(self, identity: str) -> bool:
        if self.bridge_identity:
            return identity == self.bridge_identity
        return identity.startswith("chert-bridge-")

    def _on_track_subscribed(self, track, publication, participant) -> None:
        rtc = self._rtc
        if publication.sid in self._seen_tracks:
            return
        self._seen_tracks.add(publication.sid)
        name = publication.name or ""
        from_bridge = self._is_bridge(participant.identity)
        if track.kind == rtc.TrackKind.KIND_VIDEO and (name in (REMOTE_VIDEO_TRACK, "") or from_bridge):
            log.info("Caller video track subscribed (%s from %s)", name, participant.identity)
            self._tasks.append(asyncio.ensure_future(self._read_video(track)))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            log.info("Ignoring video track %r from %s", name, participant.identity)
        elif track.kind == rtc.TrackKind.KIND_AUDIO and (name in (REMOTE_AUDIO_TRACK, "") or from_bridge):
            log.info("Caller audio track subscribed (%s from %s)", name, participant.identity)
            self._tasks.append(asyncio.ensure_future(self._read_audio(track)))

    def _on_participant_left(self, participant) -> None:
        if self._is_bridge(participant.identity):
            self._end(f"bridge {participant.identity} left")

    async def _read_video(self, track) -> None:
        rtc = self._rtc
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        async for ev in stream:
            f = ev.frame
            rgb = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._latest = (time.monotonic(), bgr)
                self._seq += 1

    async def _read_audio(self, track) -> None:
        from backend.stt import Transcriber

        if self._stt is None:
            self._stt = Transcriber(self.transcripts.put, self.stt_model, on_speech_start=self._on_speech_start)
        stream = self._rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        async for ev in stream:
            self._stt.feed_pcm(np.frombuffer(ev.frame.data, np.int16), 16000, agent_speaking=self.speaking)

    def read_frame(self):
        with self._lock:
            if self._latest is None or self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._latest

    def poll_transcripts(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.transcripts.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------ outbound media
    def _submit(self, fn, *args) -> None:
        if self._loop is not None and self.connected.is_set():
            self._loop.call_soon_threadsafe(fn, *args)

    def send_speech(self, text: str, wav: bytes | None) -> None:
        if wav:
            self._submit(self._play_q.put_nowait, wav_to_pcm(wav))

    def send_beep(self, wav: bytes) -> None:
        self.send_speech("", wav)

    # ------------------------------------------------------------ barge-in
    def _on_speech_start(self) -> None:
        """The caller started talking (STT thread or event loop)."""
        with self._lock:
            self._speech_starts += 1

    def poll_barge_in(self) -> bool:
        with self._lock:
            seen = self._speech_starts != self._speech_starts_read
            self._speech_starts_read = self._speech_starts
        return seen

    @property
    def speaking(self) -> bool:
        return self._playing or (self._play_q is not None and not self._play_q.empty())

    def stop_speech(self) -> None:
        """Cut Pinpoint off mid-sentence and drop anything queued (the caller is talking)."""
        def _stop():
            while self._play_q is not None and not self._play_q.empty():
                self._play_q.get_nowait()
            if self._playing:
                self._interrupt = True
        self._submit(_stop)

    async def _player(self) -> None:
        rtc, chunk = self._rtc, OUT_RATE // 100  # 10 ms frames
        while True:
            pcm = await self._play_q.get()
            self._playing, self._interrupt = True, False
            for i in range(0, len(pcm), chunk):
                if self._interrupt:
                    if hasattr(self._asrc, "clear_queue"):
                        self._asrc.clear_queue()  # drop audio already buffered for sending
                    break
                part = pcm[i:i + chunk]
                if len(part) < chunk:
                    part = np.pad(part, (0, chunk - len(part)))
                await self._asrc.capture_frame(rtc.AudioFrame(part.tobytes(), OUT_RATE, 1, chunk))
            self._playing = self._interrupt = False

    def send_video_frame(self, frame_bgr: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_video_out < 1 / 15 or self._vsrc is None:
            return
        self._last_video_out = now
        rgba = cv2.cvtColor(letterbox(frame_bgr), cv2.COLOR_BGR2RGBA)
        vf = self._rtc.VideoFrame(OUT_W, OUT_H, self._rtc.VideoBufferType.RGBA, rgba.tobytes())
        self._submit(self._vsrc.capture_frame, vf)

    def send_message(self, text: str, images_jpeg: list[bytes] | None = None) -> None:
        if self.messaging is None:
            log.info("[recap message]\n%s", text)
            return
        self.messaging.send(text, images_jpeg)
