"""
Pretend to be Chert's FaceTime bridge, so the whole call path can be tested without Chert.

    livekit-server --dev                      # local LiveKit (keys devkey / secret)
    python -m backend.main --source chert     # with LIVEKIT_URL=ws://127.0.0.1:7880 etc. in .env
    python -m tools.fake_bridge --say "voice mode" --say "ready"

What it does, exactly like Chert would:
  1. POSTs a `call.incoming` webhook to the backend and reads the accept response.
  2. Joins the LiveKit room with the returned participant_token.
  3. Publishes `facetime-remote-video` (an image, or a webcam) and `facetime-remote-audio`
     (silence, plus each --say phrase spoken with the laptop's TTS a few seconds apart).
  4. Subscribes to the agent's voice and video and reports what came back
     (saves the last agent video frame to --save).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import tempfile
import uuid

import cv2
import httpx
import numpy as np

from backend.livekit_transport import wav_to_pcm

RATE = 48000


def tts_wav(text: str) -> bytes:
    import pyttsx3

    path = os.path.join(tempfile.gettempdir(), f"fake_bridge_{uuid.uuid4().hex}.wav")
    eng = pyttsx3.init()
    eng.save_to_file(text, path)
    eng.runAndWait()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def default_image() -> np.ndarray:
    from backend.config import Settings
    from backend.vision.shield import ShieldLayout
    from tests.test_shield import render_board

    board = render_board(ShieldLayout.load(Settings().layout_path))
    return cv2.copyMakeBorder(board, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=(255, 255, 255))


async def run(args) -> None:
    from livekit import rtc

    call_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    event = {"event_id": str(uuid.uuid4()), "type": "call.incoming", "created_at": now,
             "data": {"call_id": call_id, "line_id": "fake-line", "media": {"audio": True, "video": True},
                      "from_handle_hash": None, "occurred_at": now}}
    async with httpx.AsyncClient(timeout=5.0) as http:
        r = await http.post(args.webhook, json=event)
    accept = r.json()
    print("webhook response:", {k: (v[:24] + "..." if isinstance(v, str) and len(v) > 40 else v) for k, v in accept.items()})
    if accept.get("action") != "accept":
        return

    room = rtc.Room()
    stats = {"agent_audio_frames": 0, "agent_loud_frames": 0, "agent_video_frames": 0, "last_video": None}

    async def read_agent_audio(track):
        async for ev in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            stats["agent_audio_frames"] += 1
            if np.abs(np.frombuffer(ev.frame.data, np.int16)).mean() > 200:
                stats["agent_loud_frames"] += 1

    async def read_agent_video(track):
        async for ev in rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24):
            f = ev.frame
            stats["agent_video_frames"] += 1
            stats["last_video"] = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 3).copy()

    @room.on("track_subscribed")
    def _sub(track, pub, participant):
        if participant.identity != accept.get("remote_participant_identity"):
            return
        task = read_agent_audio if track.kind == rtc.TrackKind.KIND_AUDIO else read_agent_video
        asyncio.ensure_future(task(track))

    await room.connect(accept["livekit_url"], accept["participant_token"])
    print("bridge joined room", room.name)

    vsrc = rtc.VideoSource(1280, 720)
    await room.local_participant.publish_track(rtc.LocalVideoTrack.create_video_track("facetime-remote-video", vsrc))
    asrc = rtc.AudioSource(RATE, 1)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("facetime-remote-audio", asrc))

    cap = cv2.VideoCapture(args.camera) if args.camera is not None else None
    still = None if cap else (cv2.imread(args.image) if args.image else default_image())

    async def push_video():
        while True:
            frame = still
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.05)
                    continue
            rgba = cv2.cvtColor(cv2.resize(frame, (1280, 720)), cv2.COLOR_BGR2RGBA)
            vsrc.capture_frame(rtc.VideoFrame(1280, 720, rtc.VideoBufferType.RGBA, rgba.tobytes()))
            await asyncio.sleep(1 / 15)

    phrases = [wav_to_pcm(tts_wav(p), RATE) for p in args.say]

    async def push_audio():
        chunk = RATE // 100
        silence = np.zeros(chunk, np.int16)
        t = 0.0
        queue = list(zip([args.first_say_after + i * args.say_every for i in range(len(phrases))], phrases, args.say))
        while True:
            if queue and t >= queue[0][0]:
                _, pcm, text = queue.pop(0)
                print(f"[caller says] {text}")
                for i in range(0, len(pcm), chunk):
                    part = np.pad(pcm[i:i + chunk], (0, max(0, chunk - len(pcm[i:i + chunk]))))
                    await asrc.capture_frame(rtc.AudioFrame(part.tobytes(), RATE, 1, chunk))
                    t += 0.01
            await asrc.capture_frame(rtc.AudioFrame(silence.tobytes(), RATE, 1, chunk))
            t += 0.01

    tasks = [asyncio.ensure_future(push_video()), asyncio.ensure_future(push_audio())]
    await asyncio.sleep(args.seconds)
    for t in tasks:
        t.cancel()
    await room.disconnect()
    print(f"agent audio frames: {stats['agent_audio_frames']} (speaking: {stats['agent_loud_frames']}), "
          f"agent video frames: {stats['agent_video_frames']}")
    if stats["last_video"] is not None and args.save:
        cv2.imwrite(args.save, cv2.cvtColor(stats["last_video"], cv2.COLOR_RGB2BGR))
        print("saved last agent video frame to", args.save)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook", default="http://127.0.0.1:8000/chert/webhook")
    ap.add_argument("--image", help="still image to send as the caller's camera (default: a synthetic shield)")
    ap.add_argument("--camera", type=int, help="send this webcam as the caller's camera instead")
    ap.add_argument("--say", action="append", default=[], help="phrase the caller speaks (repeatable)")
    ap.add_argument("--first-say-after", type=float, default=6.0)
    ap.add_argument("--say-every", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--save", default="agent_video.png")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
