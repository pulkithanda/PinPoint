"""
Speech out (blind mode): rate limiting, TTS, pre-rendered clips, proximity beeps.

SpeechScheduler is pure logic (testable): one steering instruction per ~0.8 s,
latest-wins for steering, nothing dropped for informational lines, and no
repeats of an unchanged instruction.

TTSWorker owns a single offline pyttsx3 engine on its own thread (SAPI on
Windows requires that) and can either speak locally or render WAV bytes to
send down the call. Rendered clips are cached on disk, so "left one port",
"down two", every port name, etc. are pre-rendered once and reused.
"""
from __future__ import annotations

import hashlib
import io
import logging
import queue
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from backend.guidance.engine import Say

log = logging.getLogger(__name__)


class SpeechScheduler:
    def __init__(self, min_interval_s: float = 0.8, repeat_after_s: float = 4.0):
        self.min_interval_s = min_interval_s
        self.repeat_after_s = repeat_after_s
        self.queue: deque[Say] = deque()
        self.pending_steer: Say | None = None
        self.last_key: str | None = None
        self.last_t = -1e9

    def submit(self, say: Say, now: float) -> None:
        if say.key and say.key == self.last_key and now - self.last_t < self.repeat_after_s:
            return  # unchanged instruction, said recently
        if say.kind == "steer":
            self.pending_steer = say
        elif say.kind == "urgent":
            self.pending_steer = None
            self.queue.appendleft(say)
        else:
            self.pending_steer = None
            self.queue.append(say)

    def clear(self) -> None:
        """The caller interrupted: forget everything not yet said."""
        self.queue.clear()
        self.pending_steer = None

    @property
    def pending(self) -> bool:
        return bool(self.queue) or self.pending_steer is not None

    def pop_ready(self, now: float, busy: bool = False) -> Say | None:
        if busy:
            return None
        if self.queue:
            s = self.queue.popleft()
        elif self.pending_steer and now - self.last_t >= self.min_interval_s:
            s, self.pending_steer = self.pending_steer, None
        else:
            return None
        self.last_key, self.last_t = s.key, now
        return s


def tone_wav(freq_hz: float, ms: int = 60, rate: int = 16000, volume: float = 0.4) -> bytes:
    t = np.arange(int(rate * ms / 1000)) / rate
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.005)  # 5 ms fade in/out, no clicks
    pcm = (np.sin(2 * np.pi * freq_hz * t) * env * volume * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sapi_renderer():
    """Direct Windows SAPI file rendering (~0.05 s a clip). Must be created on the thread that uses it."""
    import sys

    if sys.platform != "win32":
        return None
    try:
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        voice.Rate = 1
        from comtypes.gen import SpeechLib
    except Exception as e:
        log.warning("Direct SAPI unavailable (%s); using pyttsx3 for clips", e)
        return None

    def render(text: str, path: Path) -> None:
        tmp = path.with_suffix(".tmp.wav")
        stream = comtypes.client.CreateObject("SAPI.SpFileStream")
        stream.Open(str(tmp), SpeechLib.SSFMCreateForWrite)
        try:
            voice.AudioOutputStream = stream
            voice.Speak(text)
        finally:
            stream.Close()
        tmp.replace(path)  # never leave a half-written clip in the cache

    return render


def _neural_renderer():
    """Natural-sounding Microsoft neural voice via the `edge-tts` package (online, free, no key).
    Returns None if the package isn't installed or PINPOINT_VOICE=sapi, so the offline Windows
    voice is used instead. Any failure on a clip also falls back to the offline voice."""
    import os

    voice = os.environ.get("PINPOINT_VOICE", "en-US-AvaNeural").strip()
    if not voice or voice.lower() in ("sapi", "off", "none"):
        return None
    try:
        import asyncio

        import av
        import edge_tts
    except ImportError:
        log.info("edge-tts not installed: using the offline Windows voice (pip install edge-tts for a natural one)")
        return None
    rate = os.environ.get("PINPOINT_VOICE_RATE", "+20%")

    def render(text: str, path: Path) -> None:
        mp3 = path.with_suffix(".tmp.mp3")
        asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, voice, rate=rate).save(str(mp3)), timeout=8.0))
        try:
            pcm = []
            with av.open(str(mp3)) as container:
                resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                for frame in container.decode(audio=0):
                    for out in resampler.resample(frame):
                        pcm.append(out.to_ndarray().reshape(-1))
            data = np.concatenate(pcm).astype(np.int16) if pcm else np.zeros(0, np.int16)
            tmp = path.with_suffix(".tmp.wav")
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(data.tobytes())
            tmp.replace(path)
        finally:
            mp3.unlink(missing_ok=True)

    log.info("Using the neural voice %s", voice)
    render.cache_key = f"{voice}|{rate}"  # type: ignore[attr-defined]
    return render


class TTSWorker:
    """Single thread that owns the pyttsx3 engine; speaks locally and/or renders WAV clips."""

    _shared: dict = {}
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls, cache_dir: Path, speak_locally: bool = True) -> "TTSWorker":
        """One worker per process: pyttsx3.init() returns a process-wide engine, and driving it
        from two threads at once (e.g. two LiveKit jobs) deadlocks the Windows SAPI driver."""
        key = (str(cache_dir), speak_locally)
        with cls._shared_lock:
            if key not in cls._shared:
                cls._shared[key] = cls(cache_dir, speak_locally=speak_locally)
            return cls._shared[key]

    def __init__(self, cache_dir: Path, rate_wpm: int = 210, speak_locally: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_wpm = rate_wpm
        self.speak_locally = speak_locally
        self._neural = _neural_renderer() if not speak_locally else None
        self.jobs: queue.Queue = queue.Queue()
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate_wpm)
        except Exception as e:  # no TTS backend on this machine: degrade to printing
            log.warning("pyttsx3 unavailable (%s); speech will be printed only", e)
            engine = None
        sapi = _sapi_renderer()  # Windows: pyttsx3's save_to_file hangs after the first clip
        neural = self._neural
        while True:
            kind, text, reply = self.jobs.get()
            try:
                if kind == "say":
                    self.busy = True
                    print(f"[speak] {text}", flush=True)
                    if engine is not None and self.speak_locally:
                        engine.say(text)
                        engine.runAndWait()
                elif kind == "render":
                    path = self.clip_path(text)
                    if not path.exists() and neural is not None:
                        try:
                            neural(text, path)
                        except Exception as e:  # offline, blocked or slow: the robot voice beats silence
                            log.warning("Neural voice failed (%s); using the offline voice for this line", e)
                            path = self._fallback_path(text)
                    if not path.exists() and sapi is not None:
                        sapi(text, path)
                    elif not path.exists() and engine is not None:
                        engine.save_to_file(text, str(path))
                        engine.runAndWait()
                    reply.put(path.read_bytes() if path.exists() else None)
            except Exception as e:
                log.error("TTS job failed: %s", e)
                if reply is not None:
                    reply.put(None)
            finally:
                self.busy = False

    def clip_path(self, text: str) -> Path:
        # Clips from different voices never mix in the cache.
        profile = self._neural.cache_key if self._neural is not None else f"sapi-rate1|{self.rate_wpm}"
        key = f"{profile}|{text}"
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".wav")

    def _fallback_path(self, text: str) -> Path:
        key = f"sapi-fallback|{self.rate_wpm}|{text}"
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".wav")

    def say(self, text: str) -> None:
        self.busy = True  # set immediately so the scheduler doesn't double-book
        self.jobs.put(("say", text, None))

    def render(self, text: str, timeout_s: float = 10.0) -> bytes | None:
        reply: queue.Queue = queue.Queue()
        self.jobs.put(("render", text, reply))
        try:
            return reply.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def prerender(self, phrases: list[str]) -> None:
        for p in phrases:
            if not self.clip_path(p).exists():
                self.jobs.put(("render", p, queue.Queue()))


def common_phrases(port_names: list[str]) -> list[str]:
    nums = ["one", "two", "three", "four"]
    out = []
    for w in ("Right", "Left"):
        out += [f"{w} one port."] + [f"{w} {n} ports." for n in nums[1:]]
    for w in ("Up", "Down"):
        out += [f"{w} {n}." for n in nums]
    out += [f"Move {w}, onto the board." for w in ("right", "left", "up", "down")]
    out += [f"Stop. That's {p.replace('_', ' ')}. Plug it in." for p in port_names if not p.startswith("I2C")]
    out += ["Stop. That's an I 2 C port. Plug it in.", "Let me check."]
    return out


class LocalBeeper:
    """Parking-sensor beeps on the laptop speaker: faster and higher as the finger closes in."""

    def __init__(self):
        self.closeness: float | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="beeper", daemon=True).start()

    def set(self, closeness: float | None) -> None:
        self.closeness = closeness

    def _run(self) -> None:
        while not self._stop.is_set():
            c = self.closeness
            if c is None or c <= 0.05:
                time.sleep(0.1)
                continue
            freq = int(500 + 1000 * c)
            if sys.platform == "win32":
                import winsound

                winsound.Beep(freq, 50)
            time.sleep(max(0.05, 0.6 - 0.55 * c))
