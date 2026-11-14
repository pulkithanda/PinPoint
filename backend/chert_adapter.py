"""
The ONLY file that talks to Chert (FaceTime video in/out, audio, iMessage)

!! The Chert FaceTime API shape is not known yet. !!
Everything Chert-specific is behind the five TODO(chert) points below; once
we have the docs, only this file changes. Open questions (ask at the Chert table):
  * Does the agent receive continuous video frames? At what fps / latency?
    How are they delivered (webhook per frame, websocket, WebRTC track)?
  * Can the agent send video back (for Deaf mode), or only audio?
  * Does Chert provide speech-to-text of the caller? Audio format for TTS out?
  * SDK language, auth, and is the iMessage recap in the same API?

If live frames or outgoing video aren't available, set
PINPOINT_PLACE_THEN_CHECK=1: the engine then describes ports by touch and
only needs occasional snapshots + the serial check.

Current assumption (easy to change): Chert POSTs JSON events to
/chert/webhook, e.g.
  {"type": "call.started", "call_id": "..."}
  {"type": "video.frame", "call_id": "...", "jpeg_b64": "..."}
  {"type": "transcript", "call_id": "...", "text": "next"}
  {"type": "call.ended", "call_id": "..."}
"""
from __future__ import annotations

import base64
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from backend.transport import CallTransport

log = logging.getLogger(__name__)

CHERT_API_KEY = os.environ.get("CHERT_API_KEY", "")
CHERT_API_BASE = os.environ.get("CHERT_API_BASE", "")

class ChertTransport(CallTransport):
    local_audio = False  # speech is rendered to WAV and sent down the call

    def __init__(self):
        self.call_id: str | None = None
        self._latest: tuple[float, np.ndarray] | None = None
        self._seq = 0
        self._read_seq = 0
        self._lock = threading.Lock()
        self.transcripts: queue.Queue[str] = queue.Queue()
        self._warned: set[str] = set()

    # ------------------------------------------------------------ inbound (webhook -> here)
    def handle_webhook(self, event: dict) -> dict:
        """Called by main.py for every Chert event. TODO(chert) #1: match the real event schema."""
        etype = event.get("type", "")
        if etype == "call.started":
            self.call_id = event.get("call_id")
            log.info("Chert call started: %s", self.call_id)
        elif etype == "video.frame":
            frame = self._decode_frame(event)
            if frame is not None:
                with self._lock:
                    self._latest = (time.monotonic(), frame)
                    self._seq += 1
        elif etype == "transcript":
            text = (event.get("text") or "").strip()
            if text:
                self.transcripts.put(text)
        elif etype == "message.received":  # typed iMessage backup for Deaf users
            text = (event.get("text") or "").strip()
            if text:
                self.transcripts.put(text)
        elif etype == "call.ended":
            log.info("Chert call ended: %s", self.call_id)
        return {"ok": True}

    @staticmethod
    def _decode_frame(event: dict) -> np.ndarray | None:
        """TODO(chert) #2: frames may arrive as JPEG, raw RGB, or via a stream instead of webhooks."""
        b64 = event.get("jpeg_b64") or event.get("image_b64")
        if not b64:
            return None
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

    # ------------------------------------------------------------ outbound (here -> Chert)
    def _todo(self, what: str) -> None:
        if what not in self._warned:
            self._warned.add(what)
            log.warning("Chert %s not wired up yet (see TODO(chert) in chert_adapter.py)", what)

    def send_speech(self, text: str, wav: bytes | None) -> None:
        """TODO(chert) #3: send `wav` (16-bit PCM WAV) as the agent's audio on the call."""
        self._todo("audio out")

    def send_beep(self, wav: bytes) -> None:
        self._todo("audio out")

    def send_video_frame(self, frame_bgr: np.ndarray) -> None:
        """TODO(chert) #4: send the annotated frame as the agent's outgoing video (Deaf mode)."""
        self._todo("video out")

    def send_message(self, text: str, images_jpeg: list[bytes] | None = None) -> None:
        """TODO(chert) #5: iMessage recap to the caller (text + annotated screenshots)."""
        log.info("[recap message]\n%s", text)
        self._todo("iMessage")
