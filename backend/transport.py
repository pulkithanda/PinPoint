"""
Call transport interface + a laptop webcam implementation

Everything above this layer (vision, guidance, speech scheduling) is identical
whether frames come from a webcam or a FaceTime call. The webcam transport lets
the vision and guidance work proceed without Chert ("build the vision against a
laptop webcam first").
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger(__name__)

class CallTransport(ABC):
    #: True -> speech is played on this machine's speakers; 
    # False -> WAV bytes go down the call
    local_audio: bool = True

    def start(self) -> None:
        pass

    @abstractmethod
    def read_frame(self) -> tuple[float, np.ndarray] | None:
        """Latest frame (stale frames are dropped), or None if nothing new."""

    def poll_transcripts(self) -> list[str]:
        """Speech-to-text (or typed iMessage) lines received since the last poll."""
        return []

    def send_speech(self, text: str, wav: bytes | None) -> None:
        pass

    def send_beep(self, wav: bytes) -> None:
        pass

    def send_video_frame(self, frame_bgr: np.ndarray) -> None:
        pass

    def send_message(self, text: str, images_jpeg: list[bytes] | None = None) -> None:
        log.info("[recap message]\n%s", text)

    def close(self) -> None:
        pass


class WebcamTransport(CallTransport):
    local_audio = True

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.size = (width, height)
        self._latest: tuple[float, np.ndarray] | None = None
        self._seq = 0
        self._read_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.transcripts: queue.Queue[str] = queue.Queue()

    def start(self) -> None:
        threading.Thread(target=self._grab, name="webcam", daemon=True).start()

    def _grab(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if not cap.isOpened():
            log.error("Could not open camera %s", self.camera_index)
            return
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = (time.monotonic(), frame)
                self._seq += 1
        cap.release()

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

    def close(self) -> None:
        self._stop.set()
