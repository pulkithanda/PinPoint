"""Hand tracking: MediaPipe Gesture Recognizer (21 landmarks + gestures in one model).

The fingertip of the hand holding the cable (landmark 8) is transformed into
board mm by the caller; this module also smooths it over ~5 frames and
debounces gestures (thumbs-up = next, open palm = repeat in visual mode).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


@dataclass
class HandObservation:
    fingertip_px: tuple[float, float]
    landmarks_px: list[tuple[float, float]]
    gesture: str | None      # e.g. "Thumb_Up", "Open_Palm", "None"
    gesture_score: float
    handedness: str


class HandTracker:
    def __init__(self, model_path: str | Path, fingertip_landmark: int = 8, num_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Run: python -m tools.download_models"
            )
        self._mp = mp
        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._rec = mp_vision.GestureRecognizer.create_from_options(options)
        self.fingertip_landmark = fingertip_landmark
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, ts_s: float) -> list[HandObservation]:
        h, w = frame_bgr.shape[:2]
        ts_ms = max(int(ts_s * 1000), self._last_ts_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._rec.recognize_for_video(image, ts_ms)
        hands: list[HandObservation] = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = [(lm.x * w, lm.y * h) for lm in lms]
            g = res.gestures[i][0] if i < len(res.gestures) and res.gestures[i] else None
            hd = res.handedness[i][0].category_name if i < len(res.handedness) and res.handedness[i] else "?"
            hands.append(HandObservation(
                fingertip_px=pts[self.fingertip_landmark],
                landmarks_px=pts,
                gesture=g.category_name if g else None,
                gesture_score=float(g.score) if g else 0.0,
                handedness=hd,
            ))
        return hands

    def close(self) -> None:
        self._rec.close()


class PointSmoother:
    """Median over the last N points: kills single-frame jitter without much lag."""

    def __init__(self, n: int = 5):
        self.buf: deque[tuple[float, float]] = deque(maxlen=n)

    def add(self, pt: tuple[float, float] | None) -> tuple[float, float] | None:
        if pt is None:
            self.buf.clear()
            return None
        self.buf.append(pt)
        arr = np.array(self.buf)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


class GestureDebouncer:
    """Emit a gesture once after it is held for `hold_frames`, then wait `cooldown_s`."""

    def __init__(self, hold_frames: int = 8, cooldown_s: float = 1.5, min_score: float = 0.6,
                 watch: tuple[str, ...] = ("Thumb_Up", "Open_Palm")):
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self.min_score = min_score
        self.watch = watch
        self._current: str | None = None
        self._count = 0
        self._last_emit = -1e9
        self._armed = True

    def update(self, gesture: str | None, score: float, now: float) -> str | None:
        g = gesture if gesture in self.watch and score >= self.min_score else None
        if g != self._current:
            self._current, self._count, self._armed = g, 0, True
        if g is None:
            return None
        self._count += 1
        if self._armed and self._count >= self.hold_frames and now - self._last_emit >= self.cooldown_s:
            self._armed = False  # must release the gesture before it fires again
            self._last_emit = now
            return g
        return None
