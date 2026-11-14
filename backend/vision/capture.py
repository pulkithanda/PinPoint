"""
Photo on demand: pick the best frame from the call while the builder holds the phone up.

The call streams ~15 fps the whole time, so "taking a photo" means waiting for a
frame that is steady (phone held still) and sharp (in focus), and, for plug
checks, actually shows the target port. When the corner markers are in view
we also know where the phone is pointing, so a blind builder can be told
"move the phone to your right" / "move the phone closer" instead of having to
aim by sight. Close-ups that don't show the markers are accepted on
steadiness + sharpness alone (that's the point of zooming in).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.vision.shield import ShieldTracker

SMALL_W = 160          # analysis resolution
STEADY_DIFF = 6.0      # mean abs pixel change between frames (0-255) below which the phone is "still"
MIN_SHARPNESS = 40.0   # variance of the Laplacian; blurry frames score far lower
GOOD_FRAMES = 5        # ~1/3 s of steady, sharp frames at 15 fps
MIN_PX_PER_MM = 3.0    # board looks smaller than this -> "move the phone closer"


def _small_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, (SMALL_W, max(1, int(h * SMALL_W / w))), interpolation=cv2.INTER_AREA)


def sharpness(frame: np.ndarray) -> float:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, int(g.shape[0] * 640 / g.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@dataclass
class CaptureUpdate:
    status: str                       # "waiting" | "captured" | "timeout"
    hint: str | None = None           # aiming instruction to speak/caption, if any
    hint_key: str | None = None
    frame: np.ndarray | None = None   # the chosen frame (captured, or best-effort on timeout)
    labeled: bool = False             # markers were visible -> port labels can be drawn on it


@dataclass
class AutoCapture:
    purpose: str                      # "identify" | "plug" | "question"
    ports: list[str] = field(default_factory=list)
    started_t: float = 0.0
    timeout_s: float = 15.0
    _prev: np.ndarray | None = None
    _run: list[tuple[float, np.ndarray, bool]] = field(default_factory=list)
    _best: tuple[float, np.ndarray, bool] | None = None

    def feed(self, frame: np.ndarray, now: float, shield: ShieldTracker | None = None) -> CaptureUpdate:
        """`shield` must already be updated on this frame; shield.ready means 3+ markers are in view now."""
        small = _small_gray(frame)
        had_prev = self._prev is not None and self._prev.shape == small.shape
        steady = had_prev and float(np.mean(cv2.absdiff(small, self._prev))) < STEADY_DIFF
        self._prev = small
        sharp = sharpness(frame)

        hint = key = None
        on_target, labeled = True, False
        if shield is not None and self.purpose == "plug":
            labeled = shield.ready
            if labeled:
                on_target, hint, key = self._aim(frame, shield)

        good = steady and sharp >= MIN_SHARPNESS and on_target
        if good:
            self._run.append((sharp, frame, labeled))
            if self._best is None or sharp > self._best[0]:
                self._best = (sharp, frame, labeled)
            if len(self._run) >= GOOD_FRAMES:
                s, f, lab = max(self._run, key=lambda r: r[0])
                return CaptureUpdate("captured", frame=f, labeled=lab)
        else:
            self._run.clear()
            if hint is None and had_prev and not steady:
                hint, key = "Hold the phone still.", "aim:still"

        if now - self.started_t >= self.timeout_s:
            if self._best is not None:
                return CaptureUpdate("captured", frame=self._best[1], labeled=self._best[2])
            return CaptureUpdate("timeout")
        return CaptureUpdate("waiting", hint=hint, hint_key=key)

    def _aim(self, frame: np.ndarray, shield: ShieldTracker) -> tuple[bool, str | None, str | None]:
        """Is the target port well inside the frame? If not, which way should the phone move?"""
        h, w = frame.shape[:2]
        if shield.px_per_mm() < MIN_PX_PER_MM:
            return False, "Move the phone closer to the board.", "aim:closer"
        pts = [shield.port_pixel(p) for p in self.ports]
        pts = [p for p in pts if p is not None]
        if not pts:
            return True, None, None
        inside = [p for p in pts if 0.15 * w <= p[0] <= 0.85 * w and 0.15 * h <= p[1] <= 0.85 * h]
        if inside:
            return True, None, None
        # Work in board millimetres, so the advice is in the builder's own left/right/toward/away,
        # whichever way they happen to be holding the phone.
        centre = shield.to_board((w / 2, h / 2))
        port = shield.layout.ports[self.ports[0]]
        if centre is None:
            return False, None, None
        dx, dy = port.x - centre[0], port.y - centre[1]
        if abs(dx) >= abs(dy):
            word = "to your right" if dx > 0 else "to your left"
        else:
            word = "toward you" if dy > 0 else "away from you"
        return False, f"Move the phone a little {word}.", f"aim:{word}"


def capture_prompt(purpose: str, target: str = "", module: str = "") -> str:
    """What to say when a photo is needed."""
    if purpose == "identify":
        return ("Pick up the phone and point it at the module in your other hand, "
                "with the printed name facing the camera. Hold still and I'll take the photo.")
    if purpose == "plug":
        return (f"Pick up the phone and hold it about a hand's width above {target}, "
                f"camera pointing down. Hold still and I'll take the photo.")
    return "Point the phone at what you're asking about and hold still. I'll take the photo."
