"""
Crop photos before they go to Claude. Image tokens scale with pixel area (about one token per
28x28 patch), so sending only the region that matters costs less, and because the crop is
taken at full camera resolution it shows that region in *more* detail, not less.

Crops are deliberately generous: anything the check could depend on stays in the picture.
"""
from __future__ import annotations

import numpy as np

# A crop must save at least this much area to be worth it; otherwise send the full frame.
MAX_AREA_FRAC = 0.75


def crop_to_box(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                min_side_px: int = 320) -> tuple[np.ndarray, bool]:
    """Clamp the box to the frame, grow it to `min_side_px`, and crop. Returns (image, cropped?)."""
    h, w = frame.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, min_side_px), max(y1 - y0, min_side_px)
    x0, x1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > MAX_AREA_FRAC * w * h:
        return frame, False
    return frame[y0:y1, x0:x1].copy(), True


def crop_to_shield(frame: np.ndarray, shield, margin_mm: float = 25.0) -> tuple[np.ndarray, bool]:
    """The whole shield (every port and the cables going into them) plus a margin.
    Not a single port: telling *which* cable is the new one needs the whole board."""
    layout = shield.layout
    xs = [p.x for p in layout.ports.values()] + [m[0] for m in layout.markers.values()]
    ys = [p.y for p in layout.ports.values()] + [m[1] for m in layout.markers.values()]
    corners_mm = [(min(xs) - margin_mm, min(ys) - margin_mm), (max(xs) + margin_mm, min(ys) - margin_mm),
                  (max(xs) + margin_mm, max(ys) + margin_mm), (min(xs) - margin_mm, max(ys) + margin_mm)]
    pts = [shield.to_image(c) for c in corners_mm]
    if any(p is None for p in pts):
        return frame, False
    px, py = [p[0] for p in pts], [p[1] for p in pts]
    return crop_to_box(frame, min(px), min(py), max(px), max(py))


def crop_to_hands(frame: np.ndarray, hands, grow: float = 1.0) -> tuple[np.ndarray, bool]:
    """Region around the hand(s) holding a module: the hand box grown by `grow` x its own size
    on every side (3x wide overall), so a module held at the fingertips stays fully in frame."""
    pts = [p for h in hands for p in h.landmarks_px]
    if not pts:
        return frame, False
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    side = max(bw, bh)  # square-ish: hand pose shouldn't decide the crop's shape
    return crop_to_box(frame, min(xs) - grow * side, min(ys) - grow * side,
                       max(xs) + grow * side, max(ys) + grow * side, min_side_px=400)
