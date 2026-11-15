"""Map Elegoo breadboard holes into the marker frame around the Arduino."""
from __future__ import annotations

import math

import cv2
import numpy as np

from backend.build.board import Orientation, parse_hole
from backend.guidance.overlay import CYAN, YELLOW, _arrow, _label
from backend.vision.shield import ShieldTracker

_PITCH_MM = 2.54
_BREADBOARD_LENGTH_MM = 165.1
_BREADBOARD_WIDTH_MM = 55.88
_HOLE_COUNT = 63


def hole_position_mm(name: str, orientation: Orientation, markers: dict[int, tuple[float, float]]) \
        -> tuple[float, float] | None:
    """Return a standard 830-point breadboard hole position in the Arduino marker frame."""
    parsed = parse_hole(name)
    if parsed is None or not markers:
        return None
    col, row = parsed
    xs = [p[0] for p in markers.values()]
    ys = [p[1] for p in markers.values()]
    x_center = (min(xs) + max(xs)) / 2
    x_margin = (_BREADBOARD_LENGTH_MM - (_HOLE_COUNT - 1) * _PITCH_MM) / 2
    board_left = x_center - _BREADBOARD_LENGTH_MM / 2
    row_index = row - 1 if orientation.row1 == "left" else _HOLE_COUNT - row
    x = board_left + x_margin + row_index * _PITCH_MM

    y_far_edge = max(ys)
    y_margin = (_BREADBOARD_WIDTH_MM - 11 * _PITCH_MM) / 2
    near_order = "ABCDE" if orientation.near_half == "AE" else "FGHIJ"
    far_order = "JIHGF" if orientation.near_half == "AE" else "EDCBA"
    if col in near_order:
        y = y_far_edge + _BREADBOARD_WIDTH_MM - y_margin - near_order.index(col) * _PITCH_MM
    else:
        y = y_far_edge + y_margin + far_order.index(col) * _PITCH_MM
    return x, y


def draw_breadboard_targets(frame: np.ndarray, tracker: ShieldTracker, targets: list[str],
                            orientation: Orientation, fingertip_px: tuple[float, float] | None) -> np.ndarray:
    """Draw target-hole rings and a fingertip arrow only when marker geometry is available."""
    if not tracker.ready:
        return frame
    img = frame.copy()
    pixels: list[tuple[str, tuple[int, int]]] = []
    for name in dict.fromkeys(targets):
        pos = hole_position_mm(name, orientation, tracker.layout.markers)
        if pos is None:
            continue
        center = tracker.to_image(pos)
        edge = tracker.to_image((pos[0] + 3.5, pos[1]))
        if center is None or edge is None:
            continue
        radius = max(6, int(round(math.hypot(edge[0] - center[0], edge[1] - center[1]))))
        cv2.circle(img, center, radius, YELLOW, 3, cv2.LINE_AA)
        _label(img, name, (center[0] + radius + 3, center[1] + 5), 0.55, YELLOW, 2)
        pixels.append((name, center))

    if fingertip_px is not None and pixels:
        finger = (int(fingertip_px[0]), int(fingertip_px[1]))
        cv2.circle(img, finger, 8, CYAN, -1, cv2.LINE_AA)
        nearest = min(pixels, key=lambda item: (item[1][0] - finger[0]) ** 2 + (item[1][1] - finger[1]) ** 2)
        _arrow(img, finger, nearest[1], YELLOW, thickness=3)
    return img
