"""
Visual out (Deaf / hard-of-hearing mode): port highlight, arrow, big captions.

The same drawing helpers annotate frames sent to the vision LLM (port labels +
expected port circled) and the judges' dashboard.
"""
from __future__ import annotations

import math
import textwrap

import cv2
import numpy as np

from backend.vision.shield import ShieldTracker

GREEN = (60, 220, 60)
YELLOW = (0, 220, 255)
RED = (60, 60, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (230, 200, 40)


def _ascii(text: str) -> str:
    """OpenCV's Hershey fonts are ASCII-only."""
    return (text.replace("—", "-").replace("–", "-").replace("’", "'")
            .replace("→", "->").encode("ascii", "replace").decode())


def _label(img, text, org, scale=0.5, color=YELLOW, thickness=1):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_port_labels(img: np.ndarray, shield: ShieldTracker, highlight: list[str] | None = None) -> np.ndarray:
    """Every port labelled; `highlight` ports circled in green. Used for LLM plug checks."""
    out = img.copy()
    if not shield.ready:
        return out
    r = int(shield.px_per_mm() * shield.layout.pitch_mm * 0.4)
    for name in shield.layout.ports:
        c = shield.port_pixel(name)
        if c is None:
            continue
        if highlight and name in highlight:
            cv2.circle(out, c, r, GREEN, 3, cv2.LINE_AA)
        cv2.circle(out, c, 3, YELLOW, -1)
        _label(out, name, (c[0] - r, c[1] - r - 4), 0.45)
    return out


def _caption_bar(img: np.ndarray, text: str) -> None:
    h, w = img.shape[:2]
    text = _ascii(text)
    if not text.strip():
        return
    # Shrink the font until the whole caption fits in 4 lines; never cut off the start
    # (that's where the port name is).
    scale = max(0.7, w / 900)
    while True:
        chars = max(16, int(w / (scale * 20)))
        lines = textwrap.wrap(text, chars)
        if len(lines) <= 4 or scale <= 0.5:
            break
        scale *= 0.85
    thick = max(1, int(scale * 2))
    line_h = int(38 * scale)
    top = h - line_h * len(lines) - 20
    overlay = img.copy()
    cv2.rectangle(overlay, (0, top - 10), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, dst=img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, top + line_h * (i + 1) - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def _arrow(img, p0, p1, color, thickness=4):
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if d < 8:
        return
    cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=min(0.35, 30 / d))


def draw_guidance(frame: np.ndarray, shield: ShieldTracker, state: dict,
                  fingertip_px: tuple[float, float] | None, t: float, show_all_labels: bool = False) -> np.ndarray:
    """The agent's outgoing video in visual mode (also used for the dashboard)."""
    img = frame.copy()
    h, w = img.shape[:2]

    if shield.ready:
        r = int(shield.px_per_mm() * shield.layout.pitch_mm * 0.45)
        for name in shield.layout.ports:
            c = shield.port_pixel(name)
            if c is None:
                continue
            if name in state.get("occupied", {}):
                cv2.circle(img, c, r // 2, CYAN, 2, cv2.LINE_AA)
            if show_all_labels:
                _label(img, name, (c[0] - r, c[1] - r - 4), 0.4, WHITE)
        targets = state.get("targets") or []
        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
        for name in targets:
            c = shield.port_pixel(name)
            if c is None:
                continue
            on_it = state.get("finger_port") == name
            color = GREEN if on_it else YELLOW
            cv2.circle(img, c, int(r * (1.0 + 0.25 * pulse)), color, 4, cv2.LINE_AA)
            _label(img, name.replace("_", " "), (c[0] + r + 6, c[1] + 8), 0.8, color, 2)
        fp_port = state.get("finger_port")
        if fp_port and fp_port not in targets:
            c = shield.port_pixel(fp_port)
            if c is not None:
                cv2.circle(img, c, r, RED, 2, cv2.LINE_AA)
        if fingertip_px is not None and targets:
            fp = (int(fingertip_px[0]), int(fingertip_px[1]))
            best = min((shield.port_pixel(n) for n in targets if shield.port_pixel(n)),
                       key=lambda c: (c[0] - fp[0]) ** 2 + (c[1] - fp[1]) ** 2, default=None)
            if best is not None:
                _arrow(img, fp, best, GREEN if fp_port in targets else YELLOW)
    else:
        _label(img, "Board not found - show all 4 corner markers", (20, 40), 0.8, RED, 2)

    if fingertip_px is not None:
        cv2.circle(img, (int(fingertip_px[0]), int(fingertip_px[1])), 9, CYAN, -1, cv2.LINE_AA)

    # header: progress + mode
    n, i = state.get("n_steps", 0), state.get("step_index", 0)
    header = f"{state.get('title', '')}  |  step {min(i + 1, n)}/{n}  |  {state.get('show', '')}"
    if state.get("phase") == "done":
        header = f"{state.get('title', '')}  |  done!"
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 72), BLACK, -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, dst=img)
    cv2.putText(img, _ascii(header), (16, 32), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1400), WHITE, 2, cv2.LINE_AA)
    mode = state.get("mode") or "choose mode"
    cv2.putText(img, f"{mode} mode  |  {state.get('phase', '').replace('_', ' ')}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

    _caption_bar(img, state.get("caption", ""))
    return img


def caption_card(state: dict, w: int = 1280, h: int = 720) -> np.ndarray:
    """Plain caption screen for when there is no camera picture to annotate."""
    img = np.full((h, w, 3), (32, 28, 24), np.uint8)
    targets = state.get("targets") or []
    if targets:
        big = "  ".join(t.replace("_", " ") for t in targets[:4])
        scale = 3.0 if len(big) < 10 else 1.6
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)[0]
        cv2.putText(img, big, ((w - size[0]) // 2, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, scale, YELLOW, 6, cv2.LINE_AA)
    n, i = state.get("n_steps", 0), state.get("step_index", 0)
    header = f"{state.get('title', '')}  |  step {min(i + 1, n)}/{n}  |  {state.get('show', '')}"
    cv2.putText(img, _ascii(header), (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    _caption_bar(img, state.get("caption", ""))
    return img
