"""Photo crops for Claude must never cut out anything the check depends on."""
import numpy as np

from backend.config import Settings
from backend.vision.crop import crop_to_box, crop_to_hands, crop_to_shield
from backend.vision.hands import HandObservation
from backend.vision.shield import ShieldLayout, ShieldTracker


def _tracker(px_per_mm: float, ox: float, oy: float) -> ShieldTracker:
    t = ShieldTracker(ShieldLayout.load(Settings().layout_path))
    H_img = np.array([[px_per_mm, 0, ox], [0, px_per_mm, oy], [0, 0, 1]], np.float64)
    t.H_img_from_board, t.H_board_from_img = H_img, np.linalg.inv(H_img)
    return t


def test_shield_crop_keeps_every_port_and_marker():
    frame = np.zeros((2160, 3840, 3), np.uint8)       # phone held far away: board is small in a 4K frame
    shield = _tracker(4.0, 1500, 900)
    crop, cropped = crop_to_shield(frame, shield)
    assert cropped and crop.shape[0] * crop.shape[1] < 0.25 * frame.shape[0] * frame.shape[1]

    # Recover the crop offset by re-cropping a frame whose pixels encode their own coordinates.
    xs = np.tile(np.arange(3840, dtype=np.uint16), (2160, 1))
    ys = np.tile(np.arange(2160, dtype=np.uint16)[:, None], (1, 3840))
    coded = np.dstack([xs, ys, np.zeros_like(xs)])
    c, _ = crop_to_shield(coded, shield)
    x0, y0, x1, y1 = c[0, 0, 0], c[0, 0, 1], c[-1, -1, 0], c[-1, -1, 1]
    layout = shield.layout
    for p in list(layout.ports.values()):
        px, py = shield.to_image((p.x, p.y))
        assert x0 <= px <= x1 and y0 <= py <= y1, p.name
    for mx, my in layout.markers.values():
        px, py = shield.to_image((mx, my))
        assert x0 <= px <= x1 and y0 <= py <= y1


def test_close_up_is_sent_whole():
    frame = np.zeros((720, 1280, 3), np.uint8)
    layout = ShieldLayout.load(Settings().layout_path)
    xs = [p.x for p in layout.ports.values()] + [m[0] for m in layout.markers.values()]
    ys = [p.y for p in layout.ports.values()] + [m[1] for m in layout.markers.values()]
    scale = 1.2 * max(1280 / (max(xs) - min(xs)), 720 / (max(ys) - min(ys)))  # board overflows the view
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    crop, cropped = crop_to_shield(frame, _tracker(scale, 640 - scale * cx, 360 - scale * cy))
    assert not cropped and crop is frame


def test_no_homography_means_no_crop():
    frame = np.zeros((720, 1280, 3), np.uint8)
    t = ShieldTracker(ShieldLayout.load(Settings().layout_path))
    crop, cropped = crop_to_shield(frame, t)
    assert not cropped and crop is frame


def test_hand_crop_is_generous():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    pts = [(1800 + 10 * i, 1000 + 8 * i) for i in range(21)]   # a ~200 px hand
    hand = HandObservation(pts[8], pts, None, 0.0, "Right")
    crop, cropped = crop_to_hands(frame, [hand])
    assert cropped
    h, w = crop.shape[:2]
    assert w >= 200 + 2 * 200 and h >= 160 + 2 * 200  # 200x160 hand box grown by 200 px on every side
    assert crop_to_hands(frame, [])[1] is False


def test_box_clamps_to_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop, cropped = crop_to_box(frame, -50, -50, 60, 40, min_side_px=10)
    assert cropped and crop.shape[:2] == (40, 60)
