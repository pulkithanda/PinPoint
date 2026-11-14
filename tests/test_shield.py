"""
ArUco -> homography -> port map, on a synthetic camera image with perspective
"""
import cv2
import numpy as np
import pytest

from backend.config import Settings
from backend.vision.shield import ARUCO_DICT_ID, ShieldLayout, ShieldTracker

PX = 8.0      # board render scale, px per mm
OFF = 20.0    # mm margin around the markers


@pytest.fixture(scope="module")
def layout():
    return ShieldLayout.load(Settings().layout_path)


def render_board(layout):
    xs = [m[0] for m in layout.markers.values()]
    ys = [m[1] for m in layout.markers.values()]
    w = int((max(xs) + 2 * OFF) * PX)
    h = int((max(ys) + 2 * OFF) * PX)
    img = np.full((h, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    side = int(layout.marker_size_mm * PX)
    for mid, (cx, cy) in layout.markers.items():
        m = cv2.aruco.generateImageMarker(d, mid, side)
        x0 = int(round((cx + OFF) * PX - side / 2))
        y0 = int(round((cy + OFF) * PX - side / 2))
        img[y0:y0 + side, x0:x0 + side] = m
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def board_to_render(pt):
    return ((pt[0] + OFF) * PX, (pt[1] + OFF) * PX)


def test_ports_map_through_perspective(layout):
    board = render_board(layout)
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[80, 60], [w + 40, 30], [w + 90, h + 70], [30, h + 40]])  # tilted camera
    P = cv2.getPerspectiveTransform(src, dst)
    cam = cv2.warpPerspective(board, P, (w + 200, h + 200), borderValue=(255, 255, 255))

    tracker = ShieldTracker(layout)
    assert tracker.update(cam, now=0.0)
    for name, port in layout.ports.items():
        px = cv2.perspectiveTransform(np.float32([[board_to_render((port.x, port.y))]]), P)[0, 0]
        mm = tracker.to_board(tuple(px))
        assert abs(mm[0] - port.x) < 1.5 and abs(mm[1] - port.y) < 1.5, name
        assert layout.nearest_port(mm)[0].name == name
        back = tracker.to_image((port.x, port.y))
        assert abs(back[0] - px[0]) < 3 and abs(back[1] - px[1]) < 3


def test_three_markers_enough_and_cache_kept(layout):
    board = render_board(layout)
    tracker = ShieldTracker(layout, recheck_s=0.0)
    hidden = board.copy()
    cx, cy = board_to_render(layout.markers[2])
    s = layout.marker_size_mm * PX
    cv2.rectangle(hidden, (int(cx - s), int(cy - s)), (int(cx + s), int(cy + s)), (200, 170, 150), -1)  # a "hand"
    assert tracker.update(hidden, 0.0)
    assert sorted(tracker.visible_ids) == [0, 1, 3]
    # Only zero markers visible -> keep the cached homography.
    blank = np.full_like(board, 255)
    H = tracker.H_board_from_img.copy()
    assert tracker.update(blank, 5.0)
    assert np.allclose(H, tracker.H_board_from_img)


def test_tactile_hints(layout):
    assert "nearest you" in layout.tactile_hint("I2C_1")
    assert "far edge" in layout.tactile_hint("D2")
    assert "second port from the left" in layout.tactile_hint("D7")
