import numpy as np
import pytest

from backend.build.board import Orientation
from backend.build.visual_guidance import draw_breadboard_targets, hole_position_mm
from backend.config import Settings
from backend.vision.shield import ShieldLayout, ShieldTracker


MARKERS = {0: (0.0, 0.0), 1: (70.0, 0.0), 2: (70.0, 56.0), 3: (0.0, 56.0)}


def test_hole_positions_follow_the_standard_830_board_grid():
    orientation = Orientation()
    a1 = hole_position_mm("A1", orientation, MARKERS)
    a30 = hole_position_mm("A30", orientation, MARKERS)
    e1 = hole_position_mm("E1", orientation, MARKERS)
    f1 = hole_position_mm("F1", orientation, MARKERS)
    j1 = hole_position_mm("J1", orientation, MARKERS)

    assert a1 == pytest.approx((-43.74, 97.91))
    assert a30 == pytest.approx((29.92, 97.91))
    assert e1 == pytest.approx((-43.74, 87.75))
    assert f1 == pytest.approx((-43.74, 80.13))
    assert j1 == pytest.approx((-43.74, 69.97))
    assert hole_position_mm("H64", orientation, MARKERS) is None


def test_hole_positions_follow_detected_board_orientation():
    orientation = Orientation(near_half="FJ", row1="right")

    assert hole_position_mm("F1", orientation, MARKERS) == pytest.approx((113.74, 97.91))
    assert hole_position_mm("A1", orientation, MARKERS) == pytest.approx((113.74, 80.13))


def test_draw_targets_uses_marker_homography():
    tracker = ShieldTracker(ShieldLayout.load(Settings().layout_path))
    tracker.H_img_from_board = np.array([[4.0, 0, 300], [0, 4.0, 200], [0, 0, 1]], np.float64)
    tracker.H_board_from_img = np.linalg.inv(tracker.H_img_from_board)
    frame = np.zeros((720, 1280, 3), np.uint8)

    out = draw_breadboard_targets(frame, tracker, ["A1", "J1"], Orientation(), (350, 500))

    assert out.shape == frame.shape
    assert np.count_nonzero(out) > 0
    assert np.count_nonzero(frame) == 0
