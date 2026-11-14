"""Shield mapping: ArUco markers on the Base Shield corners -> homography -> fixed port map.

Pixels in the camera image are mapped into the "board frame" (millimetres, see
shield_layout.yaml), where every port has a fixed, measured position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


@dataclass(frozen=True)
class Port:
    name: str
    x: float
    y: float
    row: int
    col: int

    @property
    def spoken(self) -> str:
        """How TTS should say the port name ("I2C_1" -> "I 2 C one")."""
        if self.name.startswith("I2C"):
            return "an I 2 C port"
        return self.name.replace("_", " ")


@dataclass
class ShieldLayout:
    marker_size_mm: float
    markers: dict[int, tuple[float, float]]
    ports: dict[str, Port]
    tactile: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ShieldLayout":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        markers = {int(k): (float(v[0]), float(v[1])) for k, v in data["markers"].items()}
        ports = {
            name: Port(name, float(p["x"]), float(p["y"]), int(p["row"]), int(p["col"]))
            for name, p in data["ports"].items()
        }
        return cls(float(data["marker_size_mm"]), markers, ports, data.get("tactile") or {})

    # --- geometry helpers -------------------------------------------------
    def marker_corners_mm(self, marker_id: int) -> np.ndarray:
        """Corners of an upright-taped marker in board mm, in ArUco order (TL, TR, BR, BL)."""
        cx, cy = self.markers[marker_id]
        h = self.marker_size_mm / 2
        return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], np.float32)

    @property
    def pitch_mm(self) -> float:
        """Typical distance between neighbouring ports (used for 'on/off the board' thresholds)."""
        pts = np.array([[p.x, p.y] for p in self.ports.values()])
        if len(pts) < 2:
            return 12.0
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[d == 0] = np.inf
        return float(np.median(d.min(axis=1)))

    @property
    def n_rows(self) -> int:
        return max(p.row for p in self.ports.values()) + 1

    def nearest_port(self, pt_mm: tuple[float, float], among: list[str] | None = None) -> tuple[Port, float]:
        names = among or list(self.ports)
        best = min(names, key=lambda n: (self.ports[n].x - pt_mm[0]) ** 2 + (self.ports[n].y - pt_mm[1]) ** 2)
        p = self.ports[best]
        return p, float(np.hypot(p.x - pt_mm[0], p.y - pt_mm[1]))

    def tactile_hint(self, name: str) -> str:
        """Touch-based description of a port, for blind builders and the place-then-check fallback."""
        if name in self.tactile:
            return self.tactile[name]
        p = self.ports[name]
        rows_from_near = self.n_rows - 1 - p.row
        if rows_from_near == 0:
            row_txt = "the row along the edge nearest you"
        elif p.row == 0:
            row_txt = "the row along the far edge"
        else:
            row_txt = f"the {_ORDINALS[rows_from_near]} row back from the edge nearest you"
        return f"It's in {row_txt}, the {_ORDINALS[p.col]} port from the left."


class ShieldTracker:
    """Detects the corner markers and keeps a cached homography (image px <-> board mm)."""

    def __init__(self, layout: ShieldLayout, recheck_s: float = 2.0, min_markers: int = 3):
        self.layout = layout
        self.recheck_s = recheck_s
        self.min_markers = min_markers
        self.H_board_from_img: np.ndarray | None = None
        self.H_img_from_board: np.ndarray | None = None
        self.last_update_t: float = -1e9
        self.visible_ids: list[int] = []
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), cv2.aruco.DetectorParameters()
        )

    @property
    def ready(self) -> bool:
        return self.H_board_from_img is not None

    def update(self, frame_bgr: np.ndarray, now: float, force: bool = False) -> bool:
        """Re-detect markers if due. Returns True if the homography is usable."""
        if not force and self.ready and now - self.last_update_t < self.recheck_s:
            return True
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        corners, ids, _ = self._detector.detectMarkers(gray)
        self.visible_ids = [] if ids is None else [int(i) for i in ids.flatten()]
        src, dst = [], []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in self.layout.markers:
                    src.append(c.reshape(4, 2))
                    dst.append(self.layout.marker_corners_mm(int(i)))
        # Hand hides markers: keep the cached homography unless 3+ markers are visible.
        if len(src) >= self.min_markers:
            src_pts = np.concatenate(src).astype(np.float32)
            dst_pts = np.concatenate(dst).astype(np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if H is not None:
                self.H_board_from_img = H
                self.H_img_from_board = np.linalg.inv(H)
                self.last_update_t = now
        return self.ready

    def to_board(self, pt_px: tuple[float, float]) -> tuple[float, float] | None:
        if self.H_board_from_img is None:
            return None
        p = cv2.perspectiveTransform(np.array([[pt_px]], np.float32), self.H_board_from_img)[0, 0]
        return float(p[0]), float(p[1])

    def to_image(self, pt_mm: tuple[float, float]) -> tuple[int, int] | None:
        if self.H_img_from_board is None:
            return None
        p = cv2.perspectiveTransform(np.array([[pt_mm]], np.float32), self.H_img_from_board)[0, 0]
        return int(round(p[0])), int(round(p[1]))

    def port_pixel(self, name: str) -> tuple[int, int] | None:
        p = self.layout.ports[name]
        return self.to_image((p.x, p.y))

    def px_per_mm(self) -> float:
        """Approximate image scale near the board centre (for drawing circles of the right size)."""
        if self.H_img_from_board is None:
            return 4.0
        xs = [p.x for p in self.layout.ports.values()]
        ys = [p.y for p in self.layout.ports.values()]
        c = (float(np.mean(xs)), float(np.mean(ys)))
        a, b = self.to_image(c), self.to_image((c[0] + 10.0, c[1]))
        return max(1.0, float(np.hypot(b[0] - a[0], b[1] - a[1])) / 10.0)

    def warp_top_down(self, frame_bgr: np.ndarray, px_per_mm: float = 8.0, margin_mm: float = 10.0) -> np.ndarray | None:
        """Rectified top-down view of the board (used by calibration and the dashboard)."""
        if self.H_board_from_img is None:
            return None
        xs = [m[0] for m in self.layout.markers.values()]
        ys = [m[1] for m in self.layout.markers.values()]
        T = np.array([[px_per_mm, 0, (margin_mm - min(xs)) * px_per_mm],
                      [0, px_per_mm, (margin_mm - min(ys)) * px_per_mm],
                      [0, 0, 1]])
        w = int((max(xs) - min(xs) + 2 * margin_mm) * px_per_mm)
        h = int((max(ys) - min(ys) + 2 * margin_mm) * px_per_mm)
        return cv2.warpPerspective(frame_bgr, T @ self.H_board_from_img, (w, h))
