"""Generate markers/aruco_4x4.pdf: four 4x4_50 ArUco markers (ids 0-3) at true size on A4/Letter.

Print at 100% scale ("actual size", NOT "fit to page"), cut out, and tape upright on the
Base Shield corners: 0 = far-left, 1 = far-right, 2 = near-right, 3 = near-left.
Then measure the marker centre positions into backend/vision/shield_layout.yaml.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw

from backend.config import ROOT
from backend.vision.shield import ARUCO_DICT_ID

DPI = 300
NAMES = {0: "0 far-left", 1: "1 far-right", 2: "2 near-right", 3: "3 near-left"}


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mm", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "markers" / "aruco_4x4.pdf"))
    args = ap.parse_args()

    page = Image.new("L", (mm_to_px(210), mm_to_px(279)), 255)  # fits both A4 and Letter
    draw = ImageDraw.Draw(page)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    side = mm_to_px(args.size_mm)
    quiet = mm_to_px(4)  # white border so the detector can find the black edge
    draw.text((mm_to_px(15), mm_to_px(12)),
              f"Pinpoint ArUco markers - DICT_4X4_50, {args.size_mm:g} mm. Print at 100% (actual size).", fill=0)
    for i in range(4):
        marker = cv2.aruco.generateImageMarker(dictionary, i, side)
        x = mm_to_px(25) + (i % 2) * mm_to_px(80)
        y = mm_to_px(35) + (i // 2) * mm_to_px(60)
        draw.rectangle([x - quiet, y - quiet, x + side + quiet, y + side + quiet], outline=0, width=1)
        page.paste(Image.fromarray(marker), (x, y))
        draw.text((x - quiet, y + side + quiet + mm_to_px(2)), f"id {NAMES[i]}  (this side up)", fill=0)
    # 50 mm scale bar to check print scale
    y = mm_to_px(170)
    draw.line([mm_to_px(25), y, mm_to_px(75), y], fill=0, width=3)
    draw.text((mm_to_px(25), y + mm_to_px(3)), "50 mm - measure to check the print scale", fill=0)
    page.save(args.out, "PDF", resolution=DPI)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
