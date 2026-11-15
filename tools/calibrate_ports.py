"""
Measure port positions on OUR shield: click each port on a rectified top-down view.

    python -m tools.calibrate_ports [--camera 0]

1. Tape the 4 markers on the shield corners and enter their centre positions (mm,
   relative to marker 0) in backend/vision/shield_layout.yaml.
2. Run this, put the shield under the camera, and click the CENTRE of each port's
   socket in the order prompted. u = undo, s = save, q = quit without saving.
row/col values are kept from the existing layout (they drive the "right two ports" wording).
"""
from __future__ import annotations

import argparse
import time

import cv2
import yaml

from backend.config import Settings
from backend.vision.shield import ShieldLayout, ShieldTracker

PX_PER_MM = 8.0
MARGIN_MM = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=Settings().camera_index)
    args = ap.parse_args()
    path = Settings().layout_path
    layout = ShieldLayout.load(path)
    tracker = ShieldTracker(layout, recheck_s=0.0)
    names = list(layout.ports)
    clicks: dict[str, tuple[float, float]] = {}
    min_x = min(m[0] for m in layout.markers.values())
    min_y = min(m[1] for m in layout.markers.values())

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(names):
            mm = (x / PX_PER_MM - MARGIN_MM + min_x, y / PX_PER_MM - MARGIN_MM + min_y)
            clicks[names[len(clicks)]] = (round(mm[0], 1), round(mm[1], 1))

    cap = cv2.VideoCapture(args.camera)
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        tracker.update(frame, time.monotonic(), force=True)
        view = tracker.warp_top_down(frame, PX_PER_MM, MARGIN_MM)
        if view is None:
            view = frame.copy()
            cv2.putText(view, f"Markers seen: {tracker.visible_ids} (need 3+)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for n, (mx, my) in clicks.items():
                p = (int((mx - min_x + MARGIN_MM) * PX_PER_MM), int((my - min_y + MARGIN_MM) * PX_PER_MM))
                cv2.circle(view, p, 5, (0, 255, 0), -1)
                cv2.putText(view, n, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            msg = f"Click port {names[len(clicks)]}" if len(clicks) < len(names) else "All done - press s to save"
            cv2.putText(view, msg + "   (u undo, s save, q quit)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and clicks:
            clicks.pop(list(clicks)[-1])
        if key == ord("s") and clicks:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            for n, (x, y) in clicks.items():
                data["ports"][n]["x"], data["ports"][n]["y"] = float(x), float(y)
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            print(f"Saved {len(clicks)} port positions to {path}")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
