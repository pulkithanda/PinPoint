"""
Download the MediaPipe gesture recognizer model (hand landmarks + gestures)
"""
from __future__ import annotations

from backend.config import Settings
from backend.vision.hands import ensure_gesture_model


def main() -> None:
    dest = Settings().gesture_model
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"Already present: {dest}")
        return
    ensure_gesture_model(dest)
    print(f"Done ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
