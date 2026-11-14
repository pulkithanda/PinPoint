"""
Download the MediaPipe gesture recognizer model (hand landmarks + gestures)
"""
from __future__ import annotations

import urllib.request

from backend.config import Settings
from backend.vision.hands import GESTURE_MODEL_URL


def main() -> None:
    dest = Settings().gesture_model
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already present: {dest}")
        return
    print(f"Downloading {GESTURE_MODEL_URL}\n -> {dest}")
    urllib.request.urlretrieve(GESTURE_MODEL_URL, dest)
    print(f"Done ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
