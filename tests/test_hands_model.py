from pathlib import Path

from backend.vision import hands


def test_ensure_mediapipe_runtime_reports_missing_linux_egl(monkeypatch):
    monkeypatch.setattr(hands.sys, "platform", "linux")

    def missing_egl(_library):
        raise OSError("not found")

    monkeypatch.setattr(hands.ctypes, "CDLL", missing_egl)

    try:
        hands.ensure_mediapipe_runtime()
    except RuntimeError as exc:
        assert "libEGL.so.1, libGLESv2.so.2" in str(exc)
        assert "apt-get install -y libegl1 libgles2" in str(exc)
    else:
        raise AssertionError("Missing graphics libraries should stop MediaPipe startup")


def test_ensure_mediapipe_runtime_checks_each_linux_library(monkeypatch):
    monkeypatch.setattr(hands.sys, "platform", "linux")
    loaded = []

    def missing_gles(library):
        loaded.append(library)
        if library == "libGLESv2.so.2":
            raise OSError("not found")
        return object()

    monkeypatch.setattr(hands.ctypes, "CDLL", missing_gles)

    try:
        hands.ensure_mediapipe_runtime()
    except RuntimeError as exc:
        assert "libGLESv2.so.2" in str(exc)
        assert "libEGL.so.1" not in str(exc)
    else:
        raise AssertionError("Missing GLES should stop MediaPipe startup")
    assert loaded == ["libEGL.so.1", "libGLESv2.so.2"]


def test_ensure_gesture_model_downloads_to_configured_path(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "gesture_recognizer.task"
    requested = {}

    def fake_urlretrieve(url, path):
        requested["url"] = url
        Path(path).write_bytes(b"model")

    monkeypatch.setattr(hands.urllib.request, "urlretrieve", fake_urlretrieve)

    assert hands.ensure_gesture_model(dest) == dest
    assert dest.read_bytes() == b"model"
    assert requested["url"] == hands.GESTURE_MODEL_URL
    assert list(dest.parent.iterdir()) == [dest]
