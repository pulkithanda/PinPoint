"""
Runtime settings, read from environment variables (or a .env file in the repo root).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ('"', "'") and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]       # quoted: keep everything inside
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()  # drop inline comments
        os.environ.setdefault(key.strip(), value)

_load_dotenv()

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

@dataclass
class Settings:
    # Inputs
    camera_index: int = field(default_factory=lambda: int(_env("PINPOINT_CAMERA", "0")))
    serial_port: str = field(default_factory=lambda: _env("PINPOINT_SERIAL_PORT", "auto"))  # "auto", "none", or e.g. COM5
    serial_baud: int = field(default_factory=lambda: int(_env("PINPOINT_SERIAL_BAUD", "9600")))

    # Files
    layout_path: Path = field(default_factory=lambda: Path(_env("PINPOINT_LAYOUT", str(BACKEND / "vision" / "shield_layout.yaml"))))
    projects_dir: Path = BACKEND / "projects"
    gesture_model: Path = field(default_factory=lambda: Path(_env("PINPOINT_GESTURE_MODEL", str(BACKEND / "models" / "gesture_recognizer.task"))))
    recap_dir: Path = ROOT / "recaps"

    # Vision LLM (slow path). Disabled automatically if no Anthropic credentials are configured.
    verify_enabled: bool = field(default_factory=lambda: _env_bool("PINPOINT_VERIFY", True))
    claude_model: str = field(default_factory=lambda: _env("PINPOINT_CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: _env("PINPOINT_CLAUDE_EFFORT", "low"))

    # Guidance tuning
    speech_min_interval_s: float = 0.8   # one steering instruction per ~0.8 s
    speech_repeat_after_s: float = 4.0   # repeat an unchanged instruction at most this often
    hand_gone_check_s: float = 2.0       # hand leaves the target for 2 s -> plug check
    smoothing_frames: int = 5
    shield_recheck_s: float = 2.0
    fingertip_landmark: int = 8          # MediaPipe index fingertip

    # Modes
    mode: str = field(default_factory=lambda: _env("PINPOINT_MODE", ""))  # "", "voice" or "visual"
    # "snapshot": the phone stays on the FaceTime call but is put down; Pinpoint asks the builder
    #             to hold it up when it needs to look, and auto-captures a sharp, steady frame.
    # "live":     the phone sits on an overhead stand and Pinpoint steers the hand every frame.
    guidance: str = field(default_factory=lambda: _env("PINPOINT_GUIDANCE", "snapshot"))
    capture_timeout_s: float = 15.0
    sonify: bool = field(default_factory=lambda: _env_bool("PINPOINT_SONIFY", False))

    # Speech in (Chert has no transcription): local faster-whisper on the caller's audio
    stt_model: str = field(default_factory=lambda: _env("PINPOINT_STT_MODEL", "base.en"))

    @property
    def live_steering(self) -> bool:
        return self.guidance == "live"
