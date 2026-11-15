"""
Compile and upload a plan's sketch with arduino-cli (https://arduino.github.io/arduino-cli/).

One-time setup: python -m tools.setup_arduino   (installs the AVR core and the kit's libraries)
Set PINPOINT_ARDUINO_CLI if arduino-cli isn't on PATH.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

FQBN = "arduino:avr:uno"
_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "arduino-cli" / "arduino-cli.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Arduino CLI" / "arduino-cli.exe",
    Path.home() / "bin" / "arduino-cli",
    Path("/usr/local/bin/arduino-cli"),
    Path("/opt/homebrew/bin/arduino-cli"),
]


def find_cli() -> str | None:
    env = os.environ.get("PINPOINT_ARDUINO_CLI")
    if env and Path(env).is_file():
        return env
    found = shutil.which("arduino-cli")
    if found:
        return found
    return next((str(p) for p in _CANDIDATES if p.is_file()), None)


@dataclass
class UploadResult:
    ok: bool
    stage: str            # "compile" | "upload" | "setup"
    output: str           # compiler/uploader output (for fixing or logging)
    message: str          # what Pinpoint says


def first_error(output: str) -> str:
    """The first compiler error line, shortened for logs."""
    for line in output.splitlines():
        if "error:" in line:
            return re.sub(r"^.*?error:\s*", "", line).strip()[:200]
    return output.strip().splitlines()[-1][:200] if output.strip() else "unknown error"


class Uploader:
    def __init__(self, work_dir: Path, cli: str | None = None, fqbn: str = FQBN, timeout_s: float = 180.0):
        self.cli = cli or find_cli()
        self.fqbn = fqbn
        self.work_dir = Path(work_dir)
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return self.cli is not None

    def _run(self, *args: str) -> tuple[bool, str]:
        try:
            p = subprocess.run([self.cli, *args], capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "")

    def write_sketch(self, name: str, code: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name) or "pinpoint"
        d = self.work_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{safe}.ino").write_text(code, encoding="utf-8")
        return d

    def compile(self, name: str, code: str) -> UploadResult:
        if not self.available:
            return UploadResult(False, "setup", "arduino-cli not found",
                                "I can't program the Arduino from this computer yet, because arduino-cli isn't installed.")
        d = self.write_sketch(name, code)
        ok, out = self._run("compile", "--fqbn", self.fqbn, str(d))
        if not ok:
            log.warning("Compile failed: %s", first_error(out))
        return UploadResult(ok, "compile", out, "The code is ready." if ok else "The code didn't compile.")

    def upload(self, name: str, code: str, port: str) -> UploadResult:
        res = self.compile(name, code)
        if not res.ok:
            return res
        d = self.work_dir / re.sub(r"[^A-Za-z0-9_]", "_", name)
        ok, out = self._run("upload", "-p", port, "--fqbn", self.fqbn, str(d))
        if ok:
            return UploadResult(True, "upload", out, "The code is on your Arduino.")
        log.warning("Upload failed: %s", out.strip()[-300:])
        msg = ("I couldn't send the code to the Arduino. Unplug the USB cable, plug it back in, "
               "and say ready to try again.")
        return UploadResult(False, "upload", out, msg)

    def wrong_board(self, port: str) -> str | None:
        """Name of the board on `port` if arduino-cli recognises it as something other than an UNO R3
        (e.g. "Arduino UNO Q"), else None. Unrecognised boards (e.g. CH340 clones) are allowed through."""
        if not self.available:
            return None
        ok, out = self._run("board", "list", "--format", "json")
        if not ok:
            return None
        try:
            data = json.loads(out[out.index("{"):] if "{" in out else out)
        except (ValueError, json.JSONDecodeError):
            return None
        entries = data.get("detected_ports", []) if isinstance(data, dict) else data
        for entry in entries or []:
            addr = ((entry.get("port") or {}).get("address") or "").upper()
            if addr != port.upper():
                continue
            boards = entry.get("matching_boards") or []
            if not boards or any(b.get("fqbn") == self.fqbn for b in boards):
                return None
            return boards[0].get("name") or boards[0].get("fqbn") or "unknown board"
        return None

    def install_libraries(self, libraries: list[str]) -> tuple[bool, str]:
        # Shipped inside the arduino:avr core. Servo, LiquidCrystal and Stepper are NOT (the IDE bundles
        # them, arduino-cli doesn't), so they get installed like any other library.
        builtin = {"Wire", "SPI", "EEPROM", "SoftwareSerial", "HID"}
        needed = [lib for lib in libraries if lib not in builtin]
        if not needed or not self.available:
            return True, ""
        return self._run("lib", "install", *needed)
