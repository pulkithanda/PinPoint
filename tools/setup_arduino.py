"""
One-time setup so Pinpoint can program the Arduino itself.

    python -m tools.setup_arduino

Needs arduino-cli: https://arduino.github.io/arduino-cli/latest/installation/
(Windows: `winget install ArduinoSA.CLI`; macOS: `brew install arduino-cli`).
Installs the Arduino AVR core (UNO) and the libraries the kit catalog allows, then compiles every
library project as a smoke test.
"""
from __future__ import annotations

import sys

from backend.build.board import load_kit
from backend.build.library import load_library
from backend.config import ROOT
from backend.hardware.uploader import Uploader, first_error


def main() -> int:
    up = Uploader(ROOT / "cache" / "sketches")
    if not up.available:
        print(__doc__)
        print("arduino-cli was not found. Install it, or set PINPOINT_ARDUINO_CLI to its path.")
        return 1
    print(f"Using {up.cli}")
    for args in (("core", "update-index"), ("core", "install", "arduino:avr")):
        ok, out = up._run(*args)
        print(f"arduino-cli {' '.join(args)}: {'ok' if ok else 'FAILED'}")
        if not ok:
            print(out)
            return 1
    ok, out = up.install_libraries(list(load_kit().libraries))
    print(f"libraries {', '.join(load_kit().libraries)}: {'ok' if ok else 'FAILED'}")
    if not ok:
        print(out)
    failed = 0
    for e in load_library():
        res = up.compile(e.plan.id, e.plan.sketch)
        print(f"  compile {e.plan.id:16} {'ok' if res.ok else 'FAILED: ' + first_error(res.output)}")
        failed += not res.ok
    print("All library sketches compile." if not failed else f"{failed} sketch(es) failed to compile.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
