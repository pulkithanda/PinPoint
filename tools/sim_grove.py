"""
Step through a build in the terminal with fake fingertip input (no camera, no Chert).

    python -m tools.sim_grove [--project proximity_alert] [--no-verifier] [--live]

Default is snapshot guidance (photo on demand); --live simulates the phone on a stand.

Type engine commands (next, repeat, done, back, skip, help, voice mode, visual mode,
ready, what am I holding), a free-form question, or simulator commands:
    at D5             put the fingertip over port D5
    mm 30 25          put the fingertip at board coordinates (mm)
    gone              hand leaves the camera
    wait 2.5          advance the clock by 2.5 s (feeds frames meanwhile)
    id Buzzer         the vision LLM says the held module is "Buzzer" (or: id unknown)
    plug yes D7       plug check result: yes/no/unsure + port seen (optional)
    answer <text>     the vision LLM's answer to a question
    usb [dist]        Arduino starts sending readings (default dist varies)
    deadsensor        Arduino sends dist=0 (sensor not on its port)
    state             print the engine state
    quit
"""
from __future__ import annotations

import argparse
import random

from backend.config import Settings
from backend.guidance.engine import (Beep, Finished, GuidanceEngine, Observation, Project,
                                     RequestAnswer, RequestIdentify, RequestPlugCheck, Say, is_question,
                                     parse_command)
from backend.hardware.serial_monitor import FakeSerialMonitor
from backend.vision.shield import ShieldLayout
from backend.vision.verify import ModuleID, PlugCheck


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="proximity_alert")
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--live", action="store_true", help="live steering (phone on a stand)")
    args = ap.parse_args()

    s = Settings()
    layout = ShieldLayout.load(s.layout_path)
    project = Project.load(s.projects_dir / f"{args.project}.yaml")
    clock = FakeClock()
    eng = GuidanceEngine(project, layout, verifier_available=not args.no_verifier,
                         steering=args.live, clock=clock)
    serial = FakeSerialMonitor()
    finger: tuple[float, float] | None = None
    usb: int | None = None  # None = unplugged, -1 = varying, 0 = dead sensor, else fixed

    def show(actions):
        for a in actions:
            if isinstance(a, Say):
                tag = {"steer": "  >", "urgent": " !!", "info": "  *"}[a.kind]
                print(f"{tag} {a.text}")
            elif isinstance(a, RequestIdentify):
                print("  [vision LLM] identify module requested -> answer with: id <module>")
            elif isinstance(a, RequestPlugCheck):
                print(f"  [vision LLM] plug check: {a.module} in {a.ports}? -> answer with: plug yes|no|unsure [port]")
            elif isinstance(a, RequestAnswer):
                print(f"  [vision LLM] question: {a.question!r} -> answer with: answer <text>")
            elif isinstance(a, Finished):
                print("\n===== iMessage recap =====\n" + a.recap + "\n==========================")
            elif isinstance(a, Beep):
                pass

    def tick(dt: float = 0.1):
        clock.t += dt
        if usb is not None:
            serial.fake_connected = True
            d = random.randint(20, 150) if usb == -1 else usb
            serial.feed(f"dist={d} lcd=1", now=clock.t)
        show(eng.on_observation(Observation(clock.t, True, finger)))
        show(eng.on_serial_status(serial.status(clock.t)))

    print(__doc__)
    show(eng.start())
    while True:
        try:
            line = input(f"[{eng.phase.value} | finger={eng.finger_port}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split()
        head = parts[0].lower()
        if head == "quit":
            break
        if head == "at" and len(parts) > 1 and parts[1] in layout.ports:
            p = layout.ports[parts[1]]
            finger = (p.x + random.uniform(-1, 1), p.y + random.uniform(-1, 1))
            tick()
        elif head == "mm" and len(parts) == 3:
            finger = (float(parts[1]), float(parts[2]))
            tick()
        elif head == "gone":
            finger = None
            tick()
        elif head == "wait":
            for _ in range(int(float(parts[1] if len(parts) > 1 else 1) / 0.1)):
                tick()
        elif head == "id":
            name = " ".join(parts[1:]) or "unknown"
            match = next((m for m in project.modules if m.lower() == name.lower()), "unknown")
            show(eng.on_identify_result(ModuleID(match, True, "simulated")))
        elif head == "plug":
            verdict = {"yes": True, "no": False}.get(parts[1].lower() if len(parts) > 1 else "", None)
            seen = parts[2] if len(parts) > 2 else None
            show(eng.on_plug_result(PlugCheck(verdict, seen, "simulated")))
        elif head == "usb":
            usb = int(parts[1]) if len(parts) > 1 else -1
            for _ in range(40):
                tick()
        elif head == "deadsensor":
            usb = 0
            for _ in range(60):
                tick()
        elif head == "answer":
            show(eng.on_answer(" ".join(parts[1:]) or None))
        elif head == "state":
            print(eng.view_state())
        else:
            cmd = parse_command(line)
            if cmd:
                show(eng.on_command(cmd))
            elif is_question(line):
                show(eng.on_question(line))
            else:
                print("  (not understood)")
        if eng.phase.value == "done":
            print("Build complete.")
            break


if __name__ == "__main__":
    main()
