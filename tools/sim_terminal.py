"""
Rehearse an Elegoo build in the terminal: no phone, no camera, no Chert.

    python -m tools.sim_terminal              # real Claude planner (needs ANTHROPIC_API_KEY)
    python -m tools.sim_terminal --offline    # library projects only, no API calls
    python -m tools.sim_terminal --compile    # also compile the sketch with arduino-cli at upload time

Type what the builder would say ("a light that turns on in the dark", "yes", "ready", "done",
"repeat", "is this the right way round?", "visual mode", "skip", "start over").

Stand in for the camera, the upload and the Arduino with ! commands:
    !setup [rotated]        board-setup photo result (rotated = USB socket not pointing left)
    !id <part_id|unknown>   part held up to the camera, e.g. !id resistor_10k
    !ok                     placement / final check: correct
    !bad <what's wrong>     placement / final check: a problem, e.g. !bad The LED is backwards.
    !unsure                 placement / final check: couldn't tell
    !answer <text>          Claude's answer to a question
    !upload [fail]          upload result
    !usb                    Arduino sends changing readings of the test key (as if it works)
    !serial <line>          one raw serial line, e.g. !serial dist=12
    !plan                   print the plan (with its validation result)
    !state                  engine state
    !quit
"""
from __future__ import annotations

import argparse
import json
import random

from backend.build.board import validate
from backend.build.engine import (BuildEngine, Finished, RequestPartID, RequestPlacementCheck, RequestPlan,
                                  RequestReview, RequestSetupCheck, RequestUpload)
from backend.build.planner import OfflinePlanner, Planner
from backend.config import ROOT, Settings
from backend.guidance.engine import RequestAnswer, Say
from backend.hardware.serial_monitor import FakeSerialMonitor
from backend.hardware.uploader import Uploader, UploadResult, first_error
from backend.vision.verify import BoardReview, BoardSetup, ModuleID, PlacementCheck, claude_available


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="library projects only (no Claude)")
    ap.add_argument("--compile", action="store_true", help="compile the sketch with arduino-cli on upload")
    ap.add_argument("--visual", action="store_true", help="start in visual (captions) mode")
    args = ap.parse_args()

    s = Settings()
    online = not args.offline and claude_available()
    planner = Planner(s.claude_model, s.planner_effort, save_dir=ROOT / "cache" / "plans") if online \
        else OfflinePlanner()
    uploader = Uploader(ROOT / "cache" / "sketches") if args.compile else None
    clock = FakeClock()
    eng = BuildEngine(mode="visual" if args.visual else "", verifier_available=True,
                      uploader_available=True, clock=clock)
    serial = FakeSerialMonitor()
    pending: dict[str, object] = {}

    def show(actions):
        for a in actions:
            if isinstance(a, Say):
                tag = {"steer": "  >", "urgent": " !!", "info": "  *"}.get(a.kind, "  *")
                print(f"{tag} {a.text}")
            elif isinstance(a, RequestPlan):
                print(f"  [planner] designing: {a.request!r} ({'Claude' if online else 'library only'})...")
                res = planner.plan(a.request, a.previous)
                if res.plan:
                    print(f"  [planner] {res.source}: {res.plan.title}, {len(res.plan.steps)} steps, "
                          f"{res.plan.connection_count} connections (attempts: {res.attempts})")
                else:
                    print(f"  [planner] no plan ({'; '.join(res.errors or []) or 'declined'})")
                show(eng.on_plan_result(res))
            elif isinstance(a, RequestSetupCheck):
                pending["kind"] = "setup"
                print("  [camera] board-setup photo -> answer with: !setup [rotated]")
            elif isinstance(a, RequestPartID):
                pending["kind"] = "identify"
                print(f"  [camera] which part is held up? expected {a.expected} -> !id <part_id>")
            elif isinstance(a, RequestPlacementCheck):
                pending["kind"] = "placement"
                print(f"  [camera] check step {a.step_index + 1}: {a.expected}\n           -> !ok | !bad <problem> | !unsure")
            elif isinstance(a, RequestReview):
                pending["kind"] = "review"
                print("  [camera] final check of the whole board -> !ok | !bad <problem> | !unsure")
            elif isinstance(a, RequestAnswer):
                pending["kind"] = "answer"
                print(f"  [camera] question {a.question!r} -> !answer <text>")
            elif isinstance(a, RequestUpload):
                pending["kind"] = "upload"
                if uploader is not None:
                    res = uploader.compile(a.plan.id, a.plan.sketch)
                    print(f"  [arduino-cli] compile: {'ok' if res.ok else 'FAILED: ' + first_error(res.output)}")
                print("  [upload] -> !upload | !upload fail")
            elif isinstance(a, Finished):
                print("\n===== iMessage recap =====\n" + a.recap + "\n==========================")

    def tick(dt: float = 0.5):
        clock.t += dt
        show(eng.on_tick(clock.t))
        if eng.plan is not None:
            key = eng.plan.test.key
            show(eng.on_serial(serial.receiving(clock.t), serial.values(key, clock.t)))

    print(__doc__)
    show(eng.start())
    while eng.phase.value != "done":
        try:
            line = input(f"[{eng.phase.value} | step {eng.step_idx + 1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if not line.startswith("!"):
            show(eng.on_utterance(line))
            tick()
            continue
        head, _, rest = line[1:].partition(" ")
        head, rest = head.lower(), rest.strip()
        if head == "quit":
            break
        elif head == "setup":
            show(eng.on_setup_result(BoardSetup("AE", "left", "-", rest != "rotated", "simulated")))
        elif head == "id":
            show(eng.on_identify_result(ModuleID(rest or "unknown", True, "simulated")))
        elif head in ("ok", "bad", "unsure"):
            ok = {"ok": True, "bad": False}.get(head)
            if pending.get("kind") == "review":
                show(eng.on_review_result(BoardReview(ok, [rest] if rest else [], "simulated")))
            else:
                show(eng.on_placement_result(PlacementCheck(ok, rest, "simulated")))
        elif head == "answer":
            show(eng.on_answer(rest or None))
        elif head == "upload":
            ok = rest != "fail"
            show(eng.on_upload_result(UploadResult(ok, "upload", "", "Uploaded." if ok else
                                                   "I couldn't send the code. Unplug and replug the USB, then say ready.")))
        elif head == "usb":
            key = eng.plan.test.key if eng.plan else "value"
            for i in range(20):
                clock.t += 0.2
                serial.feed(f"{key}={random.choice([0, 1]) if i % 2 else random.randint(10, 900)}", now=clock.t)
            tick(0.1)
        elif head == "serial":
            clock.t += 0.2
            serial.feed(rest, now=clock.t)
            tick(0.1)
        elif head == "plan":
            if eng.plan is None:
                print("  (no plan yet)")
            else:
                print(json.dumps(eng.plan.to_dict(), indent=2))
                errs = validate(eng.plan)
                print("  validation:", "OK" if not errs else errs)
        elif head == "state":
            print(eng.view_state())
        else:
            print("  (unknown ! command)")
    if eng.phase.value == "done":
        print("Build complete.")


if __name__ == "__main__":
    main()
