"""
Step engine: deterministic state machine that turns observations into instructions

Inputs (all plain data, so this runs identically on a webcam, a FaceTime call,
or a terminal simulation):
  * on_observation(Observation)  - every frame: smoothed fingertip in board mm, gestures
  * on_command(str)              - "next", "repeat", "what am I holding?" ...
  * on_identify_result / on_plug_result / on_serial_status - slow-path results

Outputs: a list of actions (Say, Beep, RequestIdentify, RequestPlugCheck, Finished)
that the session routes to speech, overlays, the vision LLM, or iMessage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union

import yaml

from backend.hardware.serial_monitor import SerialStatus, diagnose
from backend.vision.shield import ShieldLayout
from backend.vision.verify import ModuleID, PlugCheck

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


# ---------------------------------------------------------------- project files
@dataclass
class Step:
    id: str
    module: str
    spoken_module: str
    targets: list[str]
    find: str
    say: str
    show: str
    tip: str = ""


@dataclass
class Project:
    id: str
    title: str
    summary: str
    modules: list[str]
    steps: list[Step]
    final_check: dict

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        steps = [Step(**s) for s in d["steps"]]
        return cls(d["id"], d["title"], d.get("summary", ""), d.get("modules", []), steps,
                   d.get("final_check") or {"type": "none"})

    def validate(self, layout: ShieldLayout) -> None:
        for s in self.steps:
            bad = [t for t in s.targets if t not in layout.ports]
            if bad:
                raise ValueError(f"step {s.id}: unknown ports {bad} (not in shield layout)")


# ---------------------------------------------------------------- I/O types
class Phase(str, Enum):
    MODE_SELECT = "mode_select"
    FIND_MODULE = "find_module"
    GUIDE = "guide"
    CHECKING = "checking"
    ELECTRICAL = "electrical"
    DONE = "done"


@dataclass
class Observation:
    t: float
    shield_ok: bool
    fingertip_mm: tuple[float, float] | None   # smoothed; None if no hand
    gesture: str | None = None                  # debounced event: "Thumb_Up" / "Open_Palm"


@dataclass
class Say:
    text: str
    kind: str = "info"      # "steer" = droppable/rate-limited, "info" = always delivered, "urgent" = jump the queue
    key: str | None = None  # identical keys are de-duplicated by the speech layer


@dataclass
class Beep:
    closeness: float        # 0 = far, 1 = on target (parking-sensor style)


@dataclass
class RequestIdentify:
    candidates: list[str]


@dataclass
class RequestPlugCheck:
    module: str
    ports: list[str]


@dataclass
class RequestAnswer:
    question: str
    context: str


@dataclass
class Finished:
    recap: str


Action = Union[Say, Beep, RequestIdentify, RequestPlugCheck, RequestAnswer, Finished]

_QUESTION_WORDS = {"what", "where", "which", "how", "why", "is", "are", "can", "does", "do",
                   "should", "did", "will", "could", "would", "who"}


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").replace("!", " ").split())


def is_question(text: str) -> bool:
    t = _norm(text)
    return bool(t) and (text.strip().endswith("?") or t.split()[0] in _QUESTION_WORDS)


def parse_command(text: str) -> str | None:
    """Map a transcript (or typed iMessage) to an engine command. Longer free-form
    sentences return None so they can be answered as questions instead."""
    t = _norm(text)
    if not t:
        return None
    phrases = [
        ("voice", ("voice mode",)),
        ("visual", ("visual mode", "deaf mode", "caption mode")),
        ("holding", ("what am i holding", "what is this", "what's this", "what is it")),
        ("back", ("go back",)),
        ("repeat", ("say again", "say that again", "repeat that")),
    ]
    for cmd, keys in phrases:
        if t in keys:  # exact, so "what is this cable for?" stays a question
            return cmd
    if len(t.split()) > 4:
        return None
    words = [
        ("voice", ("voice",)),
        ("visual", ("visual", "captions")),
        ("holding", ("holding", "identify")),
        ("back", ("back", "previous", "undo")),
        ("skip", ("skip",)),
        ("repeat", ("repeat", "again", "what")),
        ("help", ("help", "describe", "hint")),
        ("done", ("done", "plugged", "it's in", "finished", "check", "clicked")),
        ("next", ("next", "continue", "ready", "got it", "okay", "ok")),
    ]
    for cmd, keys in words:
        if any(k == t or (k != "what" and (t.startswith(k + " ") or f" {k} " in f" {t} ")) for k in keys):
            return cmd
    return None


# ---------------------------------------------------------------- engine
class GuidanceEngine:
    def __init__(self, project: Project, layout: ShieldLayout, *, mode: str = "",
                 verifier_available: bool = True, steering: bool = True,
                 sonify: bool = False, hand_gone_check_s: float = 2.0, clock=time.monotonic):
        """steering=True: phone on a stand, steer the fingertip every frame ("live").
        steering=False: snapshot mode - describe ports by touch, take a photo on demand to check."""
        project.validate(layout)
        self.project = project
        self.layout = layout
        self.mode = mode if mode in ("voice", "visual") else ""
        self.verifier_available = verifier_available
        self.steering = steering
        self.sonify = sonify
        self.hand_gone_check_s = hand_gone_check_s
        self.clock = clock

        self.phase = Phase.MODE_SELECT
        self.step_idx = 0
        self.occupied: dict[str, str] = {}      # port -> module
        self.caption = ""
        self.finger_port: str | None = None
        self.events: list[tuple[float, str, str]] = []  # (t, kind, text) for the recap
        self.started_t = clock()

        self._pitch = layout.pitch_mm
        self._last_instruction = ""             # what "repeat" says
        self._last_steer_key: str | None = None
        self._last_steer_fp: tuple[float, float] | None = None
        self._was_at_target_t: float | None = None
        self._away_since: float | None = None
        self._last_identify_t = -1e9
        self._identify_pending = False
        self._identify_fails = 0
        self._check_pending = False
        self._check_retries = 0
        self._throttle: dict[str, float] = {}
        self._phase_t = clock()

    # ------------------------------------------------------------ helpers
    @property
    def step(self) -> Step | None:
        return self.project.steps[self.step_idx] if self.step_idx < len(self.project.steps) else None

    def free_targets(self) -> list[str]:
        s = self.step
        if not s:
            return []
        free = [p for p in s.targets if p not in self.occupied]
        return free or list(s.targets)

    def _target_label(self) -> str:
        t = self.free_targets()
        return self.layout.ports[t[0]].spoken if len(t) == 1 else "an I 2 C port" if all(x.startswith("I2C") for x in t) else " or ".join(t)

    def _say(self, text: str, kind: str = "info", key: str | None = None, remember: bool = True) -> Say:
        self.caption = text
        if remember and kind != "steer":
            self._last_instruction = text
        return Say(text, kind, key)

    def _throttled(self, key: str, every_s: float, now: float) -> bool:
        if now - self._throttle.get(key, -1e9) < every_s:
            return True
        self._throttle[key] = now
        return False

    def _log(self, kind: str, text: str) -> None:
        self.events.append((self.clock() - self.started_t, kind, text))

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_t = self.clock()

    # ------------------------------------------------------------ public API
    def start(self) -> list[Action]:
        if self.mode:
            return self._begin_project()
        self._enter(Phase.MODE_SELECT)
        return [self._say("Hi, I'm Pinpoint. Say 'voice mode' for spoken directions, "
                          "or show a thumbs-up for visual mode with captions.")]

    def on_command(self, cmd: str) -> list[Action]:
        if cmd in ("voice", "visual"):
            switching = self.phase != Phase.MODE_SELECT
            self.mode = cmd
            self._log("mode", f"{cmd} mode")
            if switching:
                return [self._say(f"Switched to {cmd} mode.")]
            return self._begin_project()
        if cmd == "repeat":
            return [Say(self._last_instruction or self.caption, "urgent")]
        if cmd == "holding":
            return self._request_identify(force=True)
        if self.phase == Phase.MODE_SELECT:
            if cmd in ("next", "done"):
                self.mode = "voice"
                return self._begin_project()
            return []
        if cmd == "help" and self.step:
            return [self._say(f"{self.step.say} {self.layout.tactile_hint(self.free_targets()[0])}")]
        if cmd == "back":
            return self._go_back()
        if cmd == "skip":
            return self._advance(skipped=True)
        if cmd in ("next", "done"):
            if self.phase == Phase.FIND_MODULE:
                if not self.steering and self.verifier_available and self._identify_fails < 2:
                    return self._request_identify(force=True)  # snapshot mode: "ready" -> take a photo
                return self._enter_guide(confirmed_by_user=True)
            if self.phase == Phase.GUIDE:
                return self._start_check()
            if self.phase == Phase.ELECTRICAL:
                return self._finish()
        return []

    def on_observation(self, obs: Observation) -> list[Action]:
        acts: list[Action] = []
        if obs.gesture:
            acts += self._on_gesture(obs.gesture)
        if self.phase == Phase.FIND_MODULE:
            if self.steering and obs.fingertip_mm is not None and self.verifier_available \
                    and obs.t - self._last_identify_t > 4.0:
                acts += self._request_identify()
        elif self.phase == Phase.GUIDE:
            acts += self._guide(obs)
        return acts

    def on_identify_result(self, res: ModuleID) -> list[Action]:
        self._identify_pending = False
        s = self.step
        if self.phase != Phase.FIND_MODULE or s is None:
            if res.module == "unknown":
                return [self._say("I can't tell what you're holding. Turn the printed name toward the camera.")]
            return [self._say(f"You're holding the {res.module}.")]
        if res.module == s.module:
            self._log("found", f"Picked up {s.spoken_module}")
            return self._enter_guide()
        if res.module == "unknown":
            self._identify_fails += 1
            if not self.steering:
                if self._identify_fails >= 2:
                    return [self._say("I still can't read the label, so I'll trust you on this one. "
                                      "Say 'ready' to carry on.")]
                return [self._say("I couldn't read the label. Turn the printed name toward the phone "
                                  "and say 'ready' again.")]
            if self._throttled("unknown_module", 8.0, self.clock()):
                return []
            return [self._say("I can't tell which module that is yet. Hold it flat, with the printed name toward the camera.")]
        self._log("mistake", f"Picked up the {res.module} instead of {s.spoken_module}")
        again = " Say 'ready' when you have it." if not self.steering else ""
        return [self._say(f"That's the {res.module}. Put it down and find {s.spoken_module}.{again}", "urgent")]

    def on_question(self, question: str) -> list[Action]:
        """Free-form question ("is this the right way round?") -> photo -> Claude answers."""
        if not self.verifier_available:
            return [self._say("I can only answer questions about what I see when the vision model is on. "
                              "Try 'repeat' or 'help'.")]
        s = self.step
        context = (f"They are building a {self.project.title} on an Arduino with a Grove Base Shield. "
                   + (f"Current step: plug the {s.module} into {', '.join(self.free_targets())}. " if s else "")
                   + (f"Already connected: {', '.join(f'{m} in {p}' for p, m in self.occupied.items())}."
                      if self.occupied else ""))
        self._log("question", question)
        return [RequestAnswer(question, context)]

    def on_answer(self, answer: str | None) -> list[Action]:
        if not answer:
            return [self._say("Sorry, I couldn't work that out from the photo. Try asking again.")]
        return [self._say(answer)]

    def on_plug_result(self, res: PlugCheck) -> list[Action]:
        self._check_pending = False
        s = self.step
        if self.phase != Phase.CHECKING or s is None:
            return []
        targets = self.free_targets()
        seen = res.port_seen if (res.port_seen in self.layout.ports) else None
        if seen and seen not in targets:
            self._log("mistake", f"{s.spoken_module} went into {seen} instead of {self._target_label()} (caught and fixed)")
            self._check_retries = 0
            self._enter(Phase.GUIDE)
            seen_spoken = self.layout.ports[seen].spoken
            return [self._say(f"That looks like {seen_spoken}, not {self._target_label()}. "
                              f"Unplug it, and I'll guide you to {self._target_label()}.", "urgent")]
        if res.plugged is True or (res.plugged is None and not self.verifier_available):
            port = seen or targets[0]
            self.occupied[port] = s.module
            self._log("plugged", f"{s.spoken_module} -> {port}")
            port_spoken = self.layout.ports[port].spoken if seen or len(targets) == 1 else self._target_label()
            done = self._say(f"Got it. {s.spoken_module[0].upper() + s.spoken_module[1:]} is in {port_spoken}.")
            return [done] + self._advance()
        if res.plugged is None and self._check_retries < 1:
            self._check_retries += 1
            return [self._say("I couldn't see clearly. Move your hand away and hold still.")] + self._start_check(retry=True)
        if res.plugged is None:
            # Give the benefit of the doubt; the serial check at the end will catch it.
            port = targets[0]
            self.occupied[port] = s.module
            self._log("plugged", f"{s.spoken_module} -> {self._target_label()} (not visually confirmed)")
            return [self._say("I can't confirm that from the camera, so I'll test it electrically at the end.")] + self._advance()
        self._check_retries = 0
        self._enter(Phase.GUIDE)
        return [self._say(f"I don't see the cable in {self._target_label()} yet. Let's try again.", "urgent")]

    def on_serial_status(self, st: SerialStatus) -> list[Action]:
        if self.phase != Phase.ELECTRICAL:
            return []
        now = self.clock()
        fc = self.project.final_check
        if st.works and (st.dist_varies or st.seconds_receiving >= 3.0):
            self._log("electrical", f"Readings OK (last distance {st.last_dist} cm)")
            return [self._say(fc.get("success", "It works!"), "urgent")] + self._finish(success=True)
        waited = now - self._phase_t
        if (st.receiving and st.seconds_receiving >= 4.0) or waited >= 20.0:
            msg = diagnose(st, fc.get("sensor_port", "D7"), fc.get("lcd_port", "an I 2 C port"))
            if msg and not self._throttled("diagnose", 12.0, now):
                self._log("electrical", msg)
                return [self._say(msg)]
        return []

    def view_state(self) -> dict:
        s = self.step
        return {
            "phase": self.phase.value,
            "mode": self.mode,
            "step_index": self.step_idx,
            "n_steps": len(self.project.steps),
            "module": s.module if s else None,
            "show": s.show if s else "",
            "targets": self.free_targets() if self.phase in (Phase.GUIDE, Phase.CHECKING) else [],
            "finger_port": self.finger_port,
            "occupied": dict(self.occupied),
            "caption": self.caption,
            "title": self.project.title,
        }

    # ------------------------------------------------------------ transitions
    def _begin_project(self) -> list[Action]:
        p = self.project
        n = len(p.steps)
        intro = f"Let's build the {p.title}. {p.summary} There {'is' if n == 1 else 'are'} {_NUM.get(n, n)} module{'s' if n != 1 else ''} to plug in."
        if self.mode == "voice":
            intro += " When I say up, I mean away from you. Down means toward you."
        acts: list[Action] = [self._say(intro)]
        self.step_idx = 0
        return acts + self._enter_find()

    def _enter_find(self) -> list[Action]:
        self._enter(Phase.FIND_MODULE)
        s = self.step
        self._identify_fails = 0
        find = s.find if self.steering else s.find.replace(" Hold it up to the camera.", "")
        if not self.verifier_available:
            return [self._say(f"{find} Say 'next' when you have it.")]
        if not self.steering:
            return [self._say(f"{find} Say 'ready' when you have it, and I'll take a look.")]
        return [self._say(find)]

    def _enter_guide(self, confirmed_by_user: bool = False) -> list[Action]:
        s = self.step
        self._enter(Phase.GUIDE)
        self._was_at_target_t = None
        self._away_since = None
        self._last_steer_key = None
        self._check_retries = 0
        lead = "Okay." if confirmed_by_user else f"That's {s.spoken_module}."
        text = f"{lead} {s.say}"
        if self.mode == "voice" or not self.steering:
            text += " " + self.layout.tactile_hint(self.free_targets()[0])
        if s.tip:
            text += " " + s.tip
        if not self.steering:
            text += " Say 'done' when it clicks, and I'll take a photo to check."
        return [self._say(text)]

    def _start_check(self, retry: bool = False) -> list[Action]:
        s = self.step
        if s is None or self._check_pending:
            return []
        self._enter(Phase.CHECKING)
        self._check_pending = True
        acts: list[Action] = [] if retry else [self._say("Let me check.", remember=False)]
        return acts + [RequestPlugCheck(s.module, self.free_targets())]

    def _advance(self, skipped: bool = False) -> list[Action]:
        if skipped and self.step:
            self._log("skipped", f"Skipped {self.step.spoken_module}")
        self.step_idx += 1
        self.finger_port = None
        if self.step is not None:
            return self._enter_find()
        if self.project.final_check.get("type") == "serial":
            self._enter(Phase.ELECTRICAL)
            return [self._say(self.project.final_check.get("say", "Plug in the USB cable."))]
        return self._finish(success=True)

    def _go_back(self) -> list[Action]:
        if self.step_idx == 0 and self.phase in (Phase.FIND_MODULE, Phase.MODE_SELECT):
            return [self._say("This is the first step.")]
        if self.phase in (Phase.GUIDE, Phase.CHECKING):
            return self._enter_find()
        self.step_idx = max(0, self.step_idx - 1)
        s = self.step
        for port in [p for p, m in self.occupied.items() if m == s.module]:
            del self.occupied[port]
        return [self._say(f"Going back to {s.spoken_module}.")] + self._enter_find()

    def _finish(self, success: bool = False) -> list[Action]:
        self._enter(Phase.DONE)
        return [Finished(self.build_recap(success))]

    def _request_identify(self, force: bool = False) -> list[Action]:
        if not self.verifier_available:
            return [self._say("I can't identify parts right now. Ask a sighted helper or check the label.")] if force else []
        if self._identify_pending:
            return []
        self._identify_pending = True
        self._last_identify_t = self.clock()
        cands = self.project.modules or [s.module for s in self.project.steps]
        return [RequestIdentify(cands)]

    def _on_gesture(self, g: str) -> list[Action]:
        if self.phase == Phase.MODE_SELECT and g == "Thumb_Up":
            return self.on_command("visual")
        if self.mode != "visual":
            return []
        if g == "Open_Palm":
            return self.on_command("repeat")
        if g == "Thumb_Up":
            if self.phase == Phase.FIND_MODULE:
                return self.on_command("holding" if self.verifier_available else "next")
            return self.on_command("done")
        return []

    # ------------------------------------------------------------ steering
    def _guide(self, obs: Observation) -> list[Action]:
        now = obs.t
        if not self.steering:
            return []
        if not obs.shield_ok:
            if self._throttled("no_board", 6.0, now):
                return []
            return [self._say("I can't see the board. Make sure all four corner markers are in view.", "steer", "no_board")]

        targets = self.free_targets()
        fp = obs.fingertip_mm
        if fp is None:
            self.finger_port = None
            return self._maybe_auto_check(now, hand_away=True)

        port, dist = self.layout.nearest_port(fp)
        tport, tdist = self.layout.nearest_port(fp, among=targets)
        on_board = dist <= 0.75 * self._pitch
        self.finger_port = port.name if on_board else None

        acts: list[Action] = []
        if self.sonify:
            acts.append(Beep(max(0.0, min(1.0, 1.0 - tdist / (4 * self._pitch)))))

        if tdist <= 0.5 * self._pitch:
            self._was_at_target_t = now
            self._away_since = None
            key = f"at:{tport.name}"
            text = f"Stop. That's {tport.spoken}. Plug it in."
            kind = "urgent"
        else:
            if not on_board:
                acts += self._maybe_auto_check(now, hand_away=True)
                if self.phase != Phase.GUIDE:
                    return acts
            else:
                self._away_since = None
            kind = "steer"
            if on_board:
                dc, dr = tport.col - port.col, tport.row - port.row
                if dc == 0 and dr == 0:  # nearest port is the target but finger is between ports
                    dc = int(round((tport.x - fp[0]) / self._pitch))
                    dr = int(round((tport.y - fp[1]) / self._pitch))
                key, text = self._grid_instruction(dc, dr)
            else:
                dx, dy = tport.x - fp[0], tport.y - fp[1]
                if abs(dx) >= abs(dy):
                    word = "right" if dx > 0 else "left"
                else:
                    word = "down" if dy > 0 else "up"
                key, text = f"off:{word}", f"Move {word}, onto the board."

        # Stay quiet when the hand is still and the instruction hasn't changed.
        moved = self._last_steer_fp is None or abs(fp[0] - self._last_steer_fp[0]) + abs(fp[1] - self._last_steer_fp[1]) > 3.0
        if key != self._last_steer_key or moved:
            self._last_steer_key = key
            self._last_steer_fp = fp
            acts.append(self._say(text, kind, key, remember=False))
        return acts

    def _grid_instruction(self, dc: int, dr: int) -> tuple[str, str]:
        # Larger axis first; ties go horizontal (rows are easier to count by touch along an edge).
        if dc != 0 and abs(dc) >= abs(dr):
            n = abs(dc)
            word = "right" if dc > 0 else "left"
            return f"h:{word}:{n}", f"{word.capitalize()} {_NUM.get(n, n)} port{'s' if n > 1 else ''}."
        n = abs(dr)
        word = "down" if dr > 0 else "up"
        return f"v:{word}:{n}", f"{word.capitalize()} {_NUM.get(n, n)}."

    def _maybe_auto_check(self, now: float, hand_away: bool) -> list[Action]:
        """Hand was at the target, then left for ~2 s -> assume it's plugged and check."""
        if self._was_at_target_t is None or not hand_away:
            return []
        if self._away_since is None:
            self._away_since = now
        if now - self._away_since >= self.hand_gone_check_s and now - self._was_at_target_t <= self.hand_gone_check_s + 3.0:
            self._was_at_target_t = None
            self._away_since = None
            return self._start_check()
        return []

    # ------------------------------------------------------------ recap
    def build_recap(self, success: bool) -> str:
        p = self.project
        mins = max(1, round((self.clock() - self.started_t) / 60))
        lines = [f"Pinpoint recap: {p.title}", ""]
        if success:
            lines.append(f"You built it in about {mins} minute{'s' if mins != 1 else ''}.")
        else:
            lines.append("The build ended before the final check.")
        lines.append("")
        lines.append("Connections:")
        for s in p.steps:
            ports = [port for port, m in self.occupied.items() if m == s.module]
            lines.append(f"- {s.module}: {', '.join(ports) if ports else 'not confirmed'}")
        mistakes = [t for _, k, t in self.events if k == "mistake"]
        if mistakes:
            lines.append("")
            lines.append("Caught along the way:")
            lines += [f"- {m[0].upper() + m[1:]}" for m in mistakes]
        elec = [t for _, k, t in self.events if k == "electrical"]
        if elec:
            lines.append("")
            lines.append(f"Final test: {elec[-1]}")
        return "\n".join(lines)
