"""
Build engine for the Elegoo kit: "what do you want to build?" -> plan -> wire it by touch -> upload -> test.

Deterministic state machine, like guidance/engine.py: plain-data inputs, a list of actions out, so the
same flow runs on a FaceTime call, a webcam or the terminal simulator.

  on_utterance(text)            anything the builder says or texts
  on_gesture(name)              "Thumb_Up" (next / visual mode), "Open_Palm" (repeat)
  on_plan_result(PlanResult)    from the planner (worker thread)
  on_setup_result / on_identify_result / on_placement_result / on_review_result / on_answer
  on_upload_result(UploadResult)
  on_serial(receiving, values)  readings of the plan's test key while testing
  on_tick(now)                  progress chatter while waiting on slow work

Actions: Say, RequestPlan, RequestSetupCheck, RequestPartID, RequestPlacementCheck, RequestReview,
RequestUpload, RequestAnswer, Finished.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from backend.build import library
from backend.build.board import BuildPlan, Kit, Orientation, endpoint_text, load_kit, parse_hole
from backend.build.planner import PlanResult
from backend.guidance.engine import RequestAnswer, Say, is_question
from backend.hardware.uploader import UploadResult
from backend.vision.verify import BoardReview, BoardSetup, ModuleID, PlacementCheck

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve"}


class Phase(str, Enum):
    RESUME = "resume"          # called back within the window: carry on, or start fresh?
    INTAKE = "intake"
    PLANNING = "planning"
    CONFIRM = "confirm"
    SETUP = "setup"
    FIND = "find"
    PLACE = "place"
    CHECKING = "checking"
    REVIEW = "review"
    UPLOAD = "upload"
    TEST = "test"
    DONE = "done"


@dataclass
class RequestPlan:
    request: str
    previous: BuildPlan | None = None
    token: int = 0          # the result must carry this back; older (replaced/cancelled) designs are ignored


@dataclass
class RequestSetupCheck:
    pass


@dataclass
class RequestPartID:
    candidates: dict[str, str]
    expected: str


@dataclass
class RequestPlacementCheck:
    step_index: int
    expected: str
    already: str
    targets: list[str] = field(default_factory=list)


@dataclass
class RequestReview:
    expected: str


@dataclass
class RequestUpload:
    plan: BuildPlan


@dataclass
class RequestReply:
    """Something the builder said that isn't a command: Claude decides what to do (route) and what to say."""
    text: str
    context: str
    history: list = field(default_factory=list)
    seq: int = 0            # the utterance it answers; a newer utterance makes it stale


@dataclass
class Finished:
    recap: str


Action = Union[Say, RequestPlan, RequestSetupCheck, RequestPartID, RequestPlacementCheck, RequestReview,
               RequestUpload, RequestAnswer, RequestReply, Finished]

YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "let's go", "lets go", "go", "sounds good", "do it",
       "build it", "let's build it", "great", "perfect", "yes please", "go ahead", "start"}
NO = {"no", "nope", "something else", "never mind", "nevermind", "cancel", "different", "start over"}
_READY = ("ready", "next", "done", "got it", "okay", "ok", "continue", "it's in", "finished", "in", "yes", "plugged in",
          "all set", "set up", "did it", "did that", "done it", "it's done", "that's done", "already", "i'm ready",
          "im ready", "it's there", "yep", "yeah")
_READY_ANYWHERE = ("already", "all set", "i did that", "it's already", "its already", "i've done", "ive done")


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").replace("!", " ").split())


def _is_ready(t: str) -> bool:
    if any(p in t for p in _READY_ANYWHERE):        # "I did what you asked already": go ahead
        return True
    return t in _READY or (len(t.split()) <= 4 and any(t.startswith(r + " ") or t.endswith(" " + r) for r in _READY))


_NEGATIVE_START = ("no", "nope", "nah", "not", "don't", "dont", "do not", "i will not", "i won't", "i wont",
                   "i don't", "i dont", "stop", "wait", "cancel")


def _negative(t: str) -> bool:
    return t in NO or any(t == n or t.startswith(n + " ") for n in _NEGATIVE_START)


_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "great", "perfect", "absolutely",
              "definitely", "correct", "right", "cool", "awesome"}
_YES_PHRASES = ("let's build", "lets build", "build it", "build that", "go ahead", "sounds good", "do it",
                "let's do", "lets do", "let's go", "lets go", "i like it", "start building", "that's good",
                "thats good", "that works", "looks good", "go for it")
_CHANGE_CUES = ("change", "instead", " but ", "add ", "without", "make it", "use a", "use the", "rather",
                "different", "louder", "quieter", "bigger", "smaller", "faster", "slower", "also", "replace",
                "swap", "remove", "can you", "could you", "how about", "what about", "more ", "less ")


def _wants_change(t: str) -> bool:
    padded = f" {t} "
    return any(c in padded for c in _CHANGE_CUES)


def _affirmative(t: str) -> bool:
    if _negative(t):
        return False
    words = t.split()
    return t in YES or (bool(words) and words[0] in _YES_WORDS) or any(p in t for p in _YES_PHRASES)


_VAGUE_EXTRA = {"new", "one", "project", "thing", "things", "stuff", "anything", "something", "else", "different",
                "another", "cool", "fun", "idea", "ideas", "random", "start", "fresh", "build", "again", "now"}


def _vague(t: str) -> bool:
    """"build a new one", "something cool": no idea what to make yet."""
    from backend.build.library import _FILLER
    words = [w for w in t.split() if w not in _FILLER and w not in _VAGUE_EXTRA]
    return not words


def _needs_eyes(t: str) -> bool:
    """Questions only a photo can answer: "is this right?", "can you see it?", "what am I holding?".
    Short, direct requests only: in a long sentence "see" or "right" is usually just conversation
    (the router decides those, and it has its own look action)."""
    padded = f" {t} "
    if any(w in padded for w in (" take a look", " look at my", " look at the", " look at this", " check my",
                                 " check the", " check this", " is this right", " is it right", " can you see")):
        return True
    if len(t.split()) > 14:
        return False
    return any(w in padded for w in (" see ", " look", " is this", " is it ", " right ", " correct", " check",
                                     " tell me what you think", " what do you think", " how does it look",
                                     " did i ", " am i ", " does this", " wrong", " backwards", " holding",
                                     " this one", " these ", " what is this", " what's this"))


def brief(text: str, max_sentences: int = 2, max_words: int = 40) -> str:
    """Spoken replies stay short: long answers pile up and the builder can't get a word in."""
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(sentences[:max_sentences])
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(",;:") + "."
    return out


def _cut_off(text: str) -> bool:
    """Speech-to-text marks a sentence that trailed off ("I want to build a...")."""
    s = text.strip()
    return s.endswith(("...", "…", "-")) or len(_norm(s).split()) == 0


def _command(t: str) -> str | None:
    if t in ("stop", "wait", "hold on", "hang on", "pause", "quiet", "be quiet", "shh", "shush", "one sec",
             "one second", "just a sec", "just a second", "stop talking", "ok stop", "okay stop", "wait wait",
             "shut up", "shut up shut up", "be quiet please", "enough", "stop stop"):
        return "stop"
    if t in ("repeat", "say again", "say that again", "repeat that", "what", "pardon", "sorry") or (
            len(t.split()) <= 8 and any(p in t for p in ("repeat", "say that again", "say it again",
                                                          "come again", "didn't catch", "didnt catch"))):
        return "repeat"
    if t in ("voice mode", "voice"):
        return "voice"
    if t in ("visual mode", "visual", "captions", "caption mode", "deaf mode"):
        return "visual"
    if t in ("back", "go back", "previous", "undo"):
        return "back"
    if t in ("skip", "skip it", "skip this", "skip check", "skip the check"):
        return "skip"
    if t in ("help", "hint", "i'm stuck", "im stuck", "i am stuck"):
        return "help"
    if t in ("start over", "new project", "build something else", "restart"):
        return "restart"
    if t in ("what can i build", "ideas", "give me ideas", "what can you build", "suggestions", "examples"):
        return "ideas"
    return None


class _Rotate:
    """Cycle through ways of saying the same thing, so Pinpoint doesn't sound like a recording."""

    def __init__(self, *options: str):
        self.options, self.i = options, 0

    def __call__(self) -> str:
        out = self.options[self.i % len(self.options)]
        self.i += 1
        return out


class BuildEngine:
    def __init__(self, kit: Kit | None = None, *, mode: str = "", verifier_available: bool = True,
                 uploader_available: bool = True, clock=time.monotonic, check_wires: bool = False):
        self.kit = kit or load_kit()
        self.mode = mode if mode in ("voice", "visual") else "voice"
        self.mode_chosen = mode in ("voice", "visual")
        self.verifier_available = verifier_available
        self.uploader_available = uploader_available
        self.continuous_monitoring = False
        self.check_wires = check_wires
        self.clock = clock
        self.orientation = Orientation()
        self._resume: dict | None = None
        self.programmed = False                       # True only after a successful upload
        self.turns: list[tuple[str, str]] = []      # ("Builder"|"Pinpoint", text), most recent last
        self.utter_seq = 0                            # bumps on every real utterance; answers to older ones are dropped
        self._praise = _Rotate("Perfect.", "Nice, that's right.", "Spot on.", "Looks great.", "Yep, that's it.")
        self._thinking = _Rotate("Good question, one second.", "Let me think.", "Okay, one moment.")
        self._looking = _Rotate("Let me have a look.", "Okay, let me check.", "Let me see.")
        self._still = _Rotate("Still working on it. I'm double-checking every connection so it's safe.",
                              "Almost there, just making sure the wiring is safe.")

        self.phase = Phase.INTAKE
        self.plan: BuildPlan | None = None
        self.plan_note = ""
        self.request = ""
        self.step_idx = 0
        self.caption = ""
        self.show = ""
        self.targets: list[str] = []
        self.events: list[tuple[float, str, str]] = []
        self.started_t = clock()
        self._phase_t = clock()
        self._last_instruction = ""
        self._last_tick_say = clock()
        self._pending = False
        self._review_rounds = 0
        self._unclear = 0                     # unclear photos in a row for the current check
        self._touch_check: str | None = None  # "step" | "review": waiting for yes/no to a check-by-touch
        self._plan_token = 0
        self._setup_tries = 0
        self._upload_tries = 0
        self._test_seen: list[int] = []
        self._test_warned = False

    # ------------------------------------------------------------ helpers
    @property
    def step(self):
        if self.plan is None or self.step_idx >= len(self.plan.steps):
            return None
        return self.plan.steps[self.step_idx]

    def _say(self, text: str, kind: str = "info", remember: bool = True, key: str | None = None) -> Say:
        self.turns.append(("Pinpoint", text))
        del self.turns[:-16]
        self.caption = text
        if remember:
            self._last_instruction = text
        return Say(text, kind, key)

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_t = self.clock()
        self._last_tick_say = self.clock()

    def _log(self, kind: str, text: str) -> None:
        self.events.append((self.clock() - self.started_t, kind, text))

    def _part_name(self, ref: str) -> str:
        pid = self.plan.part_of(ref) if self.plan else None
        return self.kit.parts[pid].name if pid in self.kit.parts else ref

    def _ep(self, name: str) -> str:
        return endpoint_text(name, self.kit, self.orientation)

    def _ep_short(self, name: str) -> str:
        return name.replace("UNO:", "pin ").replace("RAIL:N+", "+ rail").replace("RAIL:N-", "- rail")

    # ------------------------------------------------------------ public API
    # ------------------------------------------------------------ resume after a dropped call
    _RESUMABLE = {"confirm", "setup", "find", "place", "checking", "review", "upload", "test"}

    def snapshot(self) -> dict | None:
        """Everything needed to carry on in the next call, or None if there's nothing to resume."""
        if self.plan is None or self.phase.value not in self._RESUMABLE:
            return None
        o = self.orientation
        return {"plan": self.plan.to_dict(), "plan_source": self.plan.source, "phase": self.phase.value,
                "step_idx": self.step_idx, "request": self.request, "mode": self.mode,
                "mode_chosen": self.mode_chosen, "events": self.events,
                "orientation": {"near_half": o.near_half, "row1": o.row1, "near_outer_rail": o.near_outer_rail}}

    def offer_resume(self, snap: dict) -> None:
        """Next start() asks whether to continue this build or start a fresh one."""
        self._resume = snap

    def _resume_offer_text(self) -> str:
        snap = self._resume
        title = snap["plan"].get("title", "your build")
        n = len(snap["plan"].get("steps", []))
        phase, i = snap["phase"], snap["step_idx"]
        if phase in ("find", "place", "checking"):
            where = f", on step {i + 1} of {n}"
        elif phase in ("review", "upload", "test"):
            where = ", with all the wiring done"
        else:
            where = ""
        return (f"Welcome back! You called back within 30 minutes, so I still have your build: the {title}{where}. "
                "Do you want to carry on with that build, or start a fresh one?")

    def _restore(self) -> list[Action]:
        snap, self._resume = self._resume, None
        self.plan = BuildPlan.from_dict(snap["plan"], source=snap.get("plan_source", "planner"))
        self.step_idx = int(snap.get("step_idx", 0))
        self.request = snap.get("request", "")
        self.events = [tuple(e) for e in snap.get("events", [])]
        if not self.mode_chosen and snap.get("mode_chosen"):
            self.mode, self.mode_chosen = snap.get("mode", self.mode), True
        o = snap.get("orientation") or {}
        self.orientation = Orientation(o.get("near_half", "AE"), o.get("row1", "left"), o.get("near_outer_rail", "-"))
        self._log("resume", f"Carried on after a dropped call ({snap['phase']}, step {self.step_idx + 1})")
        phase = snap["phase"]
        lead = [self._say("Great, let's carry on.", remember=False)]
        if phase == "confirm":
            self._enter(Phase.CONFIRM)
            return lead + [self._say(f"{self.plan.summary} Shall we build it? Say yes, or tell me what to change.")]
        if phase == "setup":
            return lead + self._begin_build()
        if phase in ("find", "place", "checking"):
            return lead + self._enter_step()
        # Wiring "done" in the last call isn't proof: look at the board again before programming or testing.
        return lead + self._enter_review()

    def start(self) -> list[Action]:
        if self._resume is not None:
            self._enter(Phase.RESUME)
            return [self._say(self._resume_offer_text())]
        self._enter(Phase.INTAKE)
        text = "Hi, I'm Pinpoint. What are we building? Say 'a fan' for a quick demo."
        if not self.mode_chosen:
            text += " Thumbs-up turns on captions."
        return [self._say(text)]

    def on_gesture(self, g: str) -> list[Action]:
        if g == "Thumb_Up":
            if not self.mode_chosen and self.phase == Phase.INTAKE:
                return self.on_utterance("visual mode")
            if self.mode == "visual":
                return self.on_utterance("ready")
        if g == "Open_Palm" and self.mode == "visual":
            return self.on_utterance("repeat")
        return []

    def on_utterance(self, text: str) -> list[Action]:
        t = _norm(text)
        if not t or _cut_off(text):
            return []                      # trailed off: wait for the rest
        self.utter_seq += 1
        self.turns.append(("Builder", text.strip()))
        del self.turns[:-16]
        cmd = _command(t)
        if cmd == "stop":
            return []                      # the session already cut the speech; just listen
        if cmd == "repeat":
            return [Say(self._last_instruction or self.caption, "urgent")]
        if self.phase == Phase.SETUP and self._setup_tries and t in {
            "continue", "continue anyway", "trust me", "already correct", "it is correct",
        }:
            self._log("setup", "Builder confirmed the requested layout")
            return [self._say("Okay, I'll trust your setup and use the standard layout.", remember=False)] \
                + self._enter_step()
        if cmd in ("voice", "visual"):
            self.mode, self.mode_chosen = cmd, True
            self._log("mode", f"{cmd} mode")
            if cmd == "visual":
                return [self._say("Captions on. What would you like to build today?" if self.phase == Phase.INTAKE
                                  else "Captions on.")]
            return [self._say("Voice mode on.")]
        if cmd == "restart":
            self.plan, self.step_idx, self._resume = None, 0, None
            self._enter(Phase.INTAKE)
            return [self._say("Sure. What would you like to build instead?")]
        if cmd == "ideas" and self.phase in (Phase.INTAKE, Phase.CONFIRM):
            return [self._say("You can ask for anything the kit can do. " + library.menu_text())]

        if self.phase == Phase.RESUME:
            padded = f" {t} "
            fresh = any(w in padded for w in (" fresh", " new ", " start over", " different", " something else",
                                               " another", " restart", " start again", " from scratch"))
            carry = any(w in padded for w in (" carry on", " continue", " old ", " same", " resume", " keep going",
                                               " that build", " the old", " where we left", " pick up"))
            if fresh or (_negative(t) and not carry):
                self._resume = None
                self._log("resume", "Started fresh instead of resuming")
                self._enter(Phase.INTAKE)
                return [self._say("Okay, a fresh start. What would you like to build today?")]
            if carry or _affirmative(t):
                return self._restore()
            return [self._say("Sorry, I didn't catch that. Say carry on to continue your build, or fresh start "
                              "for a new one.")]
        if self._touch_check and cmd is None:
            out = self._on_touch_answer(t)
            if out is not None:
                return out
        if self.phase == Phase.TEST and self.plan is not None and self.plan.test.type == "ask" and cmd is None:
            if _affirmative(t) or any(w in f" {t} " for w in (" spinning", " it works", " working", " spins",
                                                               " it's going", " it moves")):
                if not _negative(t):
                    self._log("test", "Builder confirmed it works")
                    return [self._say("It works! You built that yourself. Brilliant job!", "urgent")] \
                        + self._finish(True)
            if _negative(t):
                self._log("test", "Builder says it doesn't work yet")
                return [self._say("Let's sort it out. Is the green light on the power module lit? If not, press its "
                                  "button and check the cable is pushed in. If it is lit, check the red wire is in "
                                  "the plus rail and the black wire in the minus rail. Tell me when you've checked.")]
        if self.phase == Phase.INTAKE:
            if _negative(t):
                return []                  # "no, don't do that": not a build request; keep listening
            if not cmd and library.confident_match(text) is not None:
                return self._request_plan(text)   # "a fan", "traffic light": short, but clearly a project
            if ((_is_ready(t) or cmd) and len(t.split()) <= 2) or len(t.split()) < 3 or _vague(t):
                return [self._say("What would you like to build? Say 'ideas' for examples.")]
            return self._request_plan(text)
        if self.phase == Phase.PLANNING:
            if _negative(t):
                # "no, wait": drop the design in progress (its result is ignored when it arrives)
                self._plan_token += 1
                self._pending = False
                self._enter(Phase.INTAKE)
                return [self._say("Okay, I've stopped. What would you like to build?")]
            if len(t.split()) >= 4 and not _is_ready(t):
                return self._request_plan(text)   # a new or corrected idea replaces the one in progress
            return [self._say("I'm still designing it. One moment.", remember=False)]
        if self.phase == Phase.CONFIRM:
            change = _wants_change(t)
            if _affirmative(t) and not change:
                return self._begin_build()
            if _negative(t) and not change:
                self._enter(Phase.INTAKE)
                return [self._say("No problem. What would you like to build instead?")]
            first = t.split()[0]
            if is_question(text) and first not in ("can", "could", "would", "will") and self.plan is not None:
                # "what does it do?" / "how long will it take?": answer from the plan, no photo needed
                return [self._say(f"{self.plan.summary} You'll need {self._parts_sentence(self.plan)}. "
                                  f"{_NUM.get(len(self.plan.steps), len(self.plan.steps))} steps. Say yes, or tell me what to change.")]
            if change or len(t.split()) >= 6:
                return self._request_plan(text, previous=self.plan)
            # short and unclear (often misheard): ask, don't guess, and don't spend a planning call
            return [self._say(f"Sorry, I didn't quite catch that. Do you want to build the {self.plan.title}? "
                              "Say yes or no, or tell me what you'd change.")]

        if cmd == "help":
            return self._help()
        if cmd == "back":
            return self._back()
        if cmd == "skip":
            return self._skip()
        if is_question(text) and not _is_ready(t):
            self._log("question", text)
            local = self._answer_from_plan(t)
            if local is not None:              # "where does the other end go?": the plan knows; no photo
                return [self._say(local, remember=False)]
            if self.verifier_available and _needs_eyes(t):
                return [RequestAnswer(text, self._context())]
            return self._reply(text)
        if not _is_ready(t):
            # Never ignore the builder silently: say what was heard and what Pinpoint is waiting for.
            self._log("comment", text)
            if len(t.split()) < 3:
                return []                      # a stray word or two (often background talk): just keep listening
            if self.verifier_available and _needs_eyes(t):   # "take a look at the circuit": photo, not text
                return [RequestAnswer(text, self._context())]
            local = self._answer_from_plan(t)
            if local is not None:
                return [self._say(local, remember=False)]
            return self._reply(text)
        return self._on_ready()

    def on_tick(self, now: float | None = None) -> list[Action]:
        now = self.clock() if now is None else now
        if self.phase == Phase.PLANNING and now - self._last_tick_say >= 8.0:
            self._last_tick_say = now
            return [self._say(self._still(), remember=False)]
        if self.phase == Phase.CHECKING and self._pending and now - self._last_tick_say >= 8.0:
            self._last_tick_say = now
            return [self._say("I'm still checking the photo.", remember=False)]
        if self.phase == Phase.UPLOAD and self._pending and now - self._last_tick_say >= 8.0:
            self._last_tick_say = now
            return [self._say("Still programming the Arduino.", remember=False)]
        if self.phase == Phase.TEST and not self._test_warned and now - self._phase_t >= 25.0 \
                and self.plan is not None and self.plan.test.type == "serial":
            self._test_warned = True
            return [self._say("I'm not getting readings yet. Check the USB cable is pushed all the way into the "
                              "Arduino. Say skip if you want to finish without the test.")]
        return []

    # ------------------------------------------------------------ results
    def on_plan_result(self, res: PlanResult) -> list[Action]:
        token = getattr(res, "token", None)
        if self.phase != Phase.PLANNING or (token is not None and token != self._plan_token):
            return []                      # a design the builder already replaced or cancelled
        self._pending = False
        if res.plan is None:
            self._enter(Phase.INTAKE)
            return [self._say(f"{res.message} What would you like to build?")]
        self.plan, self.plan_note = res.plan, res.message
        self._log("plan", f"{res.plan.title} ({res.source})")
        parts = self._parts_sentence(res.plan)
        n = len(res.plan.steps)
        self._enter(Phase.CONFIRM)
        text = (f"Let's build {res.plan.title}: {res.plan.summary} You'll need {parts}. "
                f"{_NUM.get(n, n)} steps. Say yes, or tell me what to change.")
        return [self._say(text)]

    def on_setup_result(self, res: BoardSetup) -> list[Action]:
        self._pending = False
        if self.phase != Phase.SETUP:
            return []
        self._setup_tries += 1
        last_try = self._setup_tries >= 3
        if res.board_visible is False and not last_try:
            return [self._say("I can't see the breadboard. Switch to the back camera, hold it over the board, and say ready.",
                              "urgent")]
        if res.near_half:
            self.orientation.near_half = res.near_half
        if res.row1_end:
            self.orientation.row1 = res.row1_end
        if res.near_outer_rail:
            self.orientation.near_outer_rail = res.near_outer_rail
        layout_ok = res.usb_left
        if layout_ok is None and res.board_visible is True:
            self._log("setup", "Board visible; orientation unclear, using the requested standard layout")
            return [self._say("I can see the board, but can't reliably judge its direction. I'll use the layout "
                              "I asked you to set: centered boards with their edges touching, row 1 left, USB "
                              "pointing left.", remember=False)] \
                + self._enter_step()
        if layout_ok is True:
            self._log("setup", res.description or "board already set up right")
            row = f" Row 1 is on your {self.orientation.row1}." if res.row1_end else ""
            return [self._say(f"Setup looks good.{row} Put the phone down.", "urgent", remember=False)] + self._enter_step()
        if layout_ok is False and not last_try:
            fix = res.fix or ("Put the breadboard's long edge toward you. Place the Arduino beyond it, USB pointing left.")
            return [self._say(f"{fix} Say ready after changing it. If it's already set up that way, say "
                              "'continue' and I'll trust you.", "urgent")]
        if self._setup_tries < 2:
            return [self._say("Couldn't see it. Hold the phone lower and steady, then say ready.", "urgent")]
        self._log("setup", "Setup photo unclear; using the standard layout")
        return [self._say("Using standard layout: Arduino beyond the board, USB left, row 1 at your left.",
                          remember=False)] + self._enter_step()

    def on_identify_result(self, res: ModuleID) -> list[Action]:
        self._pending = False
        s = self.step
        if self.phase != Phase.FIND or s is None:
            return [self._say(f"That looks like the {self.kit.parts[res.module].name}.")] \
                if res.module in self.kit.parts else []
        want = self.plan.part_of(s.ref)
        if res.module == want or res.module == "unknown":
            if res.module == "unknown":
                self._log("note", f"Couldn't confirm {self._part_name(s.ref)} by camera")
            return self._enter_place()
        self._log("mistake", f"Picked up the {self.kit.parts[res.module].name} instead of the {self._part_name(s.ref)}")
        return [self._say(f"That's a {self.kit.parts[res.module].name}. Find the {self._part_name(s.ref)} instead; say ready.",
                          "urgent")]

    def on_placement_result(self, res: PlacementCheck) -> list[Action]:
        self._pending = False
        if self.phase != Phase.CHECKING:
            return []
        s = self.step
        if res.ok is False:
            self._log("mistake", f"{self._part_name(s.ref)}: {res.problem or res.seen} (caught and fixed)")
            self._enter(Phase.PLACE)
            fix = res.problem or "Something isn't in the right hole yet."
            return [self._say(f"Almost! {fix} Tell me when you've fixed it.", "urgent")]
        if res.ok is None:
            if self.continuous_monitoring:
                self._enter(Phase.PLACE)
                return [self._say("I can't see that placement clearly yet. Keep the video pointed at the circuit; "
                                  "I'll check again.", remember=False)]
            self._unclear += 1
            if self._unclear < 2 and self.verifier_available:
                # Try again before giving up, with exact aiming help.
                self._pending = True
                spot = self._aim_hint(s)
                return [self._say(f"I couldn't see it clearly. {spot} Hold still.", remember=False),
                        RequestPlacementCheck(self.step_idx, self._expected(s), self._already(), self._targets(s))]
            # Still unclear: find out from the builder, by touch, instead of skipping the check.
            self._log("note", f"{self._part_name(s.ref)} checked by touch (photo unclear)")
            self._touch_check = "step"
            return [self._say(f"I still can't see it clearly, so let's check by touch. {self._where(s)} "
                              "Is it like that? Say yes or no.")]
        self._log("placed", f"{self._part_name(s.ref)} placed correctly")
        return [self._say(self._praise(), remember=False)] + self._advance()

    def on_review_result(self, res: BoardReview) -> list[Action]:
        self._pending = False
        if self.phase != Phase.REVIEW:
            return []
        if res.ok is None and self.continuous_monitoring:
            self._enter(Phase.REVIEW)
            return [self._say("I can't see the whole circuit clearly yet. Keep the video pointed at the board; "
                              "I'll check again.", remember=False)]
        self._review_rounds += 1
        if res.ok is False and res.problems and self._review_rounds <= 2:
            for p in res.problems:
                self._log("mistake", f"Final check: {p} (caught)")
            first = brief(res.problems[0])
            more = f" There {'is' if len(res.problems) == 2 else 'are'} {len(res.problems) - 1} more thing{'s' if len(res.problems) > 2 else ''} after that." \
                if len(res.problems) > 1 else ""
            return [self._say(f"One thing to fix before we power it up. {first}{more} Tell me when it's fixed "
                              "and I'll look again.", "urgent")]
        if res.ok is None and self._review_rounds <= 2:
            self._pending = True
            return [self._say("I couldn't see the board in that photo. Hold the phone above the breadboard and "
                              "the Arduino so both are in view, and keep it still.", "urgent", remember=False),
                    RequestReview(self._expected_all())]
        if res.ok is None and self.plan is not None:
            # Still can't see it: go through the connections with the builder instead of skipping the check.
            self._log("review", "Final check by touch (photos unclear)")
            self._touch_check = "review"
            checklist = "; ".join(self._short(x) for x in self.plan.steps)
            return [self._say("I can't see the whole board clearly, so let's check it by touch. " + checklist +
                              ". Is everything like that? Say yes or no.")]
        self._log("review", "Final check passed" if res.ok else "Final check finished")
        return [self._say("The wiring looks great.", remember=False)] + self._enter_upload()

    def _short(self, s) -> str:
        """One connection, spoken briefly: "the red LED with anode in row 10, cathode in row 11"."""
        def spot(ep):
            h = parse_hole(ep)
            if h:
                return f"row {h[1]}"
            return {"RAIL:N+": "the plus rail", "RAIL:N-": "the minus rail"}.get(ep, ep.replace("UNO:", "pin "))
        if s.kind == "wire":
            return f"a wire from {spot(s.connections[0].to)} to {spot(s.connections[1].to)}"
        pt = self.kit.parts[self.plan.part_of(s.ref)]
        legs = ", ".join(f"{c.lead} in {spot(c.to)}" for c in s.connections)
        return f"the {pt.name} with {legs}"

    def on_upload_result(self, res: UploadResult) -> list[Action]:
        self._pending = False
        if self.phase != Phase.UPLOAD:
            return []
        self._upload_tries += 1
        if res.ok:
            self.programmed = True
            self._log("upload", "Code uploaded")
            return self._enter_test()
        self._log("upload", f"{res.stage} failed")
        if res.stage == "board":
            # Wrong board: nothing will work until it's swapped, so never pretend otherwise.
            return [self._say(res.message + " Or say skip to finish without programming it.")]
        if res.stage == "setup" or self._upload_tries >= 3:
            self._enter(Phase.TEST)
            self._test_seen, self._test_warned = [], True
            self.show = "Not programmed"
            return [self._say(res.message + " So it isn't programmed yet, and it won't work until the code is "
                              "uploaded from the laptop. Say skip to finish, or say back to check the wiring.")]
        return [self._say(res.message)]

    def on_answer(self, answer: str | None, seq: int = 0) -> list[Action]:
        if seq and seq != self.utter_seq:
            return []                          # they've said something since: this answer is stale
        if not answer:
            return [self._say("Sorry, I couldn't work that out from the photo. Try asking again.", remember=False)]
        return [self._say(brief(answer), remember=False)]

    def on_serial(self, receiving: bool, values: list[int]) -> list[Action]:
        if self.phase != Phase.TEST or self.plan is None:
            return []
        t = self.plan.test
        if not receiving or not values:
            return []
        self._test_seen.extend(values[-5:])
        seen = self._test_seen[-200:]
        # Binary readings (0/1) count as soon as both appear; analog ones must move clearly beyond noise.
        works = (t.expect == "present") or (max(seen) - min(seen) > max(0.5, 0.08 * max(abs(v) for v in seen)))
        if not works:
            return []
        self._log("test", f"{t.key} readings {min(seen)}..{max(seen)}")
        return [self._say("It works! I can see it responding. You built that. Brilliant job!", "urgent")] + self._finish(True)

    # ------------------------------------------------------------ flow
    def _request_plan(self, text: str, previous: BuildPlan | None = None) -> list[Action]:
        self.request = text if previous is None else f"{self.request}. Then: {text}"
        self._log("request", text)
        self._enter(Phase.PLANNING)
        self._pending = True
        self._plan_token += 1
        lead = "Designing your circuit." if previous is None else "Updating the design."
        return [self._say(lead, remember=False),
                RequestPlan(text, previous, self._plan_token)]

    def _parts_sentence(self, plan: BuildPlan) -> str:
        counts: dict[str, int] = {}
        for p in plan.parts:
            name = self.kit.parts[p.part].name if p.part in self.kit.parts else p.part
            counts[name] = counts.get(name, 0) + 1
        items = [f"{'a' if n == 1 else _NUM.get(n, n)} {name}{'' if n == 1 else 's'}" for name, n in counts.items()]
        if any(s.kind == "wire" for s in plan.steps):
            items.append("some jumper wires")
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    def _begin_build(self) -> list[Action]:
        self._enter(Phase.SETUP)
        self.step_idx = 0
        self._setup_tries = 0
        if self.plan is not None and not self.plan.uses_arduino:
            # Only the breadboard and the power module: nothing to line up, nothing to photograph.
            self._log("setup", "No Arduino in this build")
            return [self._say("Great! This one doesn't need the Arduino at all, just the breadboard and the power "
                              "module. Put the breadboard flat in front of you, long side toward you.",
                              remember=False)] + self._enter_step()
        if self.verifier_available:
            # Look first: if the board is already set up right, there's nothing to explain.
            self._pending = True
            return [self._say("Great! First, let me take a quick look at how your board is set up.", remember=False),
                    RequestSetupCheck()]
        text = ("Let's set up. Put the breadboard flat in front of you, long side toward you. Put the Arduino beyond "
                "it, centered with their long edges parallel and touching, USB pointing to your left. Leave the USB "
                "cable unplugged for now.")
        if self.verifier_available:
            text += (" Then pick up the phone and hold it flat above the breadboard, top edge pointing away from "
                     "you, and say ready.")
        else:
            text += " Tell me when it's set up."
        return [self._say(text)]

    def _on_ready(self) -> list[Action]:
        if self._pending:
            return []
        if self.phase == Phase.SETUP:
            if self.verifier_available:
                self._pending = True
                return [RequestSetupCheck()]
            return self._enter_step()
        if self.phase == Phase.FIND:
            s = self.step
            pid = self.plan.part_of(s.ref)
            if self.verifier_available and pid and self.kit.parts[pid].resistor:
                self._pending = True
                cands = {k: v.name for k, v in self.kit.parts.items() if v.resistor}
                return [RequestPartID(cands, pid)]
            return self._enter_place()
        if self.phase == Phase.PLACE:
            s = self.step
            if self.verifier_available and (s.kind == "place" or self.check_wires):
                self._enter(Phase.CHECKING)
                self._pending = True
                return [self._say(self._looking(), remember=False),
                        RequestPlacementCheck(self.step_idx, self._expected(s), self._already(), self._targets(s))]
            self._log("placed", f"step {self.step_idx + 1} done")
            return self._advance()
        if self.phase == Phase.REVIEW:
            self._pending = True
            return [RequestReview(self._expected_all())]
        if self.phase == Phase.UPLOAD:
            self._pending = True
            return [self._say("Programming your Arduino now. It takes about twenty seconds.", remember=False),
                    RequestUpload(self.plan)]
        if self.phase == Phase.TEST and self.plan and self.plan.test.type == "ask":
            return [self._say("Is it working? Say yes or no.")]   # "done" = they've switched it on: ask
        if self.phase == Phase.TEST and self.plan and self.plan.test.type != "serial":
            return self._finish(True)
        return []

    def _aim_hint(self, s) -> str:
        """Where to hold the phone for a close-up of this step."""
        rows = sorted({parse_hole(c.to)[1] for c in s.connections if parse_hole(c.to)})
        if rows:
            span = f"row {rows[0]}" if len(rows) == 1 else f"rows {rows[0]} to {rows[-1]}"
            return f"Hold the phone a hand's width above {span}, counting from the {self.orientation.row1} end."
        if any(c.to.startswith("RAIL:") for c in s.connections):
            return "Hold the phone a hand's width above the edge of the breadboard nearest you."
        return "Hold the phone a hand's width above the part you just placed."

    def _on_touch_answer(self, t: str) -> list[Action] | None:
        """The builder's yes/no to "let's check by touch"."""
        kind = self._touch_check
        if _affirmative(t) and not _negative(t):
            self._touch_check = None
            self._unclear = 0
            if kind == "step":
                self._log("placed", f"{self._part_name(self.step.ref)} confirmed by touch")
                return [self._say(self._praise(), remember=False)] + self._advance()
            self._log("review", "Final check confirmed by touch")
            return [self._say("Great, everything's in place.", remember=False)] + self._enter_upload()
        if _negative(t):
            self._touch_check = None
            self._unclear = 0
            if kind == "step":
                self._enter(Phase.PLACE)
                return [self._say(f"No problem, let's fix it. {self._where(self.step)} Tell me when it's in.")]
            return [self._say("No problem. Tell me which part isn't right, or say back to go through the steps "
                              "again.")]
        return None

    def _enter_step(self) -> list[Action]:
        self._unclear, self._touch_check = 0, None
        s = self.step
        if s is None:
            return self._enter_review()
        n = len(self.plan.steps)
        k = self.step_idx + 1
        if k == 1:
            prefix = f"Step 1 of {n}. "
        elif k == n:
            prefix = "Last step! "
        elif n >= 6 and k == n // 2 + 1:
            prefix = f"You're halfway there. Step {k}. "
        else:
            prefix = f"Step {k}. "
        self.targets = self._targets(s)
        if s.kind == "wire":
            self._enter(Phase.PLACE)
            a, b = s.connections[0].to, s.connections[1].to
            self.show = f"Wire: {self._ep_short(a)} -> {self._ep_short(b)}"
            text = (f"{prefix}Jumper wire: connect one end to {self._ep(a)} and the other to {self._ep(b)}. "
                    "Say done.")
            if self.mode == "visual":
                text = f"{prefix}Jumper wire: {self._ep_short(a)} to {self._ep_short(b)}. Thumbs-up when done."
            return [self._say(text)]
        pt = self.kit.parts[self.plan.part_of(s.ref)]
        self._enter(Phase.FIND)
        self.show = f"Find: {pt.name}"
        text = f"{prefix}Find the {pt.name}: {pt.touch}. Say ready."
        if self.mode == "visual":
            text = f"{prefix}Find the {pt.name}. Thumbs-up when you have it."
        return [self._say(text)]

    def _enter_place(self) -> list[Action]:
        s = self.step
        pt = self.kit.parts[self.plan.part_of(s.ref)]
        self._enter(Phase.PLACE)
        self.show = f"{pt.name}: " + ", ".join(f"{c.lead} -> {self._ep_short(c.to)}" for c in s.connections)
        if self.mode == "visual":
            return [self._say(f"{self.show}. Thumbs-up when done.")]
        if pt.mount:
            return [self._say(f"{pt.mount} Say done.")]
        parts = []
        if pt.layout == "wires":
            parts.append("Use one socket-ended jumper per pin.")
        elif pt.layout == "inline":
            parts.append("Put each leg in a neighboring row.")
        elif pt.layout == "straddle":
            parts.append("Place it across the groove.")
        verb = "Connect" if pt.layout == "wires" else "Put"
        for c in s.connections:
            desc = pt.leads.get(c.lead, c.lead)
            if desc.startswith("either"):
                desc = "one " + desc.split(" ", 1)[1]
            desc = desc if desc.startswith(("the ", "one ")) else f"the {desc}"
            if pt.layout in ("wires", "inline") and c.lead.lower() not in desc.lower() and c.lead[:1].isupper():
                desc += f", labelled {c.lead}"
            into = "to" if verb == "Connect" else "into"
            parts.append(f"{verb} {desc} {into} {self._ep(c.to)}.")
        return [self._say(" ".join(parts) + " Say done.")]

    def _advance(self) -> list[Action]:
        self.step_idx += 1
        return self._enter_step()

    def _enter_review(self) -> list[Action]:
        if self.plan is not None and not self.plan.uses_arduino:
            self.targets = []
            return [self._say("That's all the wiring, nice work!", remember=False)] + self._enter_test()
        self._enter(Phase.REVIEW)
        self.targets = []
        self.show = "Final check"
        if not self.verifier_available:
            return self._enter_upload()
        self._review_rounds = 0
        self._pending = True
        return [self._say("Final photo check before power-up.", remember=False),
                RequestReview(self._expected_all())]

    def _enter_upload(self) -> list[Action]:
        self._enter(Phase.UPLOAD)
        self.show = "Plug in USB"
        self._upload_tries = 0
        if not self.uploader_available:
            return [self._say("Plug the Arduino into the laptop by USB.")] + self._enter_test(manual=True)
        return [self._say("Connect the Arduino to the laptop by USB, then say ready to upload.")]

    def _enter_test(self, manual: bool = False) -> list[Action]:
        self._enter(Phase.TEST)
        self._test_seen, self._test_warned = [], False
        t = self.plan.test
        self.show = "Test it!"
        tail = " Say skip when you're done." if manual else ""
        return [self._say(f"Moment of truth! {t.instruction}{tail}")]

    def _finish(self, success: bool) -> list[Action]:
        self._enter(Phase.DONE)
        self.show = "Done!"
        self.targets = []
        return [Finished(self.build_recap(success))]

    def _help(self) -> list[Action]:
        if self.phase in (Phase.PLACE, Phase.FIND) and self.step is not None:
            s = self.step
            tip = (" Count rows from the left end with a fingertip; every hole is one row. "
                   "The groove is the long channel down the middle.")
            return [self._say(f"{s.why} {tip}" if s.why else tip.strip())]
        if self.phase == Phase.SETUP:
            return [self._say("The breadboard is the white board full of holes. The Arduino is the board with the "
                              "metal USB socket.")]
        return [Say(self._last_instruction or self.caption, "urgent")]

    def _reply(self, text: str) -> list[Action]:
        """A real answer to whatever the builder said, from the plan and the current step (no photo)."""
        if not self.verifier_available:
            s = self.step
            return [self._say(self._where(s) if s is not None else "Say help, or say back to go to the last step.",
                              remember=False)]
        # Acknowledge questions straight away; statements may be background talk, so wait for the router.
        ack = [self._say(self._thinking(), remember=False)] if is_question(text) or "?" in text else []
        return ack + [RequestReply(text, self._route_state(), list(self.turns[:-1]), self.utter_seq)]

    def expects_answer(self, text: str) -> bool:
        """Is this a direct answer to the question Pinpoint just asked? (Never filtered out as echo.)"""
        t = _norm(text)
        if self.phase == Phase.RESUME:
            return any(w in f" {t} " for w in (" fresh", " new ", " carry on", " continue", " start "))
        if self.phase in (Phase.CONFIRM, Phase.INTAKE):
            return _affirmative(t) or _negative(t)
        return _is_ready(t)

    def _route_state(self) -> str:
        """Everything the router needs: the plan, which steps are done, where Pinpoint thinks we are."""
        if self.plan is None:
            return "No build chosen yet."
        n = len(self.plan.steps)
        lines = [f"Build: {self.plan.title}. {self.plan.summary}",
                 {Phase.SETUP: "Pinpoint is checking how the board is set up.",
                  Phase.FIND: f"Pinpoint is on step {self.step_idx + 1}: waiting for them to find the part.",
                  Phase.PLACE: f"Pinpoint is on step {self.step_idx + 1}: waiting for them to place it and say done.",
                  Phase.CHECKING: f"Pinpoint is checking a photo of step {self.step_idx + 1}.",
                  Phase.REVIEW: "Pinpoint believes all wiring steps are done and is doing the final photo check.",
                  Phase.UPLOAD: "Pinpoint believes all wiring is done and is waiting for the USB cable to program it.",
                  Phase.TEST: "Pinpoint is at the test stage, waiting for them to try the finished build.",
                  }.get(self.phase, f"Phase: {self.phase.value}."),
                 "Steps (done = Pinpoint thinks it's done; the builder may say otherwise):"]
        for i, s in enumerate(self.plan.steps):
            mark = ("done" if i < self.step_idx or self.phase in (Phase.REVIEW, Phase.UPLOAD, Phase.TEST)
                    else "CURRENT" if i == self.step_idx else "to do")
            lines.append(f"  {i + 1}. [{mark}] {self._describe(s)[2:]}")
        s = self.step
        if s is not None:
            lines.append("Touch directions for the current step: " + self._where(s))
        if not self.plan.uses_arduino:
            lines.append("This build has no Arduino and no code: the power module powers it. At the end the "
                         "builder plugs in the power module's cable, presses its button, and says if it works.")
        else:
            lines.append("Programmed: yes, the code is on the Arduino." if self.programmed else
                         "Programmed: NO. The code has not been uploaded successfully, so nothing will work yet.")
        lines.append(f"The Arduino is beyond the breadboard with its USB socket on the builder's left. "
                     f"There are {n} steps; after them: final photo check, plug in USB, Pinpoint programs it, test.")
        return "\n".join(lines)

    def on_route(self, res: dict | None, seq: int, text: str = "") -> list[Action]:
        """Apply what the router decided. Answers to an utterance the builder has since followed up on are dropped."""
        if seq and seq != self.utter_seq:
            return []
        if res is None:
            return [self._say("Sorry, I didn't catch that properly. Could you say it again?", remember=False)]
        action, say = res.get("action", "answer"), brief((res.get("say") or "").strip())
        lead = [self._say(say, remember=False)] if say else []
        if action == "ignore":
            return []
        if action == "go_to_step" and self.plan is not None:
            n = int(res.get("step") or 0)
            if 1 <= n <= len(self.plan.steps):
                self._pending = False
                self.step_idx = n - 1
                self._log("route", f"Moved to step {n} ({text})")
                return lead + self._enter_step()
        if action == "next":
            if self.phase in (Phase.REVIEW, Phase.UPLOAD, Phase.TEST, Phase.SETUP, Phase.FIND, Phase.PLACE):
                self._pending = False
                return lead + self._on_ready()
        if action == "back":
            return lead + self._back()
        if action == "program" and self.plan is not None:
            if not self.uploader_available:
                return lead + [self._say("I can't program the Arduino from this laptop, so the code has to be "
                                         "uploaded by hand.", remember=False)]
            self._pending = False
            self._enter(Phase.UPLOAD)
            self._upload_tries = 0
            self._log("route", f"Programming on request ({text})")
            return lead + self._on_ready()
        if action == "look" and self.verifier_available:
            return lead + [RequestAnswer(text or "Is this right?", self._context())]
        if action == "restart":
            self.plan, self.step_idx, self._resume = None, 0, None
            self._enter(Phase.INTAKE)
            return lead + [self._say("What would you like to build instead?")]
        return lead

    def _reply_context(self) -> str:
        parts = [self._context()]
        s = self.step
        if s is not None and self.phase in (Phase.FIND, Phase.PLACE, Phase.CHECKING):
            parts.append("Exact directions for this step: " + self._where(s))
        waiting = {Phase.SETUP: "a setup photo", Phase.FIND: "them to find the part and say ready",
                   Phase.PLACE: "them to place it and say done", Phase.CHECKING: "a photo check",
                   Phase.REVIEW: "the final photo check", Phase.UPLOAD: "the USB cable to be plugged in (then 'ready')",
                   Phase.TEST: "them to try the finished build"}.get(self.phase)
        if waiting:
            parts.append(f"Pinpoint is waiting for {waiting}.")
        if self.plan is not None and self.phase in (Phase.REVIEW, Phase.UPLOAD, Phase.TEST):
            parts.append("Full wiring: " + "; ".join(self._describe(x)[2:] for x in self.plan.steps))
        return " ".join(parts)

    def _answer_from_plan(self, t: str) -> str | None:
        """Answer "where / which / how" questions from the plan itself. Questions that need eyes
        ("can you see it?", "is this right?") return None and go to the camera."""
        padded = f" {t} "
        if any(w in padded for w in (" see ", " look", " right ", " correct", " check", " is this", " is it ",
                                     " did i ", " am i ", " does this", " wrong", " backwards", " holding")):
            return None
        if any(w in padded for w in (" who are you", " who is this", " who's this", " whos this", " who is the",
                                     " who's the", " who am i talking", " your name", " who is that")):
            return ("I'm Pinpoint, your build coach. I design the circuit, guide your hands step by step, check "
                    "your work with the camera, and program the Arduino for you.")
        s = self.step
        if " jumper" in padded and " how " in padded:
            return ("A jumper wire is a single thin wire with a metal pin at each end. Push a pin straight down "
                    "into the hole until it feels snug, about half a centimetre. " + self._where(s))
        if " which one" in padded or " which of" in padded or " which is" in padded:
            return None          # "which one of these is the LED?": telling parts apart, not where it goes
        if any(w in padded for w in (" where ", " which ", " what pin", " what hole", " what row", " other end",
                                     " what next", " what's next", " whats next", " what do i do", " how do i ",
                                     " what now", " go ")):
            if s is not None:
                return self._where(s)
            if any(w in padded for w in (" usb", " cable", " laptop")) and " pin" not in padded:
                return self._where(None)
            return None          # e.g. "where does this yellow wire go?" after the steps: Claude, with the full wiring
        if " groove" in padded or " rail" in padded or " row" in padded:
            return ("The groove is the long channel down the middle of the breadboard. Rows run across it: count "
                    "them from the left end with a fingertip, one hole per row. The rails are the long strips "
                    "along the edge nearest you.")
        return None

    def _where(self, s) -> str:
        if s is None:
            if self.phase in (Phase.UPLOAD, Phase.TEST):
                return ("The USB cable's square end goes into the Arduino's USB socket, on the side pointing to "
                        "your left. The other end goes into the laptop. If something isn't wired yet, say back.")
            return "There's nothing left to place. Say back to go to the last wiring step."
        if s.kind == "wire":
            a, b = s.connections[0].to, s.connections[1].to
            return f"This wire: one end to {self._ep(a)}; the other to {self._ep(b)}."
        pt = self.kit.parts[self.plan.part_of(s.ref)]
        legs = " ".join(f"{pt.leads.get(c.lead, c.lead).capitalize()} goes into {self._ep(c.to)}."
                        for c in s.connections)
        return f"For the {pt.name}: {legs}"

    def _back(self) -> list[Action]:
        if self.phase in (Phase.REVIEW, Phase.UPLOAD, Phase.TEST) and self.plan is not None:
            self._pending = False
            self.step_idx = len(self.plan.steps) - 1
            return [self._say("Okay, back to the wiring.", remember=False)] + self._enter_step()
        if self.phase in (Phase.PLACE, Phase.FIND, Phase.CHECKING) and self.step_idx > 0:
            self.step_idx -= 1
            self._pending = False
            return [self._say("Going back one step.", remember=False)] + self._enter_step()
        return [self._say("This is the first step.", remember=False)]

    def _skip(self) -> list[Action]:
        self._pending = False
        if self.phase == Phase.SETUP:
            self._log("setup", "Setup photo skipped")
            return [self._say("Standard layout: row 1 left, holes A to E near.",
                              remember=False)] + self._enter_step()
        if self.phase in (Phase.PLACE, Phase.FIND, Phase.CHECKING):
            self._log("skipped", f"Skipped check of step {self.step_idx + 1}")
            return self._advance()
        if self.phase == Phase.REVIEW:
            return self._enter_upload()
        if self.phase == Phase.UPLOAD:
            return self._enter_test(manual=True)
        if self.phase == Phase.TEST:
            return self._finish(False)
        return []

    # ------------------------------------------------------------ descriptions for the vision checks
    def _targets(self, s) -> list[str]:
        return [c.to for c in s.connections if parse_hole(c.to) or c.to.startswith("RAIL:")]

    def _describe(self, s) -> str:
        if s.kind == "wire":
            return f"- a jumper wire from {s.connections[0].to} to {s.connections[1].to}"
        pt = self.kit.parts[self.plan.part_of(s.ref)]
        legs = "; ".join(f"{c.lead} ({pt.leads.get(c.lead, '')}) in {c.to}" for c in s.connections)
        return f"- {s.ref}, the {pt.name}: {legs}"

    def _expected(self, s) -> str:
        return self._describe(s)

    def _already(self) -> str:
        return "\n".join(self._describe(s) for s in self.plan.steps[:self.step_idx])

    def _expected_all(self) -> str:
        return "\n".join(self._describe(s) for s in self.plan.steps)

    def _context(self) -> str:
        if self.plan is None:
            return "They are deciding what to build with an Elegoo Arduino starter kit."
        s = self.step
        cur = f" Current step: {self._describe(s)[2:]}." if s is not None else ""
        return (f"They are building a {self.plan.title} ({self.plan.summary}) on an Arduino UNO and a breadboard, "
                f"step {min(self.step_idx + 1, len(self.plan.steps))} of {len(self.plan.steps)}.{cur}")

    # ------------------------------------------------------------ dashboard + recap
    def view_state(self) -> dict:
        n = len(self.plan.steps) if self.plan else 0
        return {
            "phase": self.phase.value,
            "mode": self.mode,
            "step_index": self.step_idx,
            "n_steps": n,
            "module": None,
            "show": self.show,
            "targets": list(self.targets),
            "finger_port": None,
            "occupied": {},
            "caption": self.caption,
            "title": self.plan.title if self.plan else "What shall we build?",
            "request": self.request,
            "plan_source": self.plan.source if self.plan else None,
        }

    def build_recap(self, success: bool) -> str:
        p = self.plan
        mins = max(1, round((self.clock() - self.started_t) / 60))
        lines = [f"Pinpoint recap: {p.title if p else 'build'}", ""]
        if self.request:
            lines.append(f'You asked for: "{self.request}"')
        lines.append(f"You built it in about {mins} minute{'s' if mins != 1 else ''}."
                     if success else "The build ended before the final test.")
        if p:
            lines += ["", "Parts:"] + [f"- {p_.ref}: {self.kit.parts[p_.part].name}" for p_ in p.parts
                                       if p_.part in self.kit.parts]
            lines += ["", "Connections:"]
            for s in p.steps:
                lines.append(self._describe(s))
        mistakes = [t for _, k, t in self.events if k == "mistake"]
        if mistakes:
            lines += ["", "Caught along the way:"] + [f"- {m}" for m in mistakes]
        tests = [t for _, k, t in self.events if k == "test"]
        if tests:
            lines += ["", f"Final test: {tests[-1]}"]
        return "\n".join(lines)
