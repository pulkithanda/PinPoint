"""
Turn "what do you want to build?" into a safe, buildable plan.

Claude designs the circuit from the kit's parts and writes the Arduino sketch (structured
output). Code then checks the plan (board.validate): shorts, missing resistors, floating legs,
holes used twice, rows too far to count by touch. Problems go back to Claude to fix, up to
MAX_REPAIRS times. If that still fails, or Claude isn't available, the closest project from the
hand-checked library is offered instead.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.build import library
from backend.build.board import MAX_TOUCH_ROW, BuildPlan, Kit, load_kit, validate

log = logging.getLogger(__name__)

MAX_REPAIRS = 2

_CONNECTION = {
    "type": "object",
    "properties": {"lead": {"type": "string"}, "to": {"type": "string"}},
    "required": ["lead", "to"], "additionalProperties": False,
}
_PLAN = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "snake_case id"},
        "title": {"type": "string", "description": "2-4 words"},
        "summary": {"type": "string", "description": "One spoken sentence: what it does."},
        "difficulty": {"type": "string", "enum": ["basic", "advanced"]},
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "part": {"type": "string"}},
            "required": ["ref", "part"], "additionalProperties": False}},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["place", "wire"]},
                "ref": {"type": "string", "description": "Part ref for place steps; empty for wires."},
                "why": {"type": "string", "description": "One short spoken sentence: what this connection does."},
                "connections": {"type": "array", "items": _CONNECTION},
            },
            "required": ["kind", "ref", "why", "connections"], "additionalProperties": False}},
        "sketch": {"type": "string", "description": "Complete Arduino UNO sketch."},
        "libraries": {"type": "array", "items": {"type": "string"}},
        "test": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["serial"]},
                "key": {"type": "string"},
                "expect": {"type": "string", "enum": ["varies", "present"]},
                "instruction": {"type": "string", "description": "What the builder does to test it, spoken."},
            },
            "required": ["type", "key", "expect", "instruction"], "additionalProperties": False},
    },
    "required": ["id", "title", "summary", "difficulty", "parts", "steps", "sketch", "libraries", "test"],
    "additionalProperties": False,
}
_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["build", "library", "decline"]},
        "library_id": {"type": "string", "description": "Library project id when decision is 'library', else empty."},
        "reason": {"type": "string", "description": "One or two short spoken sentences to the builder."},
        "plan": _PLAN,
    },
    "required": ["decision", "library_id", "reason", "plan"],
    "additionalProperties": False,
}
_SKETCH_SCHEMA = {
    "type": "object",
    "properties": {"sketch": {"type": "string"}, "note": {"type": "string"}},
    "required": ["sketch", "note"], "additionalProperties": False,
}


def _system(kit: Kit) -> str:
    lib = "\n".join(f"- {e.plan.id}: {e.plan.title}. {e.plan.summary}" for e in library.load_library())
    return f"""You design Arduino builds for Pinpoint, a FaceTime coach that guides blind and Deaf makers through \
wiring a breadboard by touch. A real person will build exactly what you output, one step at a time, without \
seeing it, so the plan must be simple, safe and correct.

KIT ({kit.name}). Use only these part ids, within the counts:
{kit.catalog_text()}
Plus: an Arduino UNO R3, an 830-hole breadboard, USB cable, and plenty of male-male and female-male jumper wires.

ENDPOINTS (the `to` of a connection):
- Breadboard holes "E10": column A-J + row 1-63. Holes A-E in one row are connected; F-J in one row are \
connected; the groove between E and F separates them. Use rows 1-{MAX_TOUCH_ROW} only, and prefer A-E (the \
builder's side). Holes E and F are beside the groove.
- Rails: "RAIL:N+" and "RAIL:N-", the plus and minus strips along the builder's edge. Each is one long \
connected strip; any number of leads and wires can share a rail. Never use the far rails.
- Arduino sockets: "UNO:D2".."UNO:D13", "UNO:A0".."UNO:A5", "UNO:5V", "UNO:3V3", "UNO:GND", "UNO:VIN". \
Do not use D0/D1 (USB serial). PWM pins: D3, D5, D6, D9, D10, D11.

STEPS:
- "place" a part: `ref` is its ref from `parts`; `connections` lists EVERY lead of that part (exact lead \
names from the kit) and where it goes. Parts with layout=free can have each lead in any hole. layout=inline \
parts have legs fixed in a line: their leads go in consecutive rows on one side of the groove, in the listed \
lead order. layout=straddle (button): leg_a in E<r> and leg_b in F<r+2>, or the mirror. layout=wires parts \
(sensor modules, servo) connect each lead straight to a rail or Arduino socket with a female-male jumper wire.
- "wire" a jumper wire: `ref` empty, exactly two connections with leads "end1" and "end2".
- One lead or wire end per breadboard hole. A wire's two ends must not be in the same connected strip.
- Typical order: 5V->RAIL:N+ and GND->RAIL:N- first (only the ones you need), then each part followed by \
its wires.
- Every LED (and each colour of the RGB LED) needs a 220 ohm resistor in series. A sensor read by an analog \
pin that is a plain resistive part (photoresistor) needs a 10k resistor to make a voltage divider. Never \
connect 5V/3V3/VIN to GND, and never connect a signal pin straight to 5V or GND. Buttons and tilt switches \
go between a pin and GND, read with INPUT_PULLUP.
- Keep it small: at most about 30 connections in total, fewer is better. Reuse the layouts below: \
LED anode in E<r>, cathode in E<r+1>, resistor from D<r+1> to RAIL:N-, and a wire from the pin to D<r>.
- `why`: one short, complete, friendly sentence a blind builder hears, about what the step does (e.g. "This \
wire lets pin 9 switch the LED on and off."). No hole names in it.

SKETCH: a complete, compilable Arduino UNO sketch. Serial.begin(9600). Print one line of key=value pairs \
with integer values (e.g. "dist=42" or "x=512 click=0") at least 5 times a second, so Pinpoint can test the \
build over USB. `test.key` is the key that proves it works; expect "varies" if it changes when the builder \
interacts, or "present" if the line only appears after an action. Use only these libraries if needed: \
{', '.join(kit.libraries)}, and list them in `libraries`.

DECISION:
- "build": design it. Interpret the request generously and pick the simplest build that delivers the idea \
with the kit (e.g. "alert me when my cat is near the snacks" -> distance sensor + buzzer).
- "library": the request is essentially one of these tested projects; set library_id and leave the plan \
minimal (empty lists, empty strings):
{lib}
- "decline": impossible or unsafe with this kit (mains electricity, heating, batteries beyond USB power, \
parts the kit doesn't have that matter to the idea). `reason` says why in one friendly sentence and suggests \
the closest thing you could build instead.
`reason` is spoken to the builder: for "build", one sentence on what you'll make."""


@dataclass
class PlanResult:
    plan: BuildPlan | None
    message: str               # what Pinpoint says about it
    source: str                # "planner" | "library" | "none"
    attempts: int = 0
    errors: list[str] | None = None
    token: int | None = None   # echoes RequestPlan.token (set by the session)


class Planner:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", kit: Kit | None = None,
                 client=None, timeout_s: float = 120.0, save_dir: Path | None = None, library_first: bool = True):
        self.library_first = library_first
        self.kit = kit or load_kit()
        self.model = model
        self.effort = effort
        self._client = client
        self._timeout_s = timeout_s
        self.save_dir = save_dir
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=self._timeout_s, max_retries=1)
        return self._client

    # -------------------------------------------------------------- Claude calls
    def _call(self, messages: list, schema: dict, system: str):
        """One structured request. Returns (parsed dict or None, response content for the history)."""
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            log.warning("Planner rate limited: %s", e)
            return None, None
        except anthropic.APIStatusError as e:
            log.error("Planner API error %s: %s", e.status_code, e.message)
            return None, None
        except anthropic.APIConnectionError as e:
            log.error("Planner connection error: %s", e)
            return None, None
        u = getattr(resp, "usage", None)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += sum(getattr(u, k, 0) or 0 for k in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            log.warning("Planner request declined by safety classifier")
            return None, None
        if resp.stop_reason == "max_tokens":
            log.warning("Planner output was cut off")
            return None, resp.content
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return json.loads(text), resp.content
        except json.JSONDecodeError:
            log.error("Planner returned non-JSON: %r", text[:200])
            return None, resp.content

    def _save(self, request: str, result: PlanResult, raw: list) -> None:
        if self.save_dir is None:
            return
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            stem = self.save_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-plan"
            stem.with_suffix(".json").write_text(json.dumps({
                "request": request, "source": result.source, "message": result.message, "errors": result.errors,
                "plan": result.plan.to_dict() if result.plan else None, "raw": raw}, indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("Could not save plan record: %s", e)

    # -------------------------------------------------------------- public
    def plan(self, request: str, previous: BuildPlan | None = None) -> PlanResult:
        """Blocking (seconds to a minute). Run on a worker thread."""
        if previous is None and self.library_first:
            known = library.confident_match(request)
            if known is not None:          # "a blinking light": tested design, no LLM call, no credits
                log.info("Request %r matches library project %s; skipping the planner", request, known.id)
                res = PlanResult(known, f"Let's build the {known.title}. {known.summary}", "library")
                self._save(request, res, [])
                return res
        user = f'The builder said: "{request}"'
        if previous is not None:
            user = (f"Earlier plan (the builder wants changes):\n{json.dumps(previous.to_dict())}\n\n"
                    f'The builder now said: "{request}"\nRedesign accordingly.')
        messages: list = [{"role": "user", "content": user}]
        system = _system(self.kit)
        raw: list = []
        errors: list[str] = []
        for attempt in range(1 + MAX_REPAIRS):
            out, content = self._call(messages, _SCHEMA, system)
            raw.append(out)
            if out is None:
                break
            decision = out.get("decision")
            if decision == "decline":
                res = PlanResult(None, out.get("reason") or "I can't build that safely with this kit.", "none", attempt + 1)
                self._save(request, res, raw)
                return res
            if decision == "library":
                plan = library.get(out.get("library_id", ""))
                if plan is not None:
                    res = PlanResult(plan, out.get("reason") or f"Let's build the {plan.title}.", "library", attempt + 1)
                    self._save(request, res, raw)
                    return res
                errors = [f"library_id '{out.get('library_id')}' does not exist"]
            else:
                plan = BuildPlan.from_dict(out.get("plan") or {}, source="planner")
                errors = validate(plan, self.kit)
                if not errors:
                    res = PlanResult(plan, out.get("reason") or plan.summary, "planner", attempt + 1)
                    self._save(request, res, raw)
                    return res
            log.info("Plan attempt %d rejected: %s", attempt + 1, "; ".join(errors))
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "The plan checker found these problems. Fix all of them and output the complete "
                             "corrected result:\n- " + "\n- ".join(errors)})
        return self.fallback(request, errors, raw)

    def fallback(self, request: str, errors: list[str] | None = None, raw: list | None = None) -> PlanResult:
        plan = library.match(request)
        if plan is not None:
            res = PlanResult(plan, f"I'll use a design I know works: the {plan.title}. {plan.summary}", "library",
                             errors=errors)
        else:
            res = PlanResult(None, "I couldn't come up with a safe design for that. " + library.menu_text(), "none",
                             errors=errors)
        self._save(request, res, raw or [])
        return res

    def fix_sketch(self, plan: BuildPlan, compiler_output: str) -> str | None:
        """The sketch didn't compile: ask for a corrected one (wiring stays the same)."""
        user = (f"This Arduino UNO sketch failed to compile. Fix it without changing which pins are used "
                f"(the wiring is already built). Keep Serial.begin(9600) and the '{plan.test.key}=' output.\n\n"
                f"Sketch:\n{plan.sketch}\n\nCompiler output:\n{compiler_output[-4000:]}")
        out, _ = self._call([{"role": "user", "content": user}], _SKETCH_SCHEMA,
                            "You fix Arduino sketches. Output the complete corrected sketch.")
        if not out or not out.get("sketch", "").strip():
            return None
        return out["sketch"]


class OfflinePlanner(Planner):
    """No Claude: library projects only."""

    def plan(self, request: str, previous: BuildPlan | None = None) -> PlanResult:
        return self.fallback(request)

    def fix_sketch(self, plan: BuildPlan, compiler_output: str) -> str | None:
        return None
