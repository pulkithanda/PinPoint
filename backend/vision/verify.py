"""
Slow path: vision-LLM checks on a single frame (module ID, cable-in-port)

These calls take seconds, so they run on a worker thread and never block the
per-frame guidance loop. Each frame is annotated with port labels before it is
sent, so Claude can answer "which port is the cable in?" without having to read
the shield's tiny silkscreen.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ModuleID:
    module: str          # one of the candidate names, or "unknown"
    confident: bool
    description: str


@dataclass
class PlugCheck:
    plugged: bool | None   # None = could not tell
    port_seen: str | None  # port the newest cable appears to be in, if any
    description: str


_MODULE_SCHEMA = {
    "type": "object",
    "properties": {
        "module": {"type": "string", "description": "Exactly one of the candidate names, or 'unknown'."},
        "confident": {"type": "boolean"},
        "description": {"type": "string", "description": "One short sentence: what you see."},
    },
    "required": ["module", "confident", "description"],
    "additionalProperties": False,
}

_PLUG_SCHEMA = {
    "type": "object",
    "properties": {
        "plugged": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "port_seen": {"type": "string", "description": "Label of the port the module's cable is actually in, or 'none'."},
        "description": {"type": "string", "description": "One short sentence: what you see."},
    },
    "required": ["plugged", "port_seen", "description"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are the eyes of Pinpoint, a build coach that helps blind and Deaf makers plug Seeed Grove "
    "modules into a Grove Base Shield on an Arduino. The camera looks down at the desk. "
    "Answer only from what is visible in the image. If you cannot tell, say so: a wrong 'yes' "
    "misleads a blind builder, so prefer 'unsure' over guessing."
)


_BB_SYSTEM = (
    "You are the eyes of Pinpoint, a build coach that guides blind and Deaf makers through wiring an Arduino UNO "
    "and an 830-hole breadboard by touch. Breadboard holes are named by column letter A-J and row number 1-63 "
    "(e.g. E10); A-E and F-J are separated by a central groove; the plus and minus rails run along each long "
    "edge. Answer only from what is visible. A wrong 'correct' misleads a blind builder, so say 'unsure' "
    "rather than guess."
)


@dataclass
class BoardSetup:
    near_half: str | None        # "AE" | "FJ": which half of the breadboard is nearest the builder
    row1_end: str | None         # "left" | "right"
    near_outer_rail: str | None  # "+" | "-": the rail strip right at the builder's edge
    usb_left: bool | None        # Arduino USB socket points left
    description: str
    board_visible: bool | None = None     # a breadboard is in the photo at all
    long_side_toward: bool | None = None  # the breadboard's long side runs along the builder's edge
    fix: str = ""                         # what to move, in the builder's words, if the layout is wrong


@dataclass
class PlacementCheck:
    ok: bool | None              # None = couldn't tell
    problem: str                 # spoken fix if wrong
    seen: str


@dataclass
class BoardReview:
    ok: bool | None
    problems: list[str]
    description: str


_SETUP_PROMPT = (
    "The builder has put a breadboard and an Arduino UNO on the table and holds the phone above them. The phone "
    "may be held any way round, so the photo may be rotated: do NOT use the image edges to decide directions. "
    "Wanted layout: the Arduino sits alongside one LONG side of the breadboard (not at a short end), and the "
    "builder sits at the OTHER long side, facing the Arduino across the breadboard. Seen from the builder's seat, "
    "the Arduino's USB socket points to the builder's LEFT. "
    "Work out the builder's seat from where the Arduino is (opposite long side), then answer from the builder's "
    "point of view: is the layout as wanted? Which half of the breadboard is nearest the builder (A-E or F-J, "
    "from the printed letters)? Is row 1 at the builder's left or right? Along the builder's long edge, is the "
    "outermost rail strip plus (red line) or minus (blue line)? "
    "If the layout is wrong, `fix` says in one or two short, friendly spoken sentences what to move, in the "
    "builder's terms (e.g. 'Turn the Arduino round so its USB socket points to your left.'). Otherwise `fix` "
    "is empty."
)
_SETUP_SCHEMA = {
    "type": "object",
    "properties": {
        "near_half": {"type": "string", "enum": ["AE", "FJ", "unsure"]},
        "row1_end": {"type": "string", "enum": ["left", "right", "unsure"]},
        "near_outer_rail": {"type": "string", "enum": ["+", "-", "unsure"]},
        "board_visible": {"type": "boolean", "description": "Both a breadboard and an Arduino are in the photo."},
        "layout_ok": {"type": "boolean", "description": "The layout matches what's wanted."},
        "fix": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["near_half", "row1_end", "near_outer_rail", "board_visible", "layout_ok", "fix", "description"],
    "additionalProperties": False,
}
_PLACEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["correct", "wrong", "unsure"]},
        "problem": {"type": "string", "description": "If wrong: one or two short spoken sentences on what to fix. Else empty."},
        "seen": {"type": "string", "description": "One short sentence: what you see."},
    },
    "required": ["status", "problem", "seen"],
    "additionalProperties": False,
}
_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["looks_right", "has_problems", "unsure"]},
        "problems": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
    },
    "required": ["verdict", "problems", "description"],
    "additionalProperties": False,
}


def encode_jpeg(frame_bgr: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def claude_available() -> bool:
    """Best-effort check for Anthropic credentials (env key/token or an `ant auth login` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    home = os.path.expanduser("~")
    return os.path.isdir(os.path.join(home, ".config", "anthropic"))


class Verifier:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", timeout_s: float = 30.0):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=1)
        self.model = model
        self.effort = effort
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()
        # Every call's photo, prompt and answer, so model/effort changes can be replayed and compared later.
        save = os.environ.get("PINPOINT_SAVE_CLAUDE_CALLS", "1").strip().lower() not in ("0", "false", "no", "off")
        self.call_dir = (Path(__file__).resolve().parents[2] / "cache" / "claude_calls") if save else None

    def _record(self, kind: str, image_b64: str, prompt: str, schema: dict | None, resp, answer,
                system: str = _SYSTEM) -> None:
        u = getattr(resp, "usage", None)
        tin = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
            + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        tout = getattr(u, "output_tokens", 0) or 0
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += tin
            self.usage["output_tokens"] += tout
            total = dict(self.usage)
        log.info("Claude %s: %d in / %d out tokens (this session: %d calls, %d in / %d out)",
                 kind, tin, tout, total["calls"], total["input_tokens"], total["output_tokens"])
        if self.call_dir is None:
            return
        try:
            self.call_dir.mkdir(parents=True, exist_ok=True)
            stem = self.call_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}-{kind}"
            stem.with_suffix(".jpg").write_bytes(base64.standard_b64decode(image_b64))
            stem.with_suffix(".json").write_text(json.dumps({
                "kind": kind, "model": self.model, "effort": self.effort, "system": system, "prompt": prompt,
                "schema": schema, "answer": answer, "input_tokens": tin, "output_tokens": tout,
            }, indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("Could not save the Claude call record: %s", e)

    def _ask(self, image_b64: str, prompt: str, schema: dict | None, kind: str = "call",
             system: str = _SYSTEM) -> dict | str | None:
        """One image + prompt. Returns parsed JSON if `schema` is given, else plain text."""
        a = self._anthropic
        output_config: dict = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-route on a safety-classifier refusal instead of failing the check
                output_config=output_config,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        except a.RateLimitError as e:
            log.warning("Claude rate limited: %s", e)
            return None
        except a.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            return None
        except a.APIConnectionError as e:
            log.error("Claude connection error: %s", e)
            return None
        if resp.stop_reason == "refusal":
            log.warning("Claude declined the verification request")
            self._record(kind, image_b64, prompt, schema, resp, None, system)
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        if schema is None:
            out = text.strip() or None
        else:
            try:
                out = json.loads(text)
            except json.JSONDecodeError:
                log.error("Claude returned non-JSON: %r", text[:200])
                out = None
        self._record(kind, image_b64, prompt, schema, resp, out, system)
        return out

    # ------------------------------------------------------------ breadboard (Elegoo kit)
    def check_board_setup(self, frame_bgr: np.ndarray) -> BoardSetup:
        out = self._ask(encode_jpeg(frame_bgr, max_side=1280), _SETUP_PROMPT, _SETUP_SCHEMA, kind="setup",
                        system=_BB_SYSTEM)
        if not isinstance(out, dict):
            return BoardSetup(None, None, None, None, "verification unavailable")
        def pick(key, allowed):
            v = out.get(key)
            return v if v in allowed else None
        def flag(key):
            return out.get(key) if isinstance(out.get(key), bool) else None
        ok = flag("layout_ok")
        return BoardSetup(pick("near_half", ("AE", "FJ")), pick("row1_end", ("left", "right")),
                          pick("near_outer_rail", ("+", "-")), ok, out.get("description", ""),
                          flag("board_visible"), ok, (out.get("fix") or "").strip())

    def check_placement(self, frame_bgr: np.ndarray, expected: str, already: str, mode: str = "voice") -> PlacementCheck:
        audience = ("Say any problem as a blind builder can act on it: by touch and position (which leg, which "
                    "row, which side of the groove), never by colour alone."
                    if mode != "visual" else "Say any problem in one short caption.")
        prompt = (
            "Check one step of a breadboard build. The photo is taken from above with the top of the image "
            "pointing away from the builder.\n"
            f"Expected for this step:\n{expected}\n"
            + (f"Already built earlier (context only, don't judge):\n{already}\n" if already else "")
            + "Are this step's part and leads in exactly those holes, the right way round? Count rows from the "
            "printed numbers on the breadboard. " + audience
        )
        out = self._ask(encode_jpeg(frame_bgr, max_side=1568), prompt, _PLACEMENT_SCHEMA, kind="placement",
                        system=_BB_SYSTEM)
        if not isinstance(out, dict):
            return PlacementCheck(None, "", "verification unavailable")
        ok = {"correct": True, "wrong": False}.get(out.get("status", "unsure"))
        return PlacementCheck(ok, out.get("problem", "") or "", out.get("seen", "") or "")

    def review_board(self, frame_bgr: np.ndarray, expected: str, mode: str = "voice") -> BoardReview:
        prompt = (
            "Final check before the code is uploaded. Photo from above, top of the image pointing away from the "
            f"builder. The finished circuit should be exactly:\n{expected}\n"
            "List each difference you can actually see (wrong hole, missing wire, LED or buzzer backwards, wire to "
            "the wrong Arduino pin). Ignore jumper wire colours and routing. If something can't be seen clearly, "
            "don't list it."
            + (" Describe each problem by touch and position for a blind builder." if mode != "visual" else "")
        )
        out = self._ask(encode_jpeg(frame_bgr, max_side=1568), prompt, _REVIEW_SCHEMA, kind="review",
                        system=_BB_SYSTEM)
        if not isinstance(out, dict):
            return BoardReview(None, [], "verification unavailable")
        issues = [i for i in (out.get("problems") or []) if isinstance(i, str) and i.strip()]
        verdict = {"looks_right": True, "has_problems": False}.get(out.get("verdict", "unsure"))
        if verdict is False and not issues:
            verdict = None
        return BoardReview(verdict, issues, out.get("description", "") or "")

    def identify_part(self, frame_bgr: np.ndarray, candidates: dict[str, str]) -> ModuleID:
        """`candidates`: part id -> description. Reads resistor colour bands too."""
        listing = "\n".join(f"- {k}: {v}" for k, v in candidates.items())
        prompt = ("The builder is holding up an electronic part. Which one is it? For resistors, read the colour "
                  "bands (220 ohm = red red brown; 1k = brown black red; 10k = brown black orange; a fourth band "
                  "may be gold).\n" + listing + "\nAnswer with one id exactly, or 'unknown'.")
        out = self._ask(encode_jpeg(frame_bgr), prompt, _MODULE_SCHEMA, kind="identify", system=_BB_SYSTEM)
        if not isinstance(out, dict):
            return ModuleID("unknown", False, "verification unavailable")
        name = out.get("module", "unknown")
        return ModuleID(name if name in candidates else "unknown", bool(out.get("confident")), out.get("description", ""))

    def identify_module(self, frame_bgr: np.ndarray, candidates: list[str]) -> ModuleID:
        prompt = (
            "The builder is holding a Grove module up to the camera. Which module is it? "
            "Grove modules have their name printed on the board. "
            f"Candidates: {', '.join(candidates)}. "
            "Answer with exactly one candidate name, or 'unknown' if no module is clearly held."
        )
        out = self._ask(encode_jpeg(frame_bgr), prompt, _MODULE_SCHEMA, kind="identify")
        if not isinstance(out, dict):
            return ModuleID("unknown", False, "verification unavailable")
        name = out.get("module", "unknown")
        if name not in candidates:
            name = "unknown"
        return ModuleID(name, bool(out.get("confident")), out.get("description", ""))

    def check_plug(self, frame_bgr: np.ndarray, module: str, port: str | list[str], labeled: bool = True) -> PlugCheck:
        """`labeled`: port labels were drawn on the frame (corner markers were visible).
        Close-up photos without markers rely on the shield's own printed labels instead."""
        ports = [port] if isinstance(port, str) else port
        target = ports[0] if len(ports) == 1 else "any of " + ", ".join(p.replace("_", " ") for p in ports)
        how = ("Port labels have been drawn on the image in yellow; the expected port is circled in green. "
               if labeled else
               "This is a close-up of a Grove Base Shield; read the port names printed next to each socket "
               "(D2-D8, A0-A3, I2C, UART). ")
        prompt = (
            how +
            f"Is the Grove cable from the {module} fully plugged into {target}? "
            "Report the label of the port that cable is actually in ('none' if it is not in any port). "
            "Write I2C ports as I2C_1..I2C_4 if labels are drawn, otherwise just I2C."
        )
        out = self._ask(encode_jpeg(frame_bgr, max_side=1280), prompt, _PLUG_SCHEMA, kind="plug")
        if not isinstance(out, dict):
            return PlugCheck(None, None, "verification unavailable")
        plugged = {"yes": True, "no": False}.get(out.get("plugged", "unsure"))
        seen = (out.get("port_seen") or "none").strip()
        if seen.lower() == "none":
            seen = None
        elif seen.upper().startswith("I2C"):
            # Unlabelled close-ups can't tell the four I2C ports apart; any of them counts.
            seen = next((p for p in ports if p.upper() == seen.upper()), None) or \
                next((p for p in ports if p.startswith("I2C")), seen)
        return PlugCheck(plugged, seen, out.get("description", ""))

    def answer_question(self, frame_bgr: np.ndarray, question: str, context: str, mode: str) -> str | None:
        """Free-form question from the builder, answered from one photo."""
        audience = ("The answer will be read aloud to a blind builder: describe things by touch, shape and "
                    "position (left/right/near/far), never by colour alone."
                    if mode != "visual" else
                    "The answer will be shown as a caption to a Deaf builder: keep it short and plain.")
        prompt = (
            f"Context: {context}\n{audience}\n"
            f"The builder asks: \"{question}\"\n"
            "Answer in at most two short sentences, based only on what the photo shows. "
            "If the photo doesn't show enough to answer, say what to point the camera at instead."
        )
        out = self._ask(encode_jpeg(frame_bgr), prompt, None, kind="question")
        return out if isinstance(out, str) else None
