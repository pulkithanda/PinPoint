"""Slow path: vision-LLM checks on a single frame (module ID, cable-in-port).

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
from dataclasses import dataclass

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

    def _ask(self, image_b64: str, prompt: str, schema: dict) -> dict | None:
        a = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_SYSTEM,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-route on a safety-classifier refusal instead of failing the check
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
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
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.error("Claude returned non-JSON: %r", text[:200])
            return None

    def identify_module(self, frame_bgr: np.ndarray, candidates: list[str]) -> ModuleID:
        prompt = (
            "The builder is holding a Grove module up to the camera. Which module is it? "
            "Grove modules have their name printed on the board. "
            f"Candidates: {', '.join(candidates)}. "
            "Answer with exactly one candidate name, or 'unknown' if no module is clearly held."
        )
        out = self._ask(encode_jpeg(frame_bgr), prompt, _MODULE_SCHEMA)
        if not out:
            return ModuleID("unknown", False, "verification unavailable")
        name = out.get("module", "unknown")
        if name not in candidates:
            name = "unknown"
        return ModuleID(name, bool(out.get("confident")), out.get("description", ""))

    def check_plug(self, annotated_bgr: np.ndarray, module: str, port: str | list[str]) -> PlugCheck:
        ports = [port] if isinstance(port, str) else port
        target = ports[0] if len(ports) == 1 else "any of " + ", ".join(ports)
        prompt = (
            "Port labels have been drawn on the image in yellow; the expected port is circled in green. "
            f"Is the Grove cable from the {module} fully plugged into {target}? "
            "Report the label of the port that cable is actually in ('none' if it is not in any port)."
        )
        out = self._ask(encode_jpeg(annotated_bgr, max_side=1280), prompt, _PLUG_SCHEMA)
        if not out:
            return PlugCheck(None, None, "verification unavailable")
        plugged = {"yes": True, "no": False}.get(out.get("plugged", "unsure"))
        seen = out.get("port_seen") or "none"
        return PlugCheck(plugged, None if seen.lower() == "none" else seen, out.get("description", ""))
