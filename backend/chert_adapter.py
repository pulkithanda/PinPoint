"""
The ONLY file that talks to Chert. Two separate Chert APIs (https://docs.trychert.com):

FaceTime API (https://facetime.trychert.com/api)
  * Chert POSTs `call.incoming` to our line's Integration webhook. We must answer within
    5 s with {"action": "accept", "livekit_url", "participant_token", "remote_participant_identity"}.
  * Media does NOT come over webhooks: Chert's bridge joins that LiveKit room and publishes the
    caller as tracks `facetime-remote-video` / `facetime-remote-audio`. Our agent joins the same
    room (backend/livekit_transport.py) to read them and to publish voice + video back.
  * There is no captions API, so Deaf-mode captions are burned into the agent's video track.

Messaging / iMessage API (https://console.trychert.com/api)
  * Send:     POST /v1/send {phone, body, idempotency_key}  -> chat_id
  * Images:   POST /v1/attachments -> PUT bytes to upload_url -> POST /v1/chats/{chat_id}/messages
  * Inbound:  `message.received` webhooks (subscribe with tools/chert_setup.py), signed with
              X-Webhook-Signature: t=<ts>,v1=<hmac-sha256 hex of "<ts>.<raw body>">

Not documented by Chert (ask them): FaceTime webhook signing header, call-ended event,
plaintext caller number on inbound calls (only `from_handle_hash`).
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass

import backend.config  # noqa: F401  (loads .env before the settings below are read)

log = logging.getLogger(__name__)

# ---- configuration (.env) ----------------------------------------------------------------
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")                 # wss://<project>.livekit.cloud
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

FACETIME_API_BASE = os.environ.get("CHERT_FACETIME_API_BASE", "https://facetime.trychert.com/api")
FACETIME_API_KEY = os.environ.get("CHERT_FACETIME_API_KEY", "")  # ftk_...
CHERT_PROJECT_ID = os.environ.get("CHERT_PROJECT_ID", "")

MESSAGING_API_BASE = os.environ.get("CHERT_MESSAGING_API_BASE", "https://console.trychert.com/api")
MESSAGING_SECRET = os.environ.get("CHERT_MESSAGING_SECRET", "")  # tenant signing secret (bearer)
MESSAGING_TENANT = os.environ.get("CHERT_TENANT", "")
MESSAGING_WEBHOOK_SECRET = os.environ.get("CHERT_MESSAGING_WEBHOOK_SECRET", "")  # per-subscription secret
IMESSAGE_TO = os.environ.get("CHERT_IMESSAGE_TO", "")            # builder's phone / Apple ID for the recap

AGENT_IDENTITY = "pinpoint-agent"


# ======================================================================== FaceTime
@dataclass
class IncomingCall:
    call_id: str
    room: str
    bridge_identity: str
    agent_token: str
    accept: dict          # JSON body to return to Chert


def _token(identity: str, room: str, ttl_s: int = 3600) -> str:
    from livekit import api

    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    return (api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_ttl(datetime.timedelta(seconds=ttl_s))   # Chert: exp - nbf <= 7200
            .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
            .to_jwt())


def accept_incoming(event: dict) -> IncomingCall:
    """Build the accept response for a `call.incoming` webhook (must be returned within 5 s)."""
    data = event.get("data") or {}
    call_id = data.get("call_id") or event.get("call_id") or uuid.uuid4().hex
    room = f"pinpoint-{call_id}"
    bridge = f"chert-bridge-{call_id}"   # token `sub` must differ from the agent identity
    accept = {
        "action": "accept",
        "livekit_url": LIVEKIT_URL,
        "participant_token": _token(bridge, room),
        "remote_participant_identity": AGENT_IDENTITY,
    }
    return IncomingCall(call_id, room, bridge, _token(AGENT_IDENTITY, room), accept)


def facetime_headers(idempotency_key: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {FACETIME_API_KEY}", "X-Chert-Project-Id": CHERT_PROJECT_ID}
    if idempotency_key:
        h["Idempotency-Key"] = idempotency_key
    return h


# ======================================================================== Messaging (iMessage)
def verify_messaging_signature(raw_body: bytes, headers, secret: str = MESSAGING_WEBHOOK_SECRET) -> bool:
    """HMAC check for `message.received` webhooks (from Chert's docs)."""
    header = headers.get("X-Webhook-Signature") or headers.get("x-chert-signature") or ""
    m = re.match(r"^t=(\d+),v1=([a-f0-9]+)$", header) or re.match(r"^v1,(\d+),([a-f0-9]+)$", header)
    if not m:
        return False
    ts, provided = int(m.group(1)), m.group(2).lower()
    if abs(time.time() - ts) > 300:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided, expected)


def parse_inbound_message(event: dict) -> tuple[str | None, str]:
    """`message.received` -> (sender handle, text)."""
    if event.get("event_type", event.get("type")) != "message.received":
        return None, ""
    msg = (event.get("data") or {}).get("message") or {}
    sender = (msg.get("sender_handle") or {}).get("handle")
    text = " ".join(p.get("value", "") for p in msg.get("parts") or [] if p.get("type") == "text").strip()
    return sender, text


class Messaging:
    """Sends iMessages (recap + photos) on a background thread so the call never waits."""

    def __init__(self):
        self.recipient: str | None = IMESSAGE_TO or None
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="chert-imessage", daemon=True).start()

    @property
    def configured(self) -> bool:
        return bool(MESSAGING_SECRET)

    def headers(self) -> dict:
        h = {"Authorization": f"Bearer {MESSAGING_SECRET}"}
        if MESSAGING_TENANT:
            h["x-chert-tenant"] = MESSAGING_TENANT
        return h

    def send(self, text: str, images_jpeg: list[bytes] | None = None) -> None:
        if not self.recipient:
            log.warning("No iMessage recipient: set CHERT_IMESSAGE_TO or text the line first.\n%s", text)
            return
        self._q.put((self.recipient, text, list(images_jpeg or [])[-4:]))

    def _run(self) -> None:
        import httpx

        while True:
            to, text, images = self._q.get()
            if not self.configured:
                log.info("[dry run: iMessage to %s] %s (+%d images)", to, text[:80], len(images))
                continue
            try:
                with httpx.Client(base_url=MESSAGING_API_BASE, headers=self.headers(), timeout=20.0) as http:
                    r = http.post("/v1/send", json={"phone": to, "body": text, "idempotency_key": uuid.uuid4().hex})
                    r.raise_for_status()
                    chat_id = r.json().get("chat_id")
                    if images and chat_id:
                        parts = [{"type": "media", "attachment_id": self._upload(http, img, i)}
                                 for i, img in enumerate(images)]
                        body: dict = {"message": {"parts": parts}}
                        if len(parts) >= 2:
                            body["parallel_attachments"] = True
                        http.post(f"/v1/chats/{chat_id}/messages", json=body).raise_for_status()
                log.info("iMessage recap sent to %s", to)
            except httpx.HTTPStatusError as e:
                log.error("iMessage send failed: %s %s", e.response.status_code, e.response.text[:200])
            except httpx.HTTPError as e:
                log.error("iMessage send failed: %s", e)

    def _upload(self, http, jpg: bytes, i: int) -> str:
        r = http.post("/v1/attachments", json={"filename": f"step{i + 1}.jpg", "content_type": "image/jpeg",
                                                "size_bytes": len(jpg)})
        r.raise_for_status()
        info = r.json()
        put = http.put(info["upload_url"], content=jpg,
                       headers={**self.headers(), "Content-Type": "image/jpeg"})
        put.raise_for_status()
        return info["attachment_id"]
