"""Chert integration pieces that don't need a live LiveKit server or Chert account."""
import hashlib
import hmac
import io
import json
import os
import time
import wave

import jwt
import numpy as np

from backend import chert_adapter as c
from backend.guidance.overlay import caption_card
from backend.livekit_transport import letterbox, wav_to_pcm


def test_accept_incoming_matches_chert_token_rules(monkeypatch):
    monkeypatch.setattr(c, "LIVEKIT_URL", "wss://demo.livekit.cloud")
    monkeypatch.setattr(c, "LIVEKIT_API_KEY", "devkey")
    monkeypatch.setattr(c, "LIVEKIT_API_SECRET", "x" * 32)
    call = c.accept_incoming({"type": "call.incoming", "data": {"call_id": "c716", "line_id": "l1"}})
    a = call.accept
    assert a["action"] == "accept" and a["livekit_url"] == "wss://demo.livekit.cloud"
    assert a["remote_participant_identity"] == c.AGENT_IDENTITY
    claims = jwt.decode(a["participant_token"], "x" * 32, algorithms=["HS256"])
    assert claims["sub"] == "chert-bridge-c716" != c.AGENT_IDENTITY        # bridge identity differs from agent
    assert isinstance(claims["nbf"], int) and isinstance(claims["exp"], int)
    assert 90 < claims["exp"] - claims["nbf"] <= 7200
    v = claims["video"]
    assert v["room"] == "pinpoint-c716" and v["roomJoin"] and v["canPublish"] and v["canSubscribe"]
    agent = jwt.decode(call.agent_token, "x" * 32, algorithms=["HS256"])
    assert agent["sub"] == c.AGENT_IDENTITY and agent["video"]["room"] == "pinpoint-c716"


def test_inbound_imessage_parse_and_signature():
    event = {"event_type": "message.received", "event_id": "e1",
             "data": {"chat": {"id": "ch1"}, "message": {
                 "sender_handle": {"handle": "+15551112222"},
                 "parts": [{"type": "text", "value": "is this the right"}, {"type": "media", "attachment_id": "a"},
                           {"type": "text", "value": "way round?"}]}}}
    assert c.parse_inbound_message(event) == ("+15551112222", "is this the right way round?")
    assert c.parse_inbound_message({"event_type": "message.sent"}) == (None, "")

    raw, secret, ts = json.dumps(event).encode(), "whsec_test", int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
    assert c.verify_messaging_signature(raw, {"X-Webhook-Signature": f"t={ts},v1={sig}"}, secret)
    assert c.verify_messaging_signature(raw, {"x-chert-signature": f"v1,{ts},{sig}"}, secret)
    assert not c.verify_messaging_signature(raw + b" ", {"X-Webhook-Signature": f"t={ts},v1={sig}"}, secret)
    old = ts - 600
    old_sig = hmac.new(secret.encode(), f"{old}.".encode() + raw, hashlib.sha256).hexdigest()
    assert not c.verify_messaging_signature(raw, {"X-Webhook-Signature": f"t={old},v1={old_sig}"}, secret)


def test_wav_to_pcm_resamples_to_48k_mono():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(np.zeros((22050, 2), np.int16).tobytes())
    pcm = wav_to_pcm(buf.getvalue())
    assert pcm.dtype == np.int16 and abs(len(pcm) - 48000) <= 2


def test_video_out_helpers():
    out = letterbox(np.zeros((480, 640, 3), np.uint8))
    assert out.shape == (720, 1280, 3)
    card = caption_card({"targets": ["D7"], "caption": "Plug the distance sensor into D7.", "title": "Proximity alert",
                         "n_steps": 3, "step_index": 0, "show": "Distance sensor -> D7"})
    assert card.shape == (720, 1280, 3) and card.mean() > 20


def test_dotenv_inline_comments(tmp_path, monkeypatch):
    import backend.config as cfg

    env = tmp_path / ".env"
    env.write_text('PT_A=0             # webcam index\nPT_B="a # b"\nPT_C=x\t# tab comment\n', encoding="utf-8")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    for k in ("PT_A", "PT_B", "PT_C"):
        monkeypatch.delenv(k, raising=False)
    cfg._load_dotenv()
    assert (os.environ["PT_A"], os.environ["PT_B"], os.environ["PT_C"]) == ("0", "a # b", "x")
