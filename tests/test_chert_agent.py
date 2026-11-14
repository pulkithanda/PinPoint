"""The `chert dev` agent path, without LiveKit or Chert."""
import json

from backend.chert_agent import AGENT_NAME, bridge_ready
from backend.livekit_transport import LiveKitTransport


def test_bridge_ready_signal():
    assert bridge_ready(json.dumps({"chert_call": "c1", "chert_bridge_ready": True, "at": "2026-11-14T00:00:00Z"}))
    assert not bridge_ready(json.dumps({"chert_warm": True}))
    assert not bridge_ready("")
    assert not bridge_ready(None)
    assert not bridge_ready("not json")
    assert AGENT_NAME


def test_transport_for_existing_room():
    room = object()
    t = LiveKitTransport.for_room(room)
    t.start()  # job mode: must not spin up its own connection thread
    assert t._room is room and not t._attached
    assert t._is_bridge("chert-bridge-abc") and not t._is_bridge("agent-xyz")
    t._end("test")
    assert t.call_ended
