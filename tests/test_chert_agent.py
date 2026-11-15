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


def test_greets_only_after_bridge_ready():
    """Chert's rule: never greet before chert_bridge_ready, however long FaceTime takes."""
    import asyncio

    from backend.chert_agent import wait_for_bridge_ready

    async def scenario(signal_after_s, end_instead=False):
        ready, ended = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.call_later(signal_after_s, (ended if end_instead else ready).set)
        t0 = loop.time()
        ok = await wait_for_bridge_ready(ready, ended)
        return ok, loop.time() - t0

    # Slow FaceTime connect: must keep waiting (the old 5 s fallback greeted too early here).
    ok, waited = asyncio.run(scenario(0.3))
    assert ok and waited >= 0.25
    # Caller hangs up while connecting: no greeting.
    ok, _ = asyncio.run(scenario(0.05, end_instead=True))
    assert not ok

    async def already_ready():
        ready, ended = asyncio.Event(), asyncio.Event()
        ready.set()
        return await wait_for_bridge_ready(ready, ended)

    assert asyncio.run(already_ready())


def test_transport_for_existing_room():
    room = object()
    t = LiveKitTransport.for_room(room)
    t.start()  # job mode: must not spin up its own connection thread
    assert t._room is room and not t._attached
    assert t._is_bridge("chert-bridge-abc") and not t._is_bridge("agent-xyz")
    t._end("test")
    assert t.call_ended
