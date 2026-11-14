"""
Pinpoint as a LiveKit agent for `chert dev` (Chert's CLI flow).

    Terminal 1:  python -m backend.chert_agent start      # this worker + dashboard on :8000
    Terminal 2:  chert dev --agent pinpoint                # admin PowerShell on Windows

`chert dev` keeps a "warm room" with this agent dispatched into it and, when someone
FaceTimes the line, has Chert's Mac join that room as `chert-bridge-<call id>`.
Rules from Chert's developer guide that this file follows:
  * A job can start with no caller (warm room): load models, but stay silent.
  * Greet only once the room metadata has "chert_bridge_ready": true. The Mac ends
    the call if it hears nothing within 10 s of that.
  * Only this participant publishes audio (and video, so the caller sees the captions).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
import time

from backend import config  # noqa: F401  (loads .env into os.environ before LiveKit reads it)
from backend.config import Settings

log = logging.getLogger("pinpoint.agent")

AGENT_NAME = os.environ.get("AGENT_NAME", "pinpoint")
BRIDGE_READY_WAIT_S = 45.0


def bridge_ready(metadata: str | None) -> bool:
    try:
        return json.loads(metadata or "{}").get("chert_bridge_ready") is True
    except (ValueError, AttributeError):
        return False


def _build_session(room, project: str):
    """Heavy setup (MediaPipe, TTS cache, Claude client). Runs in a worker thread."""
    from backend import chert_adapter
    from backend.livekit_transport import LiveKitTransport
    from backend.session import Session

    s = Settings()
    transport = LiveKitTransport.for_room(room, stt_model=s.stt_model, messaging=chert_adapter.Messaging())
    session = Session(transport, s, project)
    # Render the opening lines first, so the greeting is instant when FaceTime is ready.
    try:
        opening = [a.text for a in copy.deepcopy(session.engine).start() if getattr(a, "text", None)]
        for text in opening:
            session.tts.render(text, timeout_s=5.0)  # normally cached by _warm_tts already
    except Exception:
        log.exception("Could not pre-render the greeting")
    return transport, session


async def entrypoint(ctx) -> None:
    from livekit.agents import AutoSubscribe

    from backend import main as web

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    log.info("Job %s in room %s (metadata %s)", ctx.job.id, ctx.room.name, ctx.job.metadata or "{}")
    loop = asyncio.get_running_loop()

    ended = asyncio.Event()
    ready = asyncio.Event()

    def on_left(p):
        if p.identity.startswith("chert-bridge-"):
            loop.call_soon_threadsafe(ended.set)

    def on_metadata(*_):
        if bridge_ready(ctx.room.metadata):
            loop.call_soon_threadsafe(ready.set)

    ctx.room.on("participant_disconnected", on_left)
    ctx.room.on("disconnected", lambda *a: loop.call_soon_threadsafe(ended.set))
    ctx.room.on("room_metadata_changed", on_metadata)

    # Build while the room is still warm, so nothing slow happens once a caller is on the line.
    transport, session = await loop.run_in_executor(None, _build_session, ctx.room, web.STATE["project"])
    await transport.attach(ctx.room)

    async def cleanup(*_):
        await loop.run_in_executor(None, session.stop)
        if web.STATE.get("session") is session:
            web.STATE["session"] = None

    ctx.add_shutdown_callback(cleanup)

    # Warm and doctor rooms can close with no caller ever joining: don't wait forever.
    waiter = asyncio.ensure_future(ctx.wait_for_participant())
    gone = asyncio.ensure_future(ended.wait())
    await asyncio.wait({waiter, gone}, return_when=asyncio.FIRST_COMPLETED)
    if not waiter.done() or waiter.exception() is not None:
        waiter.cancel()
        log.info("Room closed before a caller joined")
        ctx.shutdown("no caller")
        return
    gone.cancel()
    caller = waiter.result()
    log.info("Caller joined: %s", caller.identity)
    on_metadata()  # the signal may have arrived before we listened
    try:
        await asyncio.wait_for(ready.wait(), BRIDGE_READY_WAIT_S)
        log.info("FaceTime is ready: greeting")
    except asyncio.TimeoutError:
        log.warning("No chert_bridge_ready after %.0f s; starting anyway", BRIDGE_READY_WAIT_S)

    with web._lock:
        old = web.STATE.get("session")
        if old is not None and old is not session:
            old.stop()
        web.STATE["session"] = session
    session.start()  # greets, then runs the build guidance loop in its own thread

    await ended.wait()
    log.info("Call over: shutting the job down")
    ctx.shutdown("call ended")


def _serve_dashboard(host: str, port: int) -> None:
    import uvicorn

    from backend import main as web

    web.STATE["source"] = "chert"
    uvicorn.run(web.app, host=host, port=port, log_level="warning")


def _warm_tts(project: str) -> None:
    """Render every fixed phrase once, before taking jobs, so no call waits on text-to-speech."""
    from backend.config import ROOT
    from backend.guidance.engine import GuidanceEngine, Project
    from backend.guidance.speech import TTSWorker, common_phrases
    from backend.vision.shield import ShieldLayout

    s = Settings()
    tts = TTSWorker.shared(ROOT / "cache" / "tts", speak_locally=False)
    layout = ShieldLayout.load(s.layout_path)
    engine = GuidanceEngine(Project.load(s.projects_dir / f"{project}.yaml"), layout, mode=s.mode)
    phrases = [a.text for a in engine.start() if getattr(a, "text", None)] + common_phrases(list(layout.ports))
    todo = [p for p in phrases if not tts.clip_path(p).exists()]
    if todo:
        print(f"Preparing {len(todo)} speech clips (first run only)...", flush=True)
    for p in todo:
        tts.render(p, timeout_s=15.0)


def run() -> None:
    import sys

    from livekit.agents import WorkerOptions, cli

    from backend import main as web

    if len(sys.argv) == 1 or sys.argv[1] in ("start", "dev"):
        _warm_tts(web.STATE["project"])

    port = int(os.environ.get("PINPOINT_DASHBOARD_PORT", "8000"))
    threading.Thread(target=_serve_dashboard, args=("127.0.0.1", port), daemon=True).start()
    time.sleep(0.5)
    print(f"Pinpoint agent '{AGENT_NAME}'. Dashboard: http://localhost:{port}  "
          f"Next, in an admin PowerShell here: chert dev --agent {AGENT_NAME}")
    if len(sys.argv) == 1:
        sys.argv.append("start")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))


if __name__ == "__main__":
    run()
