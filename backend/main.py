"""
FastAPI app:
Chert webhooks, per call sessions and the live dashboard for judges
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import cv2
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.config import ROOT, Settings
from backend import chert_adapter
from backend.livekit_transport import LiveKitTransport
from backend.session import Session
from backend.transport import WebcamTransport

log = logging.getLogger("pinpoint")
app = FastAPI(title="Pinpoint")

STATE: dict = {"session": None, "source": "webcam", "project": "proximity_alert", "messaging": None}
_lock = threading.Lock()


def _replace_session(make) -> None:
    """One call at a time (Chert: one active call per line). Stop the old session first."""
    with _lock:
        if STATE["session"] is not None:
            STATE["session"].stop()
            STATE["session"] = None
        STATE["session"] = make()


def _webcam_session() -> Session:
    s = Settings()
    return Session(WebcamTransport(s.camera_index), s, STATE["project"]).start()


def _call_session(call: chert_adapter.IncomingCall) -> Session:
    s = Settings()
    transport = LiveKitTransport(chert_adapter.LIVEKIT_URL, call.agent_token, stt_model=s.stt_model,
                                 messaging=STATE["messaging"], bridge_identity=call.bridge_identity,
                                 agent_identity=chert_adapter.AGENT_IDENTITY)
    return Session(transport, s, STATE["project"]).start()


@app.on_event("startup")
def _startup() -> None:
    STATE["messaging"] = chert_adapter.Messaging()
    if STATE["source"] == "webcam":
        STATE["session"] = _webcam_session()


@app.post("/chert/webhook")
async def chert_webhook(request: Request):
    """FaceTime line Integration webhook. `call.incoming` must be answered within 5 s."""
    event = await request.json()
    etype = event.get("type")
    if etype != "call.incoming":
        log.info("Chert FaceTime event %s", etype)
        return JSONResponse({"ok": True})
    try:
        call = chert_adapter.accept_incoming(event)
    except RuntimeError as e:
        log.error("Declining call: %s", e)
        return JSONResponse({"action": "decline"})
    log.info("Accepting call %s into LiveKit room %s", call.call_id, call.room)
    # Loading models takes ~1-2 s, so start the session after replying.
    threading.Thread(target=_replace_session, args=(lambda: _call_session(call),), daemon=True).start()
    return JSONResponse(call.accept)


@app.post("/chert/imessage/webhook")
async def chert_imessage_webhook(request: Request):
    """Chert Messaging API `message.received`: typed commands/questions (and the recap recipient)."""
    raw = await request.body()
    if chert_adapter.MESSAGING_WEBHOOK_SECRET and not chert_adapter.verify_messaging_signature(raw, request.headers):
        return JSONResponse({"ok": False, "error": "bad signature"}, status_code=401)
    sender, text = chert_adapter.parse_inbound_message(await request.json())
    if sender and STATE["messaging"] is not None:
        STATE["messaging"].recipient = sender  # reply to whoever is texting the line
    sess = STATE["session"]
    if text and sess is not None:
        sess.transcript_log.append(f"> {text}")
        sess.command(text)
    return JSONResponse({"ok": True})


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "dashboard" / "index.html")

@app.get("/state")
def state():
    sess = STATE["session"]
    return JSONResponse(sess.status() if sess else {"phase": "waiting for call"})

@app.post("/command")
async def command(request: Request):
    body = await request.json()
    sess = STATE["session"]
    if sess is None:
        return JSONResponse({"ok": False, "error": "no active session"}, status_code=409)
    return JSONResponse({"ok": True, "command": sess.command(body.get("text", ""))})

@app.post("/reset")
def reset():
    if STATE["source"] != "webcam":
        return {"ok": False, "error": "calls restart when the caller calls again"}
    _replace_session(_webcam_session)
    return {"ok": True}

@app.get("/stream.mjpg")
def stream():
    def gen():
        while True:
            sess = STATE["session"]
            img = sess.latest_annotated if sess else None
            if img is not None:
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1 / 15)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

def main() -> None:
    ap = argparse.ArgumentParser(description="Pinpoint backend")
    ap.add_argument("--source", choices=["webcam", "chert"], default="webcam")
    ap.add_argument("--project", default="proximity_alert")
    ap.add_argument("--guidance", choices=["snapshot", "live"], help="overrides PINPOINT_GUIDANCE")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    STATE["source"], STATE["project"] = args.source, args.project
    if args.guidance:
        os.environ["PINPOINT_GUIDANCE"] = args.guidance
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
