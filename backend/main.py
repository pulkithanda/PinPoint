"""
FastAPI app:
Chert webhooks, per call sessions and the live dashboard for judges
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.config import ROOT, Settings
from backend.chert_adapter import ChertTransport
from backend.session import Session
from backend.transport import WebcamTransport

log = logging.getLogger("pinpoint")
app = FastAPI(title="Pinpoint")

STATE: dict = {"session": None, "source": "webcam", "project": "proximity_alert", "chert": None}

def _new_session() -> Session:
    s = Settings()
    if STATE["source"] == "chert":
        transport = STATE["chert"] = ChertTransport()
    else:
        transport = WebcamTransport(s.camera_index)
    return Session(transport, s, STATE["project"]).start()

@app.on_event("startup")
def _startup() -> None:
    if STATE["source"] == "webcam":
        STATE["session"] = _new_session()

@app.post("/chert/webhook")
async def chert_webhook(request: Request):
    event = await request.json()
    # One session per call: start a fresh one when a call begins.
    if event.get("type") == "call.started" or STATE["session"] is None:
        if STATE["session"] is not None:
            STATE["session"].stop()
        STATE["session"] = _new_session()
    return JSONResponse(STATE["chert"].handle_webhook(event))

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
    if STATE["session"] is not None:
        STATE["session"].stop()
    STATE["session"] = _new_session()
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    STATE["source"], STATE["project"] = args.source, args.project
    if args.source == "chert":
        STATE["chert"] = ChertTransport()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
