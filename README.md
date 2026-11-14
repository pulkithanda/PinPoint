# Pinpoint

A FaceTime call that guides your hands, so blind and Deaf makers can build electronics with a standard starter kit.

Pinpoint watches your hands over a video call and steers you, plug by plug, to the right port on an Arduino + Grove Base Shield. Blind makers get spoken directions ("right two ports... down one... stop, that's D7, plug it in"). Deaf makers get the target port circled, an arrow, and big captions. You don't need any special hardware.

Our EmoryHacks 2026 project.

## How it works

```
FaceTime call ──► chert_adapter ──► Session (one per call)
                                     │
             every frame (fast path) ├─ vision/shield.py   ArUco markers → homography → port map (mm)
                                     ├─ vision/hands.py    MediaPipe fingertip (landmark 8) + gestures
                                     ├─ guidance/engine.py step state machine → "left two ports"
                                     ├─ guidance/speech.py rate-limited TTS (blind mode)
                                     └─ guidance/overlay.py port highlight + arrow + captions (Deaf mode)
             on demand (slow path)   ├─ vision/verify.py   Claude: "which module is this?", "is it in D7?"
             after the last plug     └─ hardware/serial_monitor.py  dist=… readings from the Arduino
```

The geometry path steers your hand on every frame. The vision LLM only checks a placement or identifies a part, and it runs on a worker thread, so slow AI calls never hold up guidance.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m tools.download_models   # MediaPipe gesture recognizer (~8 MB)
python -m tools.make_markers      # writes markers/aruco_4x4.pdf
```

Optional `.env` in the repo root:

```
ANTHROPIC_API_KEY=...             # enables module ID + plug checks (without it, plug checks fall back to the serial test)
PINPOINT_CAMERA=0
PINPOINT_SERIAL_PORT=auto         # or COM5 / /dev/ttyACM0 / none
PINPOINT_MODE=                    # voice | visual | (blank = ask at the start of the call)
PINPOINT_PLACE_THEN_CHECK=0       # 1 = fallback if Chert video is laggy
PINPOINT_SONIFY=0                 # 1 = parking-sensor beep as the finger closes in
```

## Run

| What | Command |
|---|---|
| Unit tests | `python -m pytest -q` |
| Step through a build in the terminal with fake fingertip input | `python -m tools.sim_terminal` |
| Live on the laptop webcam, with the dashboard at http://localhost:8000 | `python -m backend.main --source webcam` |
| Live on a FaceTime call through Chert | `python -m backend.main --source chert`, then expose `/chert/webhook` with `ngrok http 8000` or `cloudflared` |
| Measure the port positions on our shield | `python -m tools.calibrate_ports` |

In the dashboard you can send commands with the buttons or the text box: `voice mode`, `visual mode`, `what am I holding`, `done`, `repeat`, `help`, `back`, `skip`. In visual mode, a thumbs-up means next/done and an open palm means repeat.

## Before the demo (hardware checklist)

1. **Flash** `arduino/proximity_alert/proximity_alert.ino`. It needs the Seeed *Grove Ultrasonic Ranger* and *Grove LCD RGB Backlight* libraries. Plug the parts in by hand once and check that the beeps, the LCD colours, and the `dist=… lcd=…` lines on the serial monitor all work.
2. **Print** `markers/aruco_4x4.pdf` at 100% scale and check the 50 mm scale bar. Tape the markers upright on the shield corners: 0 far-left, 1 far-right, 2 near-right, 3 near-left.
3. **Measure** the marker centres into `backend/vision/shield_layout.yaml`, then run `python -m tools.calibrate_ports` and click each port. The current port coordinates are **placeholders**.
4. **Check the port labels** on the real shield against the sketch (sensor on D7, buzzer on D6, LCD on I2C) and against `backend/projects/proximity_alert.yaml`.
5. Set up the overhead phone stand and the lamp. Make sure all 4 markers are visible on the dashboard.

## Chert integration status

`backend/chert_adapter.py` is the only file that talks to Chert. We haven't seen the Chert API yet, so the adapter assumes webhook JSON events (`call.started`, `video.frame` with a base64 JPEG, `transcript`, `call.ended`). Outgoing audio, outgoing video, and the iMessage recap are marked `TODO(chert)` and only log for now. Everything else runs today on the webcam transport.

If Chert can't deliver live frames or send video back, set `PINPOINT_PLACE_THEN_CHECK=1`. Pinpoint then describes each port by touch ("the second row back from the edge nearest you, the second port from the left") and confirms each step from a snapshot plus the serial readings.

## Team split

| Role | Owns | First milestone |
|---|---|---|
| A: Chert integration | `chert_adapter.py`, `main.py`, iMessage recap | Echo test: a frame comes in and speech goes back out on a real call |
| B: Vision | `vision/shield.py`, `vision/hands.py`, `tools/calibrate_ports.py` | The live webcam dashboard shows "Fingertip at D7" correctly |
| C: Guidance + AI | `guidance/engine.py`, `guidance/speech.py`, `vision/verify.py`, project YAMLs | `tools.sim_terminal` runs through the whole build |
| D: Hardware + Deaf mode + demo | Arduino sketch, `serial_monitor.py`, markers, stand, `overlay.py`, video, Devpost | The alert device works when plugged in by hand, and serial readings reach the backend |

## Wording

Don't say "first ever". Say "live hand guidance on standard maker hardware over a normal video call, with no extra hardware". Also say "designed with" blind and Deaf makers, not "approved by", unless a blind or Deaf user has actually tested it.
