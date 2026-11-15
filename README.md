# Pinpoint

A FaceTime call that guides your hands, so blind and Deaf makers can build electronics with a standard starter kit (Arduino + Grove Base Shield). Our EmoryHacks 2026 project.

## How a build goes

You FaceTime Pinpoint and keep the call open the whole time. The phone doesn't need a stand.

1. **Pinpoint tells you what to pick up and where it goes, by touch.** For example: "Plug the distance sensor into D7. It's in the third row back from the edge nearest you, the second port from the left."
2. **When it needs to see something, it asks you to hold up the phone.** For example: "Pick up the phone and hold it about a hand's width above D7."
   - Pinpoint takes the photo automatically once the picture is steady and sharp.
   - If it can see the board's corner markers, it tells you how to aim: "Move the phone a little to your left."
   - Then it says "Got it. You can put the phone down."
3. **Claude checks the photo.** It confirms the right module, or that the cable is in the right port. If you used the wrong port, Pinpoint says something like "That looks like D8, not D6."
4. **You can ask questions at any time**, by voice or iMessage. For example, "is this plugged in the right way round?" Pinpoint asks for a photo and answers.
5. **After the last plug, you connect the USB cable.** Pinpoint reads the Arduino's sensor values and says "It works!" Then a recap arrives by iMessage.

There are two output modes:
- **Voice mode** (blind and low-vision): spoken directions.
- **Visual mode** (Deaf and hard of hearing): big captions burned into Pinpoint's FaceTime video, with the target port highlighted. Thumbs-up means next, an open palm means repeat.

`PINPOINT_GUIDANCE=live` switches to the original design: the phone sits on an overhead stand, and Pinpoint steers your fingertip in real time ("right two ports… stop, that's D7").

## Code map

| Path | What it does |
|---|---|
| `backend/main.py` | FastAPI server: Chert webhooks (`/chert/webhook`, `/chert/imessage/webhook`), dashboard, commands |
| `backend/chert_adapter.py` | **The only file that talks to Chert:** accepts `call.incoming` with LiveKit tokens; iMessage send, attachments, inbound signature check |
| `backend/chert_agent.py` | The LiveKit agent worker that `chert dev` dispatches into each call (greets on `chert_bridge_ready`), plus the dashboard |
| `backend/livekit_transport.py` | Call media: joins the call's LiveKit room, reads the caller's video and audio tracks, publishes Pinpoint's voice and a captioned video track |
| `backend/session.py` | One session per call: frame loop, photo on demand, Claude jobs, speech and captions |
| `backend/stt.py` | Speech-to-text with local faster-whisper (Chert has no transcription) |
| `backend/vision/capture.py` | Picks the steady, sharp frame and gives aiming hints |
| `backend/vision/shield.py` | Corner markers → board position → port map (`shield_layout.yaml`) |
| `backend/vision/hands.py` | MediaPipe fingertip and gestures |
| `backend/vision/verify.py` | Claude: identify the module, check the plug, answer questions |
| `backend/guidance/engine.py` | Step state machine, commands, recap |
| `backend/guidance/speech.py` / `overlay.py` | Speech timing and text-to-speech / captions and on-screen highlights |
| `backend/hardware/serial_monitor.py` | Reads `dist=… lcd=…` from the Arduino and diagnoses wrong ports |
| `backend/projects/*.yaml` | Build steps (proximity alert; colour reader is a stretch goal) |
| `arduino/proximity_alert/` | The sketch, uploaded before the demo |
| `backend/build/` | Elegoo flow: kit catalog, breadboard model + plan checker, Claude planner, project library, build engine |
| `backend/build_session.py` | One call in the Elegoo flow: planner, photo checks, arduino-cli upload, USB test |
| `tools/` | Model download, marker PDF, port calibration, terminal simulators (`sim_terminal` Elegoo, `sim_grove`), `setup_arduino`, `chert_setup` (check keys, register iMessage webhook), `fake_bridge` (pretend to be Chert for local tests) |

## Elegoo kit: build anything the judge asks for (default)

With `PINPOINT_KIT=elegoo` (the default), Pinpoint works with an ELEGOO UNO R3 Most Complete starter kit
and a breadboard, and nothing is scripted:

1. **"What would you like to build today?"** The caller describes anything ("something that beeps when my
   cat gets near the snacks").
2. **Claude designs it** from the parts in the kit (`backend/build/elegoo_most_complete.yaml`): parts, every
   wire, and the Arduino sketch. Code then checks the plan (`backend/build/board.py`): no shorts, a resistor
   on every LED, no floating legs, one lead per hole, everything within rows 1-30 so it can be counted by
   touch. Problems go back to Claude to fix (up to twice). If it still isn't safe, or Claude is offline,
   Pinpoint offers the closest of 15 hand-checked projects (`backend/build/library.yaml`: 5 beginner, 10
   advanced). Requests the kit can't do safely (mains power, heating) are politely declined with an
   alternative.
3. **Pinpoint reads the plan back** and asks "Shall we build it?". The caller can say yes, or ask for changes.
4. **Wiring by touch**, one step at a time: "Put the long leg into row 10, counting 10 from the left end, the
   hole right beside the groove, on your side." A setup photo learns how the board is facing; each part is
   photo-checked (resistors are identified by their colour bands); a final photo checks the whole circuit.
5. **Pinpoint programs the Arduino itself** with arduino-cli (the sketch is compiled, and fixed by Claude if
   needed, while the caller is still wiring), then **tests it over USB** ("cover the light sensor...") and
   says "It works!". The recap and the sketch arrive by iMessage.

Setup, once per laptop: install arduino-cli (`winget install ArduinoSA.CLI`), then
`python -m tools.setup_arduino` (AVR core + libraries, compiles every library project as a check).

Rehearse without a phone: `python -m tools.sim_terminal` (real Claude planner) or `--offline` (library
only). The Grove Base Shield flow is still there: `PINPOINT_KIT=grove`, simulator `python -m tools.sim_grove`.

## Run it with Chert, step by step

How the call works ([Chert docs](https://docs.trychert.com/facetime/introduction)):
- Chert sends a `call.incoming` webhook. Pinpoint replies "accept" with a LiveKit room and tokens.
- Chert's bridge then joins that LiveKit room and publishes the caller's camera and mic (`facetime-remote-video`, `facetime-remote-audio`).
- Pinpoint joins the same room. It reads those tracks and publishes its voice plus a captioned video. Chert has no captions API, so the captions live in that video.
- The recap and typed commands go through Chert's separate Messaging (iMessage) API.

**1. Install**, once per laptop:
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m tools.download_models   # optional pre-download; otherwise fetched automatically on startup
```
The Whisper speech model (~140 MB) downloads on the first call.
On Debian/Ubuntu Linux hosts (including remote development containers), install MediaPipe's native graphics libraries and eSpeak NG for generated call audio:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgles2 espeak-ng
```

**2. Get keys and fill in `.env`** (copy `.env.example`):
- **LiveKit:** create a free project at [cloud.livekit.io](https://cloud.livekit.io). Copy the `wss://` URL, API key and API secret.
- **Chert FaceTime:** dashboard → Developer → Project API keys. You need the `ftk_...` key and the project ID.
- **Chert Messaging:** the tenant signing secret, and the tenant slug if your account has several tenants.
- **Claude:** `ANTHROPIC_API_KEY`.
- **Recap recipient:** `CHERT_IMESSAGE_TO` = the builder's phone number. This is needed because Chert only gives a hash of the caller's number.

Then run `python -m tools.chert_setup check`. It mints a test token and calls `GET /v1/lines` to prove the keys work.

**3. Prepare the hardware.** See [Markers](#markers) below.
- Upload `arduino/proximity_alert/proximity_alert.ino`. It needs the Seeed *Grove Ultrasonic Ranger* and *Grove LCD RGB Backlight* libraries.
- Plug the Arduino's USB into this laptop.
- Stick the markers on, then run `python -m tools.calibrate_ports`.

**4 (recommended). The `chert dev` way: no tunnel, no webhook.**
```powershell
python -m backend.chert_agent start                                  # terminal 1: agent worker + dashboard on :8000
npx --yes @trychert/facetime@0.1.0-beta.20 dev --agent pinpoint      # terminal 2: ADMIN PowerShell (Windows needs it)
```
Wait for `"pinpoint" is in a waiting room`, then make a **FaceTime Video** call to the number it prints. Skip steps 4 (webhook) and 5 below.

**4 (alternative). Webhook backend and a public tunnel:**
```bash
python -m backend.main --source chert       # add --guidance live for the stand version
ngrok http 8000                            # or: cloudflared tunnel --url http://localhost:8000
```

**5. Connect Chert to the tunnel:**
- **FaceTime:** in the Chert dashboard, go to Workers & Lines → Integration. Set the HTTPS webhook to `https://<tunnel>/chert/webhook` and assign it to your line, with `customer_decides` and no default Assistant.
- **iMessage:** run `python -m tools.chert_setup subscribe-imessage https://<tunnel>/chert/imessage/webhook`. Put the printed secret in `.env` as `CHERT_MESSAGING_WEBHOOK_SECRET`, then restart the backend.

**6. Call.** FaceTime your Chert line, then open http://localhost:8000 to watch what Pinpoint sees. Say "voice mode" (or show a thumbs-up for visual mode), then follow the instructions. Commands: `ready`, `done`, `repeat`, `help`, `back`, `skip`, `what am I holding`, or any question, spoken or texted to the line.

**Testing without Chert:**
- `python -m tools.fake_bridge --say "voice mode" --say "ready"` plays Chert's part against a local `livekit-server --dev`. Set `LIVEKIT_URL=ws://127.0.0.1:7880`, `LIVEKIT_API_KEY=devkey` and `LIVEKIT_API_SECRET=secret`.
- `python -m backend.main --source webcam` runs the same flow on the laptop webcam and speakers.
- `python -m tools.sim_terminal` steps through a build in the terminal.
- `python -m pytest -q` runs the tests.

**Not in Chert's docs yet (ask them):**
- the FaceTime webhook signature header
- the call-ended event (Pinpoint detects the bridge leaving the room instead)
- the caller's plaintext number
- the frame rate and resolution of the caller's video

## Markers

The four printed squares tell Pinpoint where the shield is and which way round it is, so it can find each port in a photo.

1. **Check the print scale.** The scale bar on the page must measure 50 mm. If it doesn't, reprint at 100% ("actual size", not "fit to page").
2. **Cut out the four squares.** Keep the thin white border around each black square, because the detector needs that border.
3. **Sit the board the way the builder will use it,** with the side they reach from as "near". Then attach one marker at each corner of the Grove Base Shield. Use small loops of tape behind the marker, or tape over the white border only, never over the black pattern.
   - id 0 at the far-left corner
   - id 1 at the far-right corner
   - id 2 at the near-right corner
   - id 3 at the near-left corner
4. **Keep them flat and upright,** with the printed id label readable from the builder's seat. Don't cover any port. If the shield's corners are crowded, stick the markers on a piece of card under the Arduino, just outside each corner. They only need to stay fixed relative to the board.
5. **Measure** the distance between marker centres in mm and enter it in `backend/vision/shield_layout.yaml` under `markers`. Marker 0 is `[0, 0]`, x goes right and y goes toward the builder.
6. **Calibrate:** run `python -m tools.calibrate_ports`, hold the camera over the board, and click the centre of each port when prompted. Press `s` to save.

## Wording

Say "live hand guidance on standard maker hardware over a normal video call, with no extra hardware". Don't say "first ever". Say "designed with" blind and Deaf makers, not "approved by", unless a blind or Deaf user has actually tested it.
