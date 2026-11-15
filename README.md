<div align="center">

# PinPoint

### Your idea. Your parts. Your AI hardware co-builder on FaceTime.

**From loose components to a working circuit: one conversation at a time.**

[Watch the demo](https://www.youtube.com/watch?v=Aq5WXmQQooo) · [Devpost](https://devpost.com/software/pinpoint-9pyk0w) · [EmoryHacks 2026](https://emory-hacks-2026.devpost.com/)

</div>

---

## What is PinPoint?

**PinPoint is an AI-powered hardware building companion that you can FaceTime.** Tell it what you want to make, show it the components you have, and follow step-by-step guidance, from planning and placing wires to checking your circuit and testing supported Arduino builds.

The idea is simple: you shouldn't need an engineering degree, an experienced friend, or seventeen tutorial tabs to start building electronics. PinPoint is for **beginners, hobbyists, experienced makers, student teams, and educators**. It is also **inclusive by design**, supporting different ways to receive instructions and interact with the agent.

Built by a two-person team at **EmoryHacks 2026** for the **Rogue** main track, with a focus on the **Chert** and **Coder for Good** side challenges.

## How a build works

1. **Describe your idea.** Say what you're trying to build, from a simple LED circuit to a sensor driven alarm. PinPoint asks follow-up questions and works with the components you have.
2. **Get a feasible plan.** Claude proposes components, a wiring plan and an Arduino sketch. A separate circuit checker examines constraints before the build instructions are presented.
3. **Show your workspace.** When asked, hold your camera above your board. Camera-frame selection, visual checks, and four printed reference markers help relate what the AI sees to physical positions.
4. **Build together.** Follow instructions one step at a time. Ask questions, repeat a step, or request a wiring check without losing the conversation's current context.
5. **Verify the result.** On compatible builds, Arduino CLI and USB serial feedback can be used to compile/upload a sketch and check actual sensor behavior.
6. **Bring others along.** A local FastAPI dashboard makes the session easier to follow on a laptop or classroom display, with session information and visual guidance.

> Example: “I want an alarm when someone gets close to my desk.” PinPoint helps select a distance sensor, decide how to trigger the alert, wire the circuit, and test it.

## One platform, different ways to build

PinPoint's intended audience is **everyone who wants to build electronics**. Accessibility is part of the design rather than a separate product.

| Interaction | Experience |
| --- | --- |
| **Voice-guided** | Spoken explanations and tactile-style positional directions, especially useful for blind and low-vision makers. |
| **Visual-guided** | Large captions, highlighted targets, and step prompts for users who prefer or need visual instructions, including Deaf and hard-of-hearing makers. |
| **Text-guided** | Chert iMessage integration provides a path for typed questions and build recaps, useful for nonspeaking users and anyone who prefers typing. |
| **Hands-free / continuous guidance** | An experimental overhead-camera mode tracks hands and offers in-the-moment positioning cues; this remains an area of active development and testing. |

Accessibility features are **not a claim of independent validation by blind, Deaf, or nonspeaking users**. We welcome feedback and testing with the communities these features are intended to support.

## How it works — technical architecture

```text
                  iPhone / FaceTime + iMessage
                             |
                         Chert API
                             |
                   LiveKit audio / video
                             |
                 PinPoint Python backend
                  (FastAPI + sessions)
                      /      |      \
                     /       |       \
           Speech / captions |     Vision pipeline
          (faster-whisper,   |     (OpenCV, markers,
           TTS, overlays)    |      MediaPipe, Claude)
                             |
                   Planning / validation
                    (Claude + circuit
                   model + YAML library)
                             |
               Build guidance + Arduino CLI
                             |
                 Arduino over USB / serial
                             |
                Sensor readings / test results

                 + Local dashboard (:8000)
```

### Main components

| Component | Responsibility |
| --- | --- |
| **Chert + LiveKit** | FaceTime call connection and bidirectional media transport; iMessage for typed interactions and recaps. |
| **Claude / Anthropic API** | Understands build requests, proposes circuits and sketches, and reasons about camera images. |
| **Circuit validation** | Checks generated wiring plans against constraints such as short circuits, unsupported connections, and resistor requirements before assembly. |
| **OpenCV + printed markers** | Selects usable frames, estimates board orientation, and maps observed positions to physical reference points. |
| **MediaPipe** | Supports hand and gesture recognition for experimental guidance and hands-free input. |
| **faster-whisper + speech output** | Speech-to-text, spoken instructions, and captions/overlays for different interaction modes. |
| **FastAPI + session engine** | Tracks calls, steps, user commands, verification requests, and local monitoring. |
| **Arduino CLI + PySerial** | Compiles/uploads compatible Arduino sketches and reads serial data to verify supported builds. |

### Why physical verification matters

An LLM can produce a convincing-looking circuit that doesn't work. PinPoint therefore separates **generating a plan** from **checking it**. The repository contains a breadboard representation and plan checker, a supported parts catalog, and a fallback library of curated builds. Camera-based checks help connect those abstract plans to actual physical connections. For compatible projects, serial sensor readings provide another feedback channel after wiring.

This is a prototype, not a substitute for electrical inspection or expert supervision. PinPoint focuses on **low-voltage educational electronics**, not mains wiring or unsafe appliance repair.

## Hardware

Our EmoryHacks setup uses:

- An **ELEGOO Mega Project / Mega 2560-compatible starter kit** (including its breadboard components).
- A separate **Arduino UNO board** for compatible builds.
- Grove modules, including a **distance sensor** and **RGB-backlit LCD**.
- Breadboards, jumper cables, USB A-to-B cables, and other standard kit components.
- **Four printed reference markers** used for visual positioning and calibration.

**Important board note:** The codebase's default ELEGOO catalog was initially written around an **UNO R3-style starter kit**. The physical kit pictured at the hackathon includes a **Mega 2560-compatible board**. Select the actual board and pin mapping when running Arduino CLI; not all boards, sensor shields, or sample sketches are interchangeable.

## Repository structure

```text
PinPoint/
├── arduino/proximity_alert/      # Sample Arduino sketch
├── backend/
│   ├── main.py                  # FastAPI app and webhook entry points
│   ├── chert_adapter.py         # Chert FaceTime/iMessage integration
│   ├── chert_agent.py           # LiveKit/Chert agent worker
│   ├── livekit_transport.py     # Real-time audio/video tracks
│   ├── session.py               # Per-call state and frame processing
│   ├── stt.py                   # Speech recognition
│   ├── build/                   # Parts catalog, planner, validation, library
│   ├── build_session.py         # Kit-aware guided assembly flow
│   ├── guidance/                # Step engine, speech, visual overlays
│   ├── vision/                  # Capture, markers, hands, verification
│   ├── hardware/                # Arduino USB / serial monitor
│   └── projects/                # Curated project definitions
├── dashboard/                   # Local session-monitoring interface
├── markers/                     # Printable calibration/reference markers
├── tests/                       # Automated tests
├── tools/                       # Setup, simulation, calibration, mock bridge
├── .env.example                 # Credential and mode configuration template
└── requirements.txt             # Python dependencies
```

## Getting started

### Prerequisites

- Python and `pip` (use a version supported by this project's pinned dependencies).
- A Chert account with an assigned FaceTime line and the credentials required by the selected Chert integration path.
- LiveKit project credentials and an Anthropic API key for the AI-powered flow.
- An Arduino board and USB connection for physical tests; `arduino-cli` installed for sketch compilation/upload.
- Printed reference markers for calibrated board positioning.

### 1. Clone and install

```bash
git clone https://github.com/pulkithanda/PinPoint.git
cd PinPoint
python -m venv .venv
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional: pre-download speech models with `python -m tools.download_models`. The first run may otherwise download model files.

### 2. Configure credentials and modes

Copy `.env.example` to `.env` and fill in only the keys relevant to your chosen integration. **Never commit `.env`, API keys, phone numbers, or secrets.**

Important environment variable groups (the authoritative list is in [`.env.example`](.env.example)):

```dotenv
ANTHROPIC_API_KEY=
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
CHERT_FACETIME_API_KEY=
CHERT_PROJECT_ID=
CHERT_MESSAGING_SECRET=
AGENT_NAME=pinpoint
PINPOINT_KIT=elegoo
PINPOINT_GUIDANCE=snapshot
PINPOINT_MODE=
PINPOINT_SERIAL_PORT=auto
```

`PINPOINT_KIT=elegoo` selects kit-aware planning; `grove` selects the Grove Base Shield workflow. `PINPOINT_GUIDANCE=snapshot` prompts the caller to hold the phone up when needed; `live` is the experimental overhead-camera mode. Set `PINPOINT_MODE=voice` or `visual`, or leave it empty to choose at call start.

### 3. Test without a FaceTime call

Try a terminal build first:

```bash
python -m tools.sim_terminal
```

For the curated-project fallback without Claude:

```bash
python -m tools.sim_terminal --offline
```

Run automated checks:

```bash
python -m pytest -q
```

A local webcam workflow is also provided:

```bash
python -m backend.main --source webcam
```

### 4. Set up Arduino and markers

```bash
python -m tools.setup_arduino
python -m tools.calibrate_ports
```

Print the four marker squares at **100% / actual size**, keep the white border visible, and follow the marker IDs and calibration instructions in the existing project documentation. Connect the board by USB and ensure the sketch's target board, port, and pins match your actual hardware.

The curated proximity-alarm sketch lives in `arduino/proximity_alert/`.

### 5. Connect Chert

The repository documents two integration routes. The **development worker** path uses two terminals:

```bash
# Terminal 1
python -m backend.chert_agent start
```

```bash
# Terminal 2 (with Chert CLI access)
npx --yes @trychert/facetime@0.1.0-beta.20 dev --agent pinpoint
```

Once the agent is in a Chert waiting room, FaceTime the assigned line. Open `http://localhost:8000` for the local dashboard.

Alternatively, the webhook-based path runs `python -m backend.main --source chert` behind a public HTTPS tunnel and connects `/chert/webhook` and `/chert/imessage/webhook` in the Chert integration settings. Consult the existing Chert setup documentation and [Chert's official docs](https://docs.trychert.com/) for credentials and current CLI requirements; do not use real secrets in shared commands or screenshots.

### Useful guidance commands

During a build, the session engine supports instructions such as `ready`, `done`, `repeat`, `help`, `back`, and `skip`. It also accepts natural-language questions about the current step and requests to verify components or connections.

## What we've built and what's next

The repository includes a kit-aware planner, circuit checker, curated builds, Chert/LiveKit integration code, speech and camera processing, step tracking, Arduino tooling, testing utilities, and a local dashboard. **Having code for a feature does not mean every integration has been validated end to end**; behavior depends on hardware, credentials, camera calibration, and the deployed Chert line.

Next we want to expand the supported components and verified projects, refine continuous hands-free guidance and visual overlays, improve resume-after-disconnect behavior, strengthen iMessage-based interaction, and test accessibility workflows with actual makers. Longer-term possibilities include electronics diagnostics and classroom-scale collaborative builds.

## Built with

**Python · FastAPI · Anthropic Claude · Chert · LiveKit · OpenCV · MediaPipe · faster-whisper · Arduino · Arduino CLI · PySerial · PyYAML · ELEGOO · Grove · Coder · VS Code · Claude Code · GitHub**

## Team and links

Created at **EmoryHacks 2026**.

- **Devpost:** https://devpost.com/software/pinpoint-9pyk0w
- **Demo video:** https://www.youtube.com/watch?v=Aq5WXmQQooo
- **Source code:** https://github.com/pulkithanda/PinPoint

> **Your idea. Your parts. Your AI hardware co-builder on FaceTime.**
