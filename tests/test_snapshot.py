"""
Photo-on-demand (snapshot) guidance: auto-capture, engine flow, questions, speech segmentation,
and the session's capture -> Claude -> engine round trip (with a stub verifier).
"""
import time

import cv2
import numpy as np
import pytest

from backend.config import Settings
from backend.guidance.engine import (GuidanceEngine, Phase, Project, RequestAnswer, RequestIdentify,
                                     RequestPlugCheck, Say, is_question, parse_command)
from backend.stt import UtteranceSegmenter
from backend.vision.capture import AutoCapture
from backend.vision.shield import ShieldLayout, ShieldTracker
from backend.vision.verify import ModuleID, PlugCheck
from tests.test_shield import render_board


@pytest.fixture(scope="module")
def layout():
    return ShieldLayout.load(Settings().layout_path)


def says(actions):
    return [a.text for a in actions if isinstance(a, Say)]


# ---------------------------------------------------------------- auto-capture
def textured(seed=0, shift=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((360, 480)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(np.roll(img, shift, axis=1), cv2.COLOR_GRAY2BGR)


def test_capture_waits_for_steady_sharp_frames():
    cap = AutoCapture("identify", started_t=0.0)
    for i in range(4):  # phone moving
        upd = cap.feed(textured(shift=i * 25), i * 0.07)
        assert upd.status == "waiting"
    assert upd.hint == "Hold the phone still."
    frame = textured()
    statuses = [cap.feed(frame, 1 + i * 0.07).status for i in range(8)]
    assert "captured" in statuses and statuses.index("captured") <= 6


def test_capture_rejects_blur_and_times_out():
    blurry = cv2.GaussianBlur(textured(), (31, 31), 0)
    cap = AutoCapture("question", started_t=0.0, timeout_s=2.0)
    for i in range(10):
        assert cap.feed(blurry, i * 0.07).status == "waiting"
    assert cap.feed(blurry, 2.5).status == "timeout"


def test_capture_aims_phone_toward_target(layout):
    board = render_board(layout)
    canvas = np.full((board.shape[0] + 200, 3200, 3), 255, np.uint8)
    canvas[100:100 + board.shape[0], 0:board.shape[1]] = board  # board hard against the left edge
    shield = ShieldTracker(layout, recheck_s=0.0)
    assert shield.update(canvas, 0.0, force=True)
    cap = AutoCapture("plug", ["D7"], started_t=0.0)
    cap.feed(canvas, 0.0, shield)
    upd = cap.feed(canvas, 0.07, shield)
    assert upd.status == "waiting" and upd.hint == "Move the phone a little to your left."

    centred = np.full((board.shape[0] + 200, board.shape[1] + 200, 3), 255, np.uint8)
    centred[100:100 + board.shape[0], 100:100 + board.shape[1]] = board
    shield.update(centred, 1.0, force=True)
    cap = AutoCapture("plug", ["D7"], started_t=1.0)
    out = [cap.feed(centred, 1.0 + i * 0.07, shield) for i in range(8)]
    done = [u for u in out if u.status == "captured"]
    assert done and done[0].labeled


# ---------------------------------------------------------------- engine (snapshot mode)
def test_questions_vs_commands():
    assert parse_command("what is the buzzer for?") is None and is_question("what is the buzzer for?")
    assert parse_command("is this plugged in the right way round") is None
    assert is_question("is this plugged in the right way round")
    assert parse_command("what") == "repeat"
    assert parse_command("ready") == "next"
    assert parse_command("what am I holding") == "holding"
    assert not is_question("next")


@pytest.fixture
def snap(layout):
    s = Settings()
    eng = GuidanceEngine(Project.load(s.projects_dir / "proximity_alert.yaml"), layout,
                         mode="voice", steering=False)
    return eng


def test_snapshot_flow(snap, layout):
    eng = snap
    assert "Say 'ready'" in says(eng.start())[-1]
    acts = eng.on_command("next")                       # "ready"
    assert any(isinstance(a, RequestIdentify) for a in acts)
    text = says(eng.on_identify_result(ModuleID("Ultrasonic Distance Sensor", True, "")))[0]
    assert "photo to check" in text and "second port from the left" in text
    from backend.guidance.engine import Observation
    assert eng.on_observation(Observation(1.0, True, (layout.ports["D2"].x, layout.ports["D2"].y))) == []
    acts = eng.on_command("done")
    assert any(isinstance(a, RequestPlugCheck) for a in acts)
    eng.on_plug_result(PlugCheck(True, "D7", ""))
    assert eng.step.id == "buzzer" and eng.phase == Phase.FIND_MODULE


def test_snapshot_unreadable_label_trusts_user(snap):
    eng = snap
    eng.start()
    eng.on_command("next")
    assert "couldn't read the label" in says(eng.on_identify_result(ModuleID("unknown", False, "")))[0]
    eng.on_command("next")
    assert "trust you" in says(eng.on_identify_result(ModuleID("unknown", False, "")))[0]
    eng.on_command("next")
    assert eng.phase == Phase.GUIDE


def test_question_round_trip(snap):
    eng = snap
    eng.start()
    acts = eng.on_question("Is the buzzer the round black one?")
    req = [a for a in acts if isinstance(a, RequestAnswer)][0]
    assert "Ultrasonic Distance Sensor" in req.context
    assert says(eng.on_answer("Yes, the buzzer is the small board with a round cylinder.")) == \
        ["Yes, the buzzer is the small board with a round cylinder."]
    assert "couldn't work that out" in says(eng.on_answer(None))[0]


# ---------------------------------------------------------------- speech-to-text segmentation
def test_segmenter_cuts_on_silence():
    seg = UtteranceSegmenter()
    rate = 16000
    t = np.arange(int(0.05 * rate)) / rate
    speech = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    silence = np.zeros_like(speech)
    out = [seg.feed(speech, rate) for _ in range(12)]          # 0.6 s of "speech"
    out += [seg.feed(silence, rate) for _ in range(14)]        # 0.7 s silence -> utterance ends
    utts = [u for u in out if u is not None]
    assert len(utts) == 1 and 1.1 < len(utts[0]) / rate < 1.4
    assert all(seg.feed(silence, rate) is None for _ in range(30))  # silence alone is never an utterance


def test_segmenter_resamples_48k():
    seg = UtteranceSegmenter()
    t = np.arange(int(0.05 * 48000)) / 48000
    speech = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    out = [seg.feed(speech, 48000) for _ in range(10)] + [seg.feed(np.zeros_like(speech), 48000) for _ in range(14)]
    utt = next(u for u in out if u is not None)
    assert abs(len(utt) / 16000 - 1.2) < 0.1


# ---------------------------------------------------------------- session round trip
class _StubHands:
    def process(self, frame, ts):
        return []


class _StubVerifier:
    def __init__(self):
        self.calls = []

    def identify_module(self, frame, candidates):
        self.calls.append("identify")
        return ModuleID("Ultrasonic Distance Sensor", True, "")

    def check_plug(self, frame, module, ports, labeled=True):
        self.calls.append(("plug", labeled))
        return PlugCheck(True, "D7", "")

    def answer_question(self, frame, question, context, mode):
        self.calls.append("answer")
        return "That's the distance sensor."


class _QuietTTS:
    busy = False

    def __init__(self, *a, **k):
        pass

    @classmethod
    def shared(cls, *a, **k):
        return cls()

    def say(self, text):
        pass

    def render(self, text):
        return None

    def prerender(self, phrases):
        pass


def test_session_snapshot_round_trip(monkeypatch, layout):
    import backend.session as session_mod
    from backend.hardware.serial_monitor import FakeSerialMonitor
    from backend.transport import CallTransport

    monkeypatch.setattr(session_mod, "TTSWorker", _QuietTTS)
    board = render_board(layout)
    frame = cv2.copyMakeBorder(board, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    class FakeTransport(CallTransport):
        local_audio = False

        def __init__(self):
            self.captions = []

        def read_frame(self):
            time.sleep(0.02)
            return (time.monotonic(), frame)

        def send_caption(self, text):
            self.captions.append(text)

    s = Settings()
    s.serial_port, s.mode, s.guidance = "none", "voice", "snapshot"
    tr, ver = FakeTransport(), _StubVerifier()
    sess = session_mod.Session(tr, s, serial=FakeSerialMonitor(), hand_tracker=_StubHands(), verifier=ver).start()
    try:
        def wait_for(cond, timeout=5.0):
            end = time.monotonic() + timeout
            while time.monotonic() < end and not cond():
                time.sleep(0.02)
            return cond()

        sess.command("ready")
        assert wait_for(lambda: sess.engine.phase == Phase.GUIDE)
        sess.command("done")
        assert wait_for(lambda: sess.engine.step and sess.engine.step.id == "buzzer")
        assert ("plug", True) in ver.calls                      # markers visible -> labelled photo
        assert any("Pick up the phone" in c for c in tr.captions)
        assert any("put the phone down" in c for c in tr.captions)
        assert sess.command("what is this cable for?") == "question"
        assert wait_for(lambda: "answer" in ver.calls and "That's the distance sensor." in tr.captions)
    finally:
        sess.stop()
