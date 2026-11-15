"""Elegoo build flow: kit, breadboard nets, plan checker, library, planner repair loop, engine."""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from backend.build import library
from backend.build.board import BuildPlan, Orientation, endpoint_text, load_kit, pin_text, validate
from backend.build.engine import (BuildEngine, Finished, Phase, RequestPartID, RequestPlacementCheck, RequestPlan,
                                  RequestReview, RequestSetupCheck, RequestUpload)
from backend.build.planner import OfflinePlanner, Planner
from backend.guidance.engine import RequestAnswer, Say
from backend.hardware.uploader import UploadResult
from backend.vision.verify import BoardReview, BoardSetup, ModuleID, PlacementCheck

KIT = load_kit()


def plan(pid: str) -> BuildPlan:
    return copy.deepcopy(library.get(pid))


# ------------------------------------------------------------------ library + checker
@pytest.mark.parametrize("entry", library.load_library(), ids=lambda e: e.plan.id)
def test_every_library_project_passes_the_checker(entry):
    assert validate(entry.plan, KIT) == []


def test_library_has_basics_and_ten_advanced():
    plans = [e.plan for e in library.load_library()]
    assert sum(p.difficulty == "basic" for p in plans) >= 5
    assert sum(p.difficulty == "advanced" for p in plans) >= 10


@pytest.mark.parametrize("request_text,expected", [
    ("something that beeps when my cat gets close to the snacks", "proximity_alarm"),
    ("a light that turns on in the dark", "night_light"),
    ("a robot arm I can steer with a joystick", "joystick_servo"),
    ("play music by waving my hand", "light_theremin"),
    ("build a quantum computer", None),
])
def test_library_matching(request_text, expected):
    got = library.match(request_text)
    assert (got.id if got else None) == expected


def _first(errs, needle):
    return any(needle in e for e in errs)


def test_checker_catches_short_circuit():
    p = plan("blink_led")
    p.steps.insert(0, BuildPlan.from_dict({"steps": [{"kind": "wire", "connections": [
        {"lead": "end1", "to": "UNO:5V"}, {"lead": "end2", "to": "RAIL:N-"}]}]}).steps[0])
    assert _first(validate(p, KIT), "SHORT CIRCUIT")


def test_checker_catches_missing_led_resistor():
    p = plan("blink_led")
    p.parts = [x for x in p.parts if x.ref != "R1"]
    p.steps = [s for s in p.steps if s.ref != "R1"]
    # cathode straight to the minus rail instead
    for s in p.steps:
        if s.ref == "LED1":
            s.connections[1].to = "RAIL:N-"
    assert _first(validate(p, KIT), "resistor")


def test_checker_catches_reused_hole_and_far_rows():
    p = plan("blink_led")
    led = next(s for s in p.steps if s.ref == "LED1")
    led.connections[0].to = "D11"          # the resistor's hole
    errs = validate(p, KIT)
    assert _first(errs, "already used")
    p = plan("blink_led")
    led = next(s for s in p.steps if s.ref == "LED1")
    led.connections[0].to = "E45"
    assert _first(validate(p, KIT), "too far to count")


def test_checker_catches_floating_and_shorted_leads():
    p = plan("blink_led")
    p.steps = [s for s in p.steps if not (s.kind == "wire" and any(c.to == "UNO:D13" for c in s.connections))]
    assert _first(validate(p, KIT), "not connected to anything")
    p = plan("blink_led")
    led = next(s for s in p.steps if s.ref == "LED1")
    led.connections[1].to = "A10"           # both legs in row 10 A-E
    assert _first(validate(p, KIT), "shorts it out")


def test_checker_enforces_part_shapes():
    p = plan("button_led")
    sw = next(s for s in p.steps if s.ref == "SW1")
    sw.connections[1].to = "F16"            # rows must be 2 apart
    assert _first(validate(p, KIT), "straddle")
    p = plan("dimmer")
    pot = next(s for s in p.steps if s.ref == "POT1")
    pot.connections[1].to, pot.connections[2].to = "E22", "E21"   # legs out of order
    assert _first(validate(p, KIT), "consecutive rows")


def test_checker_rejects_unknown_parts_and_stock():
    p = plan("blink_led")
    p.parts[0].part = "flux_capacitor"
    assert _first(validate(p, KIT), "not in the kit")


# ------------------------------------------------------------------ touch directions
def test_touch_directions():
    o = Orientation()
    assert endpoint_text("E10", KIT, o) == "row 10, counting 10 from the left end, the hole right beside the groove, on your side"
    assert endpoint_text("A3", KIT, o).endswith("the outermost hole on your side")
    assert "second hole from the groove, on the far side" in endpoint_text("G3", KIT, o)
    assert "minus rail" in endpoint_text("RAIL:N-", KIT, o) and "outer strip" in endpoint_text("RAIL:N-", KIT, o)
    assert "inner strip" in endpoint_text("RAIL:N+", KIT, o)
    assert pin_text("UNO:D13", KIT) == ("pin D13 on the Arduino: the socket strip along the far edge, left-hand "
                                        "block, fifth socket from the left")
    assert "near edge, right-hand block, first socket" in pin_text("UNO:A0", KIT)
    o.row1 = "right"
    assert "from the right end" in endpoint_text("E10", KIT, o)


# ------------------------------------------------------------------ planner repair loop (fake Claude)
class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        out = self.outputs.pop(0)
        return SimpleNamespace(stop_reason="end_turn", usage=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(out))])


def _planner_output(p: BuildPlan, decision="build"):
    return {"decision": decision, "library_id": "", "reason": "I'll build it.", "plan": p.to_dict()}


def test_planner_repairs_a_bad_plan():
    good = plan("night_light")
    bad = copy.deepcopy(good)
    bad.parts = [x for x in bad.parts if x.ref != "R2"]
    bad.steps = [s for s in bad.steps if s.ref != "R2"]
    client = FakeClient([_planner_output(bad), _planner_output(good)])
    res = Planner(client=client).plan("a light for when it's dark")
    assert res.source == "planner" and res.attempts == 2 and res.plan.title == good.title
    repair_msg = client.requests[1]["messages"][-1]["content"]
    assert "plan checker found these problems" in repair_msg


def test_planner_falls_back_to_library_after_repeated_failures():
    bad = plan("night_light")
    bad.steps = bad.steps[:2]
    client = FakeClient([_planner_output(bad)] * 3)
    res = Planner(client=client).plan("a light that comes on in the dark")
    assert res.source == "library" and res.plan.id == "night_light" and res.errors


def test_planner_library_decision_and_decline():
    client = FakeClient([{"decision": "library", "library_id": "traffic_light", "reason": "A classic!",
                          "plan": plan("blink_led").to_dict()}])
    res = Planner(client=client).plan("a traffic light")
    assert res.plan.id == "traffic_light" and res.source == "library"
    client = FakeClient([{"decision": "decline", "library_id": "", "reason": "That needs mains power.",
                          "plan": plan("blink_led").to_dict()}])
    res = Planner(client=client).plan("control my kettle")
    assert res.plan is None and "mains" in res.message


# ------------------------------------------------------------------ engine end to end
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _says(actions):
    return " ".join(a.text for a in actions if isinstance(a, Say))


def test_full_build_with_a_caught_mistake():
    clock = Clock()
    eng = BuildEngine(KIT, mode="voice", clock=clock)
    out = eng.start()
    assert "What would you like to build" in _says(out)
    assert eng.on_utterance("ok") and eng.phase == Phase.INTAKE       # a stray "ok" isn't a request

    out = eng.on_utterance("I want a light that turns on in the dark")
    req = next(a for a in out if isinstance(a, RequestPlan))
    assert eng.phase == Phase.PLANNING
    clock.t += 16
    assert "Still working on it" in _says(eng.on_tick())
    out = eng.on_plan_result(OfflinePlanner().plan(req.request))
    assert eng.phase == Phase.CONFIRM and "Want to build it" in _says(out)
    assert "light sensor" in _says(out)

    out = eng.on_utterance("yes let's do it")
    assert eng.phase == Phase.SETUP
    assert any(isinstance(a, RequestSetupCheck) for a in out)      # looks first, no instructions yet
    out = eng.on_setup_result(BoardSetup("AE", "left", "-", True, "ok"))
    assert eng.phase in (Phase.PLACE, Phase.FIND) and "Step 1 of" in _says(out)

    mistake_done = False
    while eng.phase in (Phase.FIND, Phase.PLACE, Phase.CHECKING):
        out = eng.on_utterance("done")
        for a in out:
            if isinstance(a, RequestPartID):
                out = eng.on_identify_result(ModuleID(a.expected, True, "ok"))
                break
            if isinstance(a, RequestPlacementCheck):
                if not mistake_done:
                    mistake_done = True
                    fix = eng.on_placement_result(PlacementCheck(False, "The long leg is in row 11; swap the legs.", ""))
                    assert "swap the legs" in _says(fix) and eng.phase == Phase.PLACE
                else:
                    eng.on_placement_result(PlacementCheck(True, "", "ok"))
                break
            if isinstance(a, RequestReview):
                break
    assert eng.phase == Phase.REVIEW
    out = eng.on_review_result(BoardReview(False, ["The wire from A0 is in row 17, not row 16."], ""))
    assert "row 17" in _says(out)
    out = eng.on_utterance("done")
    assert any(isinstance(a, RequestReview) for a in out)
    out = eng.on_review_result(BoardReview(True, [], "ok"))
    assert eng.phase == Phase.UPLOAD
    out = eng.on_utterance("ready")
    assert any(isinstance(a, RequestUpload) for a in out)
    out = eng.on_upload_result(UploadResult(True, "upload", "", "ok"))
    assert eng.phase == Phase.TEST and "Cover the light sensor" in _says(out)
    assert eng.on_serial(True, [800, 805, 799]) == []                 # noise isn't a pass
    out = eng.on_serial(True, [800, 300, 120, 640])
    fin = next(a for a in out if isinstance(a, Finished))
    assert eng.phase == Phase.DONE
    assert "Automatic night light" in fin.recap and "Caught along the way" in fin.recap
    assert "a light that turns on in the dark" in fin.recap


def test_change_request_and_questions():
    eng = BuildEngine(KIT, clock=Clock())
    eng.start()
    eng.on_utterance("something that beeps when my cat is near")
    eng.on_plan_result(OfflinePlanner().plan("beeps when something is close"))
    out = eng.on_utterance("can you make it quieter and use a light instead")
    assert any(isinstance(a, RequestPlan) and a.previous is not None for a in out)
    eng.on_plan_result(OfflinePlanner().plan("night light"))
    eng.on_utterance("yes")
    eng.on_utterance("ready")
    eng.on_setup_result(BoardSetup(None, None, None, None, "no photo"))
    assert "Pinpoint" in _says(eng.on_utterance("Who is the lady here?"))
    out = eng.on_utterance("is this the right way round?")
    assert any(isinstance(a, RequestAnswer) for a in out)
    assert eng.on_answer("Yes, the long leg is on the left.")[0].text.startswith("Yes")


def test_visual_mode_by_thumbs_up_and_no_verifier_flow():
    eng = BuildEngine(KIT, verifier_available=False, uploader_available=False, clock=Clock())
    eng.start()
    out = eng.on_gesture("Thumb_Up")
    assert eng.mode == "visual" and "Captions on" in _says(out)
    eng.on_utterance("a blinking light")
    eng.on_plan_result(OfflinePlanner().plan("blink an led"))
    eng.on_utterance("yes")
    eng.on_utterance("ready")                     # no setup photo without a verifier
    steps = 0
    while eng.phase in (Phase.FIND, Phase.PLACE):
        eng.on_utterance("done")
        steps += 1
        assert steps < 50
    assert eng.phase == Phase.TEST                # no uploader -> manual test
    out = eng.on_utterance("skip")
    assert any(isinstance(a, Finished) for a in out)


def test_build_session_smoke(tmp_path, monkeypatch):
    """The session wiring: transcript in -> plan on a worker -> confirm question spoken."""
    from backend import build_session as bs
    from backend.build_session import BuildSession

    monkeypatch.setattr(bs, "RESUME_FILE", tmp_path / "resume.json")   # never touch the real one
    from backend.config import Settings
    from backend.hardware.serial_monitor import FakeSerialMonitor

    class Transport:
        local_audio = False
        wants_video = False

        def __init__(self):
            self.inbox = ["I want to build a traffic light"]
            self.captions = []

        def start(self): pass
        def read_frame(self): return None
        def poll_transcripts(self):
            out, self.inbox = self.inbox, []
            return out
        def send_speech(self, text, wav): pass
        def send_caption(self, text): self.captions.append(text)
        def send_beep(self, wav): pass
        def send_video_frame(self, frame): pass
        def send_message(self, text, images=None): pass
        def close(self): pass

    class NoUploader:
        available = False

    t = Transport()
    s = Settings()
    sess = BuildSession(t, s, planner=OfflinePlanner(), verifier=None, uploader=NoUploader(),
                        serial=FakeSerialMonitor(), hand_tracker=None)
    sess.verifier = None
    sess.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and sess.engine.phase != Phase.CONFIRM:
            time.sleep(0.05)
        assert sess.engine.phase == Phase.CONFIRM
        assert any("Traffic light" in c or "traffic" in c.lower() for c in t.captions)
    finally:
        sess.stop()


# ------------------------------------------------------------------ listening (from the first real call)
def _engine_at_confirm():
    eng = BuildEngine(KIT, clock=Clock())
    eng.start()
    eng.on_utterance("I want to build a light that blinks faster the more I turn a knob")
    eng.on_plan_result(OfflinePlanner().plan("a blinking light"))
    assert eng.phase == Phase.CONFIRM
    return eng


@pytest.mark.parametrize("said", ["Yes.", "Yeah, let's build that one", "Okay.", "yes please go ahead",
                                  "sounds good, let's do it", "Sure, build it"])
def test_yes_starts_the_build(said):
    eng = _engine_at_confirm()
    out = eng.on_utterance(said)
    assert eng.phase == Phase.SETUP and not any(isinstance(a, RequestPlan) for a in out)


@pytest.mark.parametrize("said", ["No.", "I will not.", "no thanks", "nope, something else", "don't build that"])
def test_no_goes_back_to_asking(said):
    eng = _engine_at_confirm()
    out = eng.on_utterance(said)
    assert eng.phase == Phase.INTAKE and not any(isinstance(a, RequestPlan) for a in out)


@pytest.mark.parametrize("said", ["Now what a villa.", "It's like, keep the meat.", "hmm okay so"])
def test_misheard_fragments_dont_replan(said):
    eng = _engine_at_confirm()
    out = eng.on_utterance(said)
    assert eng.phase == Phase.CONFIRM and not any(isinstance(a, RequestPlan) for a in out)
    assert "I heard" in _says(out)


def test_yes_but_change_replans():
    eng = _engine_at_confirm()
    out = eng.on_utterance("yes but make it beep too")
    assert any(isinstance(a, RequestPlan) and a.previous is not None for a in out)


def test_intake_ignores_no_and_cut_off_fragments():
    eng = BuildEngine(KIT, clock=Clock())
    eng.start()
    for said in ("No, no, no, don't do that, don't do that.", "I wanna", "I want to build a...", "stop", "wait"):
        out = eng.on_utterance(said)
        assert not any(isinstance(a, RequestPlan) for a in out), said
    assert eng.phase == Phase.INTAKE


def test_cancel_and_replace_while_planning():
    eng = BuildEngine(KIT, clock=Clock())
    eng.start()
    first = next(a for a in eng.on_utterance("a light that turns on in the dark") if isinstance(a, RequestPlan))
    second = next(a for a in eng.on_utterance("actually a traffic light with three colours")
                  if isinstance(a, RequestPlan))
    assert second.token > first.token
    stale = OfflinePlanner().plan("night light")
    stale.token = first.token
    assert eng.on_plan_result(stale) == [] and eng.phase == Phase.PLANNING      # replaced: ignored
    fresh = OfflinePlanner().plan("traffic light")
    fresh.token = second.token
    eng.on_plan_result(fresh)
    assert eng.phase == Phase.CONFIRM and eng.plan.id == "traffic_light"
    eng.on_utterance("no wait")
    third = next(a for a in eng.on_utterance("a game I can play with a button") if isinstance(a, RequestPlan))
    assert eng.on_utterance("no, stop")[0].text.startswith("Okay, I've stopped")
    late = OfflinePlanner().plan("reaction game")
    late.token = third.token
    assert eng.on_plan_result(late) == [] and eng.phase == Phase.INTAKE           # cancelled: ignored


def test_unrecognised_comment_mid_build_is_acknowledged():
    eng = _engine_at_confirm()
    eng.on_utterance("yes")
    eng.on_utterance("skip")                     # skip the setup photo
    out = eng.on_utterance("this wire feels really short")
    assert "I heard" in _says(out)


def test_setup_photo_feedback():
    eng = _engine_at_confirm()
    out = eng.on_utterance("yes")
    assert any(isinstance(a, RequestSetupCheck) for a in out)     # photo first, straight away
    out = eng.on_setup_result(BoardSetup(None, None, None, None, "selfie camera", board_visible=False))
    assert "back camera" in _says(out) and eng.phase == Phase.SETUP
    out = eng.on_utterance("I did that already")                  # "already done": look again now
    assert any(isinstance(a, RequestSetupCheck) for a in out)
    out = eng.on_setup_result(BoardSetup(None, None, "-", False, "rotated", board_visible=True,
                                         fix="Turn the Arduino round so its USB socket points to your left."))
    assert "USB socket points to your left" in _says(out) and eng.phase == Phase.SETUP
    eng.on_utterance("done")
    out = eng.on_setup_result(BoardSetup(None, None, None, None, "blurry", board_visible=None))
    assert eng.phase in (Phase.FIND, Phase.PLACE)            # third try: go with the standard layout


def test_setup_already_right_starts_in_one_photo():
    eng = _engine_at_confirm()
    eng.on_utterance("yes")
    out = eng.on_setup_result(BoardSetup("AE", "left", "-", True, "layout ok", board_visible=True, long_side_toward=True))
    assert "already set up just right" in _says(out) and "Step 1 of" in _says(out)
    assert eng.phase in (Phase.FIND, Phase.PLACE)


@pytest.mark.parametrize("said", ["I already did it", "it's already in", "all set", "done", "yeah"])
def test_already_done_counts_as_ready(said):
    eng = _engine_at_confirm()
    eng.on_utterance("yes")
    eng.on_utterance("skip")
    before = eng.step_idx
    eng.on_utterance(said)
    assert eng.step_idx == before + 1 or eng.phase != Phase.PLACE


@pytest.mark.parametrize("request_text,expected", [
    ("a blinking light", "blink_led"), ("Yeah, I wanna build a light that blinks.", "blink_led"),
    ("I would like to make a night light", "night_light"), ("control my kettle", None),
    ("a light that blinks faster the more I turn a knob", None),
])
def test_confident_library_match_skips_the_llm(request_text, expected):
    got = library.confident_match(request_text)
    assert (got.id if got else None) == expected
    if expected:
        client = FakeClient([])                  # any LLM call would fail: there are no outputs
        res = Planner(client=client).plan(request_text)
        assert res.plan.id == expected and client.requests == []


# ------------------------------------------------------------------ barge-in
def test_segmenter_flags_speech_start_once_per_utterance():
    import numpy as np

    from backend.stt import UtteranceSegmenter

    seg = UtteranceSegmenter()
    loud = (np.sin(np.arange(1600) / 3.0) * 8000).astype(np.int16)    # 0.1 s of speech-level audio
    quiet = np.zeros(1600, np.int16)
    starts = 0
    for chunk in [loud] * 6 + [quiet] * 8 + [loud] * 6:
        seg.feed(chunk, 16000)
        if seg.speech_started:
            seg.speech_started = False
            starts += 1
    assert starts == 2
    seg2 = UtteranceSegmenter()
    seg2.feed(loud, 16000)                       # a 0.1 s click is not the caller talking
    assert not seg2.speech_started


def test_scheduler_clear():
    from backend.guidance.speech import SpeechScheduler

    sch = SpeechScheduler()
    sch.submit(Say("one"), 0.0)
    sch.submit(Say("two"), 0.0)
    assert sch.pending
    sch.clear()
    assert not sch.pending and sch.pop_ready(10.0) is None


def test_where_and_how_questions_are_answered_from_the_plan():
    eng = _engine_at_confirm()
    eng.on_utterance("yes")
    eng.on_utterance("skip")                      # standard layout, step 1: 5V -> plus rail
    out = eng.on_utterance("Okay so I inserted one end onto the positive rail. Where does the other end go?")
    assert not any(isinstance(a, RequestAnswer) for a in out)
    assert "pin GND" in _says(out) and "minus rail" in _says(out)      # blink_led starts with GND -> minus rail
    assert "near edge" in _says(out)                                   # the GND socket beside the breadboard
    out = eng.on_utterance("How do I put a jumper wire onto the breadboard?")
    assert not any(isinstance(a, RequestAnswer) for a in out) and "metal pin" in _says(out)
    out = eng.on_utterance("Are you able to see it now?")
    assert any(isinstance(a, RequestAnswer) for a in out)        # needs eyes: photo
    assert "Pinpoint" in _says(eng.on_utterance("Who is the lady here?"))
    out = eng.on_utterance("is this the right way round?")
    assert any(isinstance(a, RequestAnswer) for a in out)



# ------------------------------------------------------------------ call limit + resume
def _engine_mid_build():
    eng = _engine_at_confirm()
    eng.on_utterance("yes")
    eng.on_utterance("skip")                     # setup skipped -> step 1
    eng.on_utterance("done")                     # step 1 (wire) done -> step 2
    return eng


def test_resume_offer_continue():
    old = _engine_mid_build()
    snap = old.snapshot()
    assert snap["step_idx"] == 1 and snap["phase"] in ("place", "find")
    eng = BuildEngine(KIT, clock=Clock())
    eng.offer_resume(json.loads(json.dumps(snap)))       # survives a trip through the JSON file
    out = eng.start()
    assert eng.phase == Phase.RESUME
    assert "within 30 minutes" in _says(out) and "step 2 of" in _says(out) and "fresh" in _says(out)
    out = eng.on_utterance("yes let's carry on")
    assert "Step 2" in _says(out) and eng.plan.id == old.plan.id and eng.step_idx == 1


def test_resume_offer_fresh_and_unclear():
    snap = _engine_mid_build().snapshot()
    eng = BuildEngine(KIT, clock=Clock())
    eng.offer_resume(snap)
    eng.start()
    assert "didn't catch" in _says(eng.on_utterance("hmm"))
    out = eng.on_utterance("let's start a fresh one")
    assert eng.phase == Phase.INTAKE and eng.plan is None and "fresh start" in _says(out)


def test_nothing_to_resume_when_finished_or_not_started():
    eng = BuildEngine(KIT, clock=Clock())
    eng.start()
    assert eng.snapshot() is None                # nothing chosen yet
    eng.phase = Phase.DONE
    assert eng.snapshot() is None


def test_resume_file_window(tmp_path, monkeypatch):
    from backend import build_session as bs

    monkeypatch.setattr(bs, "RESUME_FILE", tmp_path / "resume.json")
    snap = _engine_mid_build().snapshot()
    bs.save_resume(snap)
    assert bs.load_resume(1800)["plan"]["id"] == snap["plan"]["id"]
    data = json.loads((tmp_path / "resume.json").read_text())
    data["saved_at"] -= 1801                      # 30 minutes and a second ago
    (tmp_path / "resume.json").write_text(json.dumps(data))
    assert bs.load_resume(1800) is None
    bs.save_resume(None)
    assert not (tmp_path / "resume.json").exists()



def test_waiting_room_sessions_never_touch_the_saved_build(tmp_path, monkeypatch):
    """chert dev replaces its idle waiting room every few minutes; that idle session must not wipe the
    build saved by the call that just ended, and the next call must still be offered it."""
    from backend import build_session as bs
    from backend.config import Settings
    from backend.hardware.serial_monitor import FakeSerialMonitor

    monkeypatch.setattr(bs, "RESUME_FILE", tmp_path / "resume.json")
    snap = _engine_mid_build().snapshot()

    class T:
        local_audio, wants_video = False, False
        def __init__(self): self.captions = []
        def start(self): pass
        def read_frame(self): return None
        def poll_transcripts(self): return []
        def send_speech(self, text, wav): pass
        def send_caption(self, text): self.captions.append(text)
        def send_video_frame(self, frame): pass
        def send_message(self, text, images=None): pass
        def close(self): pass

    class NoUploader:
        available = False

    def session(t):
        return bs.BuildSession(t, Settings(), planner=OfflinePlanner(), verifier=None, uploader=NoUploader(),
                               serial=FakeSerialMonitor(), hand_tracker=None)

    idle = session(T())          # waiting room prepared BEFORE the build was saved
    bs.save_resume(snap)         # the call ends with a build in progress
    idle.stop()                  # waiting room replaced with no caller
    assert bs.load_resume(1800) is not None

    t = T()
    caller = session(t)          # prepared earlier, but the caller joins now
    caller.start()
    try:
        assert caller.engine.phase == Phase.RESUME
        assert any("Welcome back" in c for c in t.captions)
    finally:
        caller.stop()            # hung up during the question: the offer is kept
    assert bs.load_resume(1800) is not None
