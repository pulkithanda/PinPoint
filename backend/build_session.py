"""
One call, Elegoo kit: the build engine wired to the call transport, the planner, Claude's eyes,
arduino-cli and the USB serial line.

Slow work (planning, photo checks, compiling, uploading) runs on a worker pool; results come back
through a queue and are fed to the engine on the next loop, so the call never waits on them.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from backend.build.board import BuildPlan, load_kit, parse_hole
from backend.build.engine import (BuildEngine, Finished, Phase, RequestPartID, RequestPlacementCheck, RequestPlan,
                                  RequestReply, RequestReview, RequestSetupCheck, RequestUpload)
from backend.build.planner import OfflinePlanner, Planner, PlanResult
from backend.build.visual_guidance import draw_breadboard_targets
from backend.config import ROOT, Settings
from backend.guidance.engine import RequestAnswer, Say
from backend.guidance.overlay import _ascii, _caption_bar, _label, caption_card
from backend.guidance.speech import SpeechScheduler, TTSWorker
from backend.hardware.serial_monitor import SerialMonitor
from backend.hardware.uploader import Uploader, UploadResult, first_error
from backend.transport import CallTransport
from backend.vision.capture import AutoCapture
from backend.vision.hands import GestureDebouncer, HandTracker
from backend.vision.shield import ShieldLayout, ShieldTracker
from backend.vision.verify import BoardReview, BoardSetup, ModuleID, PlacementCheck, Verifier, claude_available

log = logging.getLogger(__name__)

PhotoRequest = RequestSetupCheck | RequestPartID | RequestPlacementCheck | RequestReview | RequestAnswer

_PROMPTS = {
    "setup": "Hold the phone upright and flat above the breadboard and the Arduino, back camera down, with the "
             "top of the phone pointing away from you. Hold still and I'll take the photo.",
    "identify": "Hold the part up in front of the phone's camera, about a hand's width away. Hold still.",
    "placement": "Pick up the phone, turn it sideways, and hold it flat about 30 centimetres above the breadboard, "
                 "back camera down. Hold still and I'll take the photo.",
    "review": "Hold the phone upright and flat above the breadboard and the Arduino, high enough to see both, back "
              "camera down, top of the phone pointing away from you. Hold still.",
    "question": "Point the phone at what you're asking about and hold still. I'll take the photo.",
}


RESUME_FILE = ROOT / "cache" / "resume.json"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def is_echo(heard: str, recently_said: list[str]) -> bool:
    """Did the caller's microphone just pick up Pinpoint's own voice? True when (nearly) every word heard
    appears, in order, in something Pinpoint said moments ago. ("Ask me a question." after Pinpoint said
    "... ask me a question, or say help.")"""
    h = _words(heard)
    if not h:
        return False
    for said in recently_said:
        w = _words(said)
        if len(h) > len(w) + 1:
            continue
        # longest run of heard words found in order inside what was said
        i = hits = 0
        for word in h:
            j = i
            while j < len(w) and w[j] != word:
                j += 1
            if j < len(w):
                hits, i = hits + 1, j + 1
        if len(h) >= 2 and hits >= len(h) - (1 if len(h) >= 4 else 0):   # one word is never judged echo
            return True
    return False


def split_sentences(text: str, min_len: int = 40) -> list[str]:
    """Split speech at sentence ends, merging short bits so each clip is a natural phrase."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < min_len:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out or [text]
_resume_lock = threading.Lock()


def save_resume(snap: dict | None) -> None:
    """Remember an unfinished build for the next call (or forget it, with None). Kept on disk so it
    survives an agent restart."""
    with _resume_lock:
        try:
            if snap is None:
                RESUME_FILE.unlink(missing_ok=True)
            else:
                RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
                RESUME_FILE.write_text(json.dumps({"saved_at": time.time(), "snapshot": snap}), encoding="utf-8")
        except OSError as e:
            log.warning("Could not save the build for resuming: %s", e)


def load_resume(window_s: float) -> dict | None:
    """The last unfinished build, if it ended less than `window_s` ago."""
    with _resume_lock:
        try:
            d = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if time.time() - float(d.get("saved_at", 0)) > window_s:
        return None
    return d.get("snapshot")


def fixed_phrases(mode: str = "") -> list[str]:
    """Lines the build flow always says, pre-rendered to speech before any call (so they play instantly).
    Long lines are split the same way they're spoken."""
    eng = BuildEngine(mode=mode)
    lines = [a.text for a in eng.start() if isinstance(a, Say)] + list(_PROMPTS.values())
    lines += list(eng._praise.options) + list(eng._looking.options) + list(eng._still.options)
    lines += [
        "Ooh, nice idea. Give me a moment to design that with the parts in your kit.",
        "Sure, let me rework the design.", "Great! First, let me take a quick look at how your board is set up.",
        "Got it. You can put the phone down.", "I didn't get a clear photo.", "I'm still checking the photo.",
        "Programming your Arduino now. It takes about twenty seconds.",
        "It works! I can see it responding. You built that. Brilliant job!",
        "Great, let's carry on.", "Good question, one second.", "Let me think.", "Okay, one moment.", "Okay, a fresh start. What would you like to build today?",
        "Heads up: we're near the five-minute call limit. If the call drops, just call back within "
        "30 minutes and we'll pick up where we left off.",
    ]
    out: list[str] = []
    for line in lines:
        out += split_sentences(line)
    return list(dict.fromkeys(out))


def _purpose(req: PhotoRequest) -> str:
    return {RequestSetupCheck: "setup", RequestPartID: "identify", RequestPlacementCheck: "placement",
            RequestReview: "review"}.get(type(req), "question")


class BuildSession:
    def __init__(self, transport: CallTransport, settings: Settings | None = None, *, planner: Planner | None = None,
                 verifier: Verifier | None = None, uploader: Uploader | None = None,
                 serial: SerialMonitor | None = None, hand_tracker: HandTracker | None = None):
        self.s = settings or Settings()
        self.transport = transport
        self.kit = load_kit()
        have_claude = self.s.verify_enabled and claude_available()
        if verifier is None and have_claude:
            verifier = Verifier(self.s.claude_model, self.s.claude_effort)
        self.verifier = verifier
        save = ROOT / "cache" / "plans"
        self.planner = planner or (Planner(self.s.claude_model, self.s.planner_effort, self.kit, save_dir=save)
                                   if have_claude else OfflinePlanner(kit=self.kit, save_dir=save))
        self.uploader = uploader or Uploader(ROOT / "cache" / "sketches")
        if not self.uploader.available:
            log.warning("arduino-cli not found: Pinpoint can't program the Arduino. Run python -m tools.setup_arduino")
        self.serial = serial or SerialMonitor(self.s.serial_port, self.s.serial_baud)
        self.hands = hand_tracker if hand_tracker is not None else HandTracker(self.s.gesture_model, self.s.fingertip_landmark)
        self.gestures = GestureDebouncer()

        self.engine = BuildEngine(self.kit, mode=self.s.mode, verifier_available=self.verifier is not None,
                                  uploader_available=self.uploader.available)
        # The existing ArUco tags around the Arduino define the frame used for breadboard overlays.
        self.marker_tracker = ShieldTracker(ShieldLayout.load(self.s.layout_path), recheck_s=0.25)
        self._last_marker_update = 0.0
        self.fingertip_px: tuple[float, float] | None = None
        self._call_started: float | None = None
        self._limit_warned = False
        self.scheduler = SpeechScheduler(self.s.speech_min_interval_s, self.s.speech_repeat_after_s)
        self.tts = TTSWorker.shared(ROOT / "cache" / "tts", speak_locally=transport.local_audio)

        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="build")
        self._speech_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech")
        self._said: deque[tuple[float, str]] = deque(maxlen=12)   # (time, line) Pinpoint spoke recently
        self._speech_epoch = 0
        self._results: queue.Queue = queue.Queue()
        self._inputs: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop = threading.Event()
        self._capture: tuple[AutoCapture, PhotoRequest] | None = None
        self.continuous_monitoring = False
        self._monitor_phase: Phase | None = None
        self._next_monitor_check = 0.0
        self._remote_busy_until = 0.0
        self._last_frame_t = 0.0
        self._last_card_t = 0.0
        self._last_serial_t = 0.0
        self._precompiled: str | None = None   # plan id whose sketch compiled cleanly in the background
        self._sketch_lock = threading.Lock()
        self.finished = False
        self.latest_frame: np.ndarray | None = None
        self.latest_annotated: np.ndarray | None = None
        self.snapshots: list[bytes] = []
        self.transcript_log: list[str] = []
        self.fps = 0.0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "BuildSession":
        """The caller is on the line. (Sessions are built ahead of time for chert dev's waiting room, so
        anything about the previous call, like an unfinished build, is read here, not in __init__.)"""
        self._call_started = time.monotonic()
        snap = load_resume(self.s.resume_window_s) if self.s.resume_window_s > 0 else None
        if snap is not None:
            log.info("Unfinished build from a recent call: offering to resume %s", snap["plan"].get("title"))
            self.engine.offer_resume(snap)
        self.transport.start()
        self.serial.start()
        self._handle(self.engine.start())
        self._thread = threading.Thread(target=self._loop, name="build-session", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._remember_build()
        self._stop.set()
        self.serial.stop()
        self.transport.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._speech_pool.shutdown(wait=False, cancel_futures=True)
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)
        if hasattr(self.hands, "close") and (thread is None or not thread.is_alive()):
            self.hands.close()

    def _remember_build(self) -> None:
        """Call over: keep an unfinished build for a call-back; forget a finished or untouched one."""
        try:
            if self._call_started is None:
                return                      # a waiting-room session that never had a caller: touch nothing
            if self.engine.phase == Phase.RESUME:
                return                      # they hung up before answering: keep the old offer as it was
            save_resume(self.engine.snapshot())
        except Exception:  # noqa: BLE001
            log.exception("Could not remember the build")

    def _call_limit_warning(self, now: float) -> None:
        """Chert cuts calls at call_limit_s: say so ~30 s before, so a dropped call isn't a surprise."""
        if self._limit_warned or self._call_started is None or self.s.call_limit_s <= 0:
            return
        # Chert's clock starts at answer, a few seconds before this session starts (bridge_ready).
        if now - self._call_started < self.s.call_limit_s - 45:
            return
        self._limit_warned = True
        building = self.engine.plan is not None and self.engine.phase.value in BuildEngine._RESUMABLE
        if building:
            save_resume(self.engine.snapshot())   # in case the call drops before stop() runs
            text = ("Heads up: we're near the five-minute call limit. If the call drops, just call back within "
                    "30 minutes and we'll pick up where we left off.")
        else:
            text = "Heads up: this call will end in about thirty seconds because of the five-minute limit. Just call back to carry on."
        log.info("Call-limit warning given")
        self._say(Say(text, "urgent"))

    def command(self, text: str) -> str:
        """Thread-safe: something the builder typed (dashboard, iMessage) or said."""
        self._inputs.put(("text", text))
        return "queued"

    def set_continuous_monitoring(self, enabled: bool, command_text: str) -> None:
        """Only explicit iMessage requests switch the Elegoo flow into continuous checking."""
        self._inputs.put(("monitoring", (enabled, command_text)))

    # ------------------------------------------------------------ main loop
    def _log_line(self, text: str) -> None:
        self.transcript_log.append(text)
        del self.transcript_log[:-12]

    def _loop(self) -> None:
        last = [time.monotonic()]
        errors = 0
        while not self._stop.is_set():
            try:
                self._step(last)
                errors = 0
            except Exception:  # noqa: BLE001 - one bad frame or result must not end the call's brain
                errors += 1
                log.exception("Session loop error (%d in a row); carrying on", errors)
                time.sleep(min(1.0, 0.05 * errors))

    def _step(self, last: list[float]) -> None:
        # caller started talking -> stop speaking first, before anything else is queued
        self._barge_in(time.monotonic())
        for text in self.transport.poll_transcripts():
            now_t = time.monotonic()
            recent = [t for when, t in self._said if now_t - when < 25.0]
            # Echo only arrives while Pinpoint is talking (or just after, given transcription delay); the
            # builder repeating a phrase Pinpoint suggested ("is this right?") later is a real question.
            # Only while Pinpoint is still talking: someone answering "carry on, or start a fresh one?" with
            # "start a fresh one" after she's finished is a real answer, not echo.
            talking_now = getattr(self.transport, "speaking", None)
            if talking_now is None:
                talking_now = now_t < self._remote_busy_until
            if talking_now and len(text.split()) >= 3 and not self.engine.expects_answer(text)                     and is_echo(text, recent):
                log.info("Ignored echo of my own voice: %r", text)
                continue
            self._log_line(f"> {text}")
            self._hush()                       # they spoke: anything queued from before is out of date
            self._handle(self.engine.on_utterance(text))
        while not self._inputs.empty():
            kind, payload = self._inputs.get_nowait()
            if kind == "monitoring":
                enabled, command_text = payload
                self._log_line(f"> {command_text}")
                if enabled and self.verifier is None:
                    self._set_continuous_monitoring(False)
                    self._say(Say("Continuous monitoring needs vision checks, which aren't available right now. "
                                  "I'll keep using picture checks.", "urgent"))
                else:
                    self._set_continuous_monitoring(enabled)
                    self._say(Say(
                        "Continuous monitoring is on. Keep video pointed at the circuit; I'll check each placement."
                        if enabled else "Continuous monitoring is off. I'll use picture checks when needed.",
                        "urgent",
                    ))
            else:
                text = payload
                self._log_line(f"> {text}")
                self._handle(self.engine.on_utterance(text))
        while not self._results.empty():
            self._dispatch(*self._results.get_nowait())

        item = self.transport.read_frame()
        now = time.monotonic()
        if item is not None:
            self._process_frame(item[1], now)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, now - last[0]))
            last[0] = now
        else:
            time.sleep(0.005)
        self._idle_video(now)
        self._call_limit_warning(now)
        self._handle(self.engine.on_tick(now))
        self._continuous_monitor_tick(now)
        if self.engine.phase == Phase.TEST and now - self._last_serial_t > 0.5 and self.engine.plan:
            self._last_serial_t = now
            key = self.engine.plan.test.key
            self._handle(self.engine.on_serial(self.serial.receiving(now), self.serial.values(key, now)))
        self._speak(now)

    def _set_continuous_monitoring(self, enabled: bool) -> None:
        self.continuous_monitoring = enabled
        self.engine.continuous_monitoring = enabled
        self._monitor_phase = None
        self._next_monitor_check = 0.0
        if enabled and not self.engine.mode_chosen:
            self.engine.mode = "voice"
            self.engine.mode_chosen = True

    def _continuous_monitor_tick(self, now: float) -> None:
        if not self.continuous_monitoring:
            return
        phase = self.engine.phase
        if phase != self._monitor_phase:
            self._monitor_phase = phase
            self._next_monitor_check = now + 5.0
        if phase in (Phase.PLACE, Phase.REVIEW) and not self.engine._pending and now >= self._next_monitor_check:
            self._next_monitor_check = now + 5.0
            self._handle(self.engine.on_utterance("done"))

    def _process_frame(self, frame: np.ndarray, now: float) -> None:
        self.latest_frame = frame
        if now - self._last_marker_update >= 0.25:
            self._last_marker_update = now
            self.marker_tracker.update(frame, now, force=True)
            if len(set(self.marker_tracker.visible_ids) & set(self.marker_tracker.layout.markers)) < 3:
                self.marker_tracker.H_board_from_img = self.marker_tracker.H_img_from_board = None
        self.fingertip_px = None
        if self.hands is not None:
            hands = self.hands.process(frame, now)
            g = max(hands, key=lambda h: h.gesture_score) if hands else None
            self.fingertip_px = g.fingertip_px if g else None
            gesture = self.gestures.update(g.gesture if g else None, g.gesture_score if g else 0.0, now)
            if gesture:
                self._handle(self.engine.on_gesture(gesture))
        if self._capture is not None:
            self._feed_capture(frame, now)
        self.latest_annotated = self._draw(frame)
        self.transport.send_video_frame(self.latest_annotated)
        self._last_frame_t = now

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        img = frame.copy()
        st = self.engine.view_state()
        h, w = img.shape[:2]
        n, i = st["n_steps"], st["step_index"]
        header = st["title"] + (f"  |  step {min(i + 1, n)}/{n}" if n else "") + (f"  |  {st['show']}" if st["show"] else "")
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (w, 56), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, img, 0.45, 0, dst=img)
        cv2.putText(img, _ascii(header), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1500), (255, 255, 255), 2,
                    cv2.LINE_AA)
        targets = self._visual_hole_targets()
        if self.engine.mode == "visual" and targets:
            if self.marker_tracker.ready:
                img = draw_breadboard_targets(img, self.marker_tracker, targets, self.engine.orientation,
                                              self.fingertip_px)
            else:
                _label(img, "Show at least 3 Arduino corner markers for hole overlays", (16, 82), 0.6, (0, 220, 255), 2)
        _caption_bar(img, st["caption"])
        return img

    def _visual_hole_targets(self) -> list[str]:
        if self.engine.phase == Phase.REVIEW and self.engine.plan is not None:
            steps = self.engine.plan.steps
        elif self.engine.phase in (Phase.FIND, Phase.PLACE, Phase.CHECKING) and self.engine.step is not None:
            steps = [self.engine.step]
        else:
            return []
        return list(dict.fromkeys(c.to for step in steps for c in step.connections if parse_hole(c.to)))

    def _idle_video(self, now: float) -> None:
        if getattr(self.transport, "wants_video", False) and now - self._last_frame_t > 0.5 \
                and now - self._last_card_t > 0.2:
            self._last_card_t = now
            self.latest_annotated = caption_card(self.engine.view_state())
            self.transport.send_video_frame(self.latest_annotated)

    # ------------------------------------------------------------ actions
    def _handle(self, actions) -> None:
        for a in actions:
            if isinstance(a, Say):
                self._say(a)
            elif isinstance(a, RequestPlan):
                self._pool.submit(self._plan_job, a)
            elif isinstance(a, RequestUpload):
                self._pool.submit(self._upload_job, a.plan)
            elif isinstance(a, RequestReply):
                self._pool.submit(self._reply_job, a)
            elif isinstance(a, (RequestSetupCheck, RequestPartID, RequestPlacementCheck, RequestReview, RequestAnswer)):
                self._request_photo(a)
            elif isinstance(a, Finished):
                self._finish(a.recap)

    def _dispatch(self, kind: str, res) -> None:
        if kind == "route":
            self._handle(self.engine.on_route(*res))
            return
        if kind == "answer" and isinstance(res, tuple):
            self._handle(self.engine.on_answer(*res))
            return
        handler = {
            "plan": self.engine.on_plan_result, "setup": self.engine.on_setup_result,
            "identify": self.engine.on_identify_result, "placement": self.engine.on_placement_result,
            "review": self.engine.on_review_result, "answer": self.engine.on_answer,
            "upload": self.engine.on_upload_result, "reply": self.engine.on_answer,
        }[kind]
        self._handle(handler(res))
        if kind == "plan" and isinstance(res, PlanResult) and res.plan is not None and self.uploader.available \
                and self.engine.plan is res.plan:          # not a design the builder replaced or cancelled
            self._pool.submit(self._precompile_job, res.plan)

    def _say(self, a: Say) -> None:
        log.info("Pinpoint: %s", a.text)
        self._log_line(a.text)
        self._said.append((time.monotonic(), a.text))
        self.transport.send_caption(a.text)
        if self.engine.mode != "visual":
            self.scheduler.submit(a, time.monotonic())

    def _barge_in(self, now: float) -> None:
        """The caller started talking: stop mid-sentence and drop what was queued, then listen."""
        poll = getattr(self.transport, "poll_barge_in", None)
        if poll is None or not poll() or not self.s.barge_in:
            return
        talking = getattr(self.transport, "speaking", now < self._remote_busy_until)
        if talking or self.scheduler.pending:
            self._speech_epoch += 1          # lines still rendering are dropped when they finish
            getattr(self.transport, "stop_speech", lambda: None)()
            self.scheduler.clear()
            self._remote_busy_until = 0.0
            log.info("Caller interrupted: stopped speaking")

    def _hush(self) -> None:
        """Stop talking now and forget queued lines (like a person who's been interrupted)."""
        self._speech_epoch += 1
        getattr(self.transport, "stop_speech", lambda: None)()
        self.scheduler.clear()
        self._remote_busy_until = 0.0

    def _send_speech(self, text: str, epoch: int) -> None:
        # Sentence by sentence: the first one starts playing while the next is still being rendered,
        # so a long instruction doesn't mean a long silence first. An interruption drops the rest.
        for part in split_sentences(text):
            if epoch != self._speech_epoch:
                return
            wav = self.tts.render(part)
            if epoch != self._speech_epoch:
                return
            self.transport.send_speech(part, wav)

    def _speak(self, now: float) -> None:
        self._barge_in(now)
        busy = self.tts.busy if self.transport.local_audio else now < self._remote_busy_until
        say = self.scheduler.pop_ready(now, busy=busy)
        if say is None:
            return
        if self.transport.local_audio:
            self.tts.say(say.text)
        else:
            self._remote_busy_until = now + 0.5 + len(say.text.split()) / 2.8
            # One speech worker: lines go down the call in the order they were said (with several
            # workers a short line could finish rendering first and jump ahead of a long one).
            self._speech_pool.submit(self._send_speech, say.text, self._speech_epoch)

    # ------------------------------------------------------------ slow jobs
    def _plan_job(self, req: RequestPlan) -> None:
        try:
            res = self.planner.plan(req.request, req.previous)
        except Exception:  # noqa: BLE001 - never leave the builder waiting in silence
            log.exception("Planner crashed")
            try:
                res = self.planner.fallback(req.request)
            except Exception:  # noqa: BLE001
                res = PlanResult(None, "Sorry, something went wrong while I was designing that.", "none")
        res.token = req.token
        self._results.put(("plan", res))

    def _reply_job(self, req: RequestReply) -> None:
        try:
            res = self.verifier.route(req.text, req.context, req.history, self.engine.mode) if self.verifier else None
        except Exception:  # noqa: BLE001
            log.exception("Route failed")
            res = None
        self._results.put(("route", (res, req.seq, req.text)))

    def _precompile_job(self, plan: BuildPlan) -> None:
        """Compile while the builder wires, and get Claude to fix the sketch if it doesn't build."""
        if not plan.uses_arduino:
            return                                   # power-module-only build: no code to compile
        with self._sketch_lock:
            ok_lib, out = self.uploader.install_libraries(plan.libraries)
            if not ok_lib:
                log.warning("Library install failed: %s", out[-300:])
            for attempt in range(3):
                res = self.uploader.compile(plan.id, plan.sketch)
                if res.ok:
                    self._precompiled = plan.id
                    log.info("Sketch for %s compiles (attempt %d)", plan.id, attempt + 1)
                    return
                if res.stage == "setup" or attempt == 2:
                    return
                log.info("Sketch for %s failed to compile (%s); asking Claude to fix it", plan.id, first_error(res.output))
                fixed = self.planner.fix_sketch(plan, res.output)
                if not fixed or (plan.test.key and f"{plan.test.key}=" not in fixed):
                    return  # keep the original; the upload step reports the compile error
                plan.sketch = fixed

    def _upload_job(self, plan: BuildPlan) -> None:
        try:
            self._upload_job_inner(plan)
        except Exception as e:  # noqa: BLE001 - always answer, or the engine waits forever
            log.exception("Upload job crashed")
            self.serial.resume()
            self._results.put(("upload", UploadResult(False, "upload", str(e),
                                                      "Something went wrong while programming the Arduino. "
                                                      "Unplug the USB cable, plug it back in, and say ready.")))

    def _upload_job_inner(self, plan: BuildPlan) -> None:
        with self._sketch_lock:  # wait for a background compile/fix to finish first
            port = self._port()
            if port is None:
                self._results.put(("upload", UploadResult(False, "upload", "no port",
                                                          "I can't find the Arduino. Check the USB cable is plugged "
                                                          "into both the Arduino and the laptop, then say ready.")))
                return
            wrong = self.uploader.wrong_board(port)
            if wrong:
                log.warning("Board on %s is %s, not an UNO R3", port, wrong)
                self._results.put(("upload", UploadResult(False, "board", wrong,
                                                          f"The board plugged in is an {wrong}, not the kit's "
                                                          "Arduino UNO R3. Plug in the UNO R3, the board with the "
                                                          "big square USB socket, and say ready.")))
                return
            self.serial.pause()
            try:
                res = self.uploader.upload(plan.id, plan.sketch, port)
                if not res.ok and res.stage == "compile":
                    fixed = self.planner.fix_sketch(plan, res.output)
                    if fixed:
                        plan.sketch = fixed
                        res = self.uploader.upload(plan.id, plan.sketch, port)
                    if not res.ok and res.stage == "compile":
                        res = UploadResult(False, "compile", res.output,
                                           "The code for this design doesn't compile, so I can't program it.")
            finally:
                time.sleep(0.5)
                self.serial.resume()
        self._results.put(("upload", res))

    def _port(self) -> str | None:
        fake = getattr(self.serial, "fake_port", None)
        if fake is not None:
            return fake
        if self.s.serial_port.lower() not in ("auto", "none"):
            return self.s.serial_port
        return SerialMonitor.find_port()

    # ------------------------------------------------------------ photos
    def _request_photo(self, req: PhotoRequest) -> None:
        purpose = _purpose(req)
        try:
            req._seq = self.engine.utter_seq      # which utterance this answers (RequestAnswer only matters)
        except AttributeError:
            pass
        if self.verifier is None:
            self._fail(req)
            return
        if self._capture is not None:
            self._fail(self._capture[1])
        self._capture = (AutoCapture(purpose, [], time.monotonic(), self.s.capture_timeout_s), req)
        if not self.continuous_monitoring:
            self._say(Say(_PROMPTS[purpose], "info"))

    def _feed_capture(self, frame: np.ndarray, now: float) -> None:
        cap, req = self._capture
        upd = cap.feed(frame, now, None)
        if upd.status == "waiting":
            if upd.hint and now - getattr(self, "_last_hint_t", 0.0) >= 4.0:
                self._last_hint_t = now
                self._say(Say(upd.hint, "steer", upd.hint_key))
            return
        self._capture = None
        if upd.status == "timeout" or upd.frame is None:
            self._say(Say("I didn't get a clear photo.", "info"))
            self._fail(req)
            return
        if not self.continuous_monitoring and not isinstance(req, (RequestSetupCheck, RequestReview)):
            self._say(Say("Got it. You can put the phone down.", "urgent", "photo_ok"))
        ok, jpg = cv2.imencode(".jpg", upd.frame)
        if ok and isinstance(req, (RequestPlacementCheck, RequestReview)):
            self.snapshots.append(jpg.tobytes())
        self._pool.submit(self._photo_job, req, upd.frame)

    def _photo_job(self, req: PhotoRequest, photo: np.ndarray) -> None:
        v, mode = self.verifier, self.engine.mode
        try:
            if isinstance(req, RequestSetupCheck):
                self._results.put(("setup", v.check_board_setup(photo)))
            elif isinstance(req, RequestPartID):
                self._results.put(("identify", v.identify_part(photo, req.candidates)))
            elif isinstance(req, RequestPlacementCheck):
                self._results.put(("placement", v.check_placement(photo, req.expected, req.already, mode)))
            elif isinstance(req, RequestReview):
                self._results.put(("review", v.review_board(photo, req.expected, mode)))
            else:
                ans = v.answer_question(photo, req.question, req.context, mode)
                self._results.put(("answer", (ans, getattr(req, "_seq", 0))))
        except Exception:  # noqa: BLE001
            log.exception("Photo check failed")
            self._fail(req)

    def _fail(self, req: PhotoRequest) -> None:
        if isinstance(req, RequestSetupCheck):
            self._results.put(("setup", BoardSetup(None, None, None, None, "no photo")))
        elif isinstance(req, RequestPartID):
            self._results.put(("identify", ModuleID("unknown", False, "no photo")))
        elif isinstance(req, RequestPlacementCheck):
            self._results.put(("placement", PlacementCheck(None, "", "no photo")))
        elif isinstance(req, RequestReview):
            self._results.put(("review", BoardReview(None, [], "no photo")))
        else:
            self._results.put(("answer", None))

    # ------------------------------------------------------------ end
    def _finish(self, recap: str) -> None:
        if self.finished:
            return
        self.finished = True
        save_resume(None)                   # finished: nothing to resume
        out_dir = self.s.recap_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recap.txt").write_text(recap, encoding="utf-8")
        if self.engine.plan is not None:
            (out_dir / f"{self.engine.plan.id}.ino").write_text(self.engine.plan.sketch, encoding="utf-8")
        for i, img in enumerate(self.snapshots):
            (out_dir / f"check{i + 1}.jpg").write_bytes(img)
        self.transport.send_message(recap, list(self.snapshots[-1:]) if self.engine.mode == "visual" else None)
        log.info("Recap saved to %s", out_dir)

    def status(self) -> dict:
        usage = dict(getattr(self.verifier, "usage", {}) or {})
        pu = getattr(self.planner, "usage", {}) or {}
        for k, v in pu.items():
            usage[k] = usage.get(k, 0) + v
        return {
            **self.engine.view_state(),
            "guidance": "continuous" if self.continuous_monitoring else "snapshot",
            "continuous_monitoring": self.continuous_monitoring,
            "marker_ready": self.marker_tracker.ready,
            "markers": self.marker_tracker.visible_ids,
            "call_active": self._call_started is not None and not self._stop.is_set()
            and not getattr(self.transport, "call_ended", False),
            "call_status": "on call" if self._call_started is not None and not self._stop.is_set()
            and not getattr(self.transport, "call_ended", False) else
            "ended" if self._call_started is not None else "waiting",
            "call_elapsed_s": max(0, int(time.monotonic() - self._call_started))
            if self._call_started is not None else 0,
            "session_running": getattr(self, "_thread", None) is not None and self._thread.is_alive(),
            "capturing": self._capture[0].purpose if self._capture else None,
            "claude_usage": usage,
            "fps": round(self.fps, 1),
            "verifier": self.verifier is not None,
            "uploader": self.uploader.available,
            "serial": {"connected": self.serial.connected, "receiving": self.serial.receiving()},
            "log": list(self.transcript_log),
        }
