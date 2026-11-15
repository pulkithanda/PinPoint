"""
One session per call: frames in -> fast geometric path -> guidance -> speech/captions out.

Two guidance styles (PINPOINT_GUIDANCE):
  * snapshot (default): the phone stays on the FaceTime call but is put down. When Pinpoint
    needs to look (identify a module, check a plug, answer a question) it asks the builder to
    hold the phone up, auto-captures a steady, sharp frame from the 15 fps stream, and says
    "you can put the phone down".
  * live: the phone sits on an overhead stand and the fingertip is steered every frame.

The vision LLM (slow path) runs on a worker pool; its results are fed back to the engine on
the next loop, so the call never waits on it.
"""
from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from backend.config import ROOT, Settings
from backend.guidance.engine import (Beep, Finished, GuidanceEngine, Observation, Project, RequestAnswer,
                                     RequestIdentify, RequestPlugCheck, Say, is_question, parse_command)
from backend.guidance.overlay import caption_card, draw_guidance, draw_port_labels
from backend.guidance.speech import LocalBeeper, SpeechScheduler, TTSWorker, common_phrases, tone_wav
from backend.hardware.serial_monitor import SerialMonitor
from backend.transport import CallTransport
from backend.vision.capture import AutoCapture, capture_prompt
from backend.vision.crop import crop_to_hands, crop_to_shield
from backend.vision.hands import GestureDebouncer, HandTracker, PointSmoother
from backend.vision.shield import ShieldLayout, ShieldTracker
from backend.vision.verify import ModuleID, PlugCheck, Verifier, claude_available

log = logging.getLogger(__name__)

PhotoRequest = RequestIdentify | RequestPlugCheck | RequestAnswer


class Session:
    def __init__(self, transport: CallTransport, settings: Settings | None = None, project: str = "proximity_alert",
                 serial: SerialMonitor | None = None, hand_tracker: HandTracker | None = None,
                 verifier: Verifier | None = None):
        self.s = settings or Settings()
        self.transport = transport
        self.layout = ShieldLayout.load(self.s.layout_path)
        self.project = Project.load(self.s.projects_dir / f"{project}.yaml")
        self.shield = ShieldTracker(self.layout, recheck_s=self.s.shield_recheck_s)
        self.hands = hand_tracker or HandTracker(self.s.gesture_model, self.s.fingertip_landmark)
        self.smoother = PointSmoother(self.s.smoothing_frames)
        self.gestures = GestureDebouncer()
        self.live = self.s.live_steering

        if verifier is None and self.s.verify_enabled and claude_available():
            verifier = Verifier(self.s.claude_model, self.s.claude_effort)
        self.verifier = verifier
        if self.verifier is None:
            log.warning("Vision LLM disabled (no Anthropic credentials or PINPOINT_VERIFY=0): "
                        "module ID and questions are skipped; plug checks rely on the serial test.")

        self.engine = GuidanceEngine(
            self.project, self.layout, mode=self.s.mode, verifier_available=self.verifier is not None,
            steering=self.live, sonify=self.s.sonify, hand_gone_check_s=self.s.hand_gone_check_s,
        )
        self.scheduler = SpeechScheduler(self.s.speech_min_interval_s, self.s.speech_repeat_after_s)
        self.tts = TTSWorker.shared(ROOT / "cache" / "tts", speak_locally=transport.local_audio)
        if not transport.local_audio:
            self.tts.prerender(common_phrases(list(self.layout.ports)))
        self.beeper = LocalBeeper() if transport.local_audio else None
        self.serial = serial or SerialMonitor(self.s.serial_port, self.s.serial_baud)

        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="verify")
        self._results: queue.Queue = queue.Queue()
        self._commands: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop = threading.Event()
        self._remote_busy_until = 0.0
        self._last_serial_poll = 0.0
        self._capture: tuple[AutoCapture, PhotoRequest] | None = None
        self._last_frame_t = 0.0
        self._last_card_t = 0.0
        self._ended_logged = False
        self._recent: deque = deque(maxlen=12)

        self.latest_frame: np.ndarray | None = None
        self.latest_annotated: np.ndarray | None = None
        self.fingertip_px: tuple[float, float] | None = None
        self.snapshots: list[bytes] = []
        self.transcript_log: deque[str] = deque(maxlen=12)
        self.fps = 0.0
        self.finished = False
        self._call_started: float | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "Session":
        self._call_started = time.monotonic()
        self.transport.start()
        self.serial.start()
        self._handle(self.engine.start())
        self._thread = threading.Thread(target=self._loop, name="session", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.serial.stop()
        self.transport.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)
        if hasattr(self.hands, "close") and (thread is None or not thread.is_alive()):
            self.hands.close()

    def command(self, text: str) -> str | None:
        """Thread-safe: queue a spoken/typed/button command or a free-form question."""
        self.transcript_log.append(f"> {text}")
        cmd = parse_command(text)
        if cmd:
            self._commands.put(("cmd", cmd))
            return cmd
        if is_question(text):
            self._commands.put(("question", text.strip()))
            return "question"
        return None

    # ------------------------------------------------------------ main loop
    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            for text in self.transport.poll_transcripts():
                self.transcript_log.append(f"> {text}")
                self.command(text)
            while not self._commands.empty():
                kind, val = self._commands.get_nowait()
                self._handle(self.engine.on_command(val) if kind == "cmd" else self.engine.on_question(val))
            while not self._results.empty():
                kind, res = self._results.get_nowait()
                handler = {"identify": self.engine.on_identify_result, "plug": self.engine.on_plug_result,
                           "answer": self.engine.on_answer}[kind]
                self._handle(handler(res))

            item = self.transport.read_frame()
            if item is not None:
                ts, frame = item
                self.process_frame(frame, ts)
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, now - last))
                last = now
            else:
                time.sleep(0.005)

            now = time.monotonic()
            self._idle_video(now)
            if getattr(self.transport, "call_ended", False) and not self._ended_logged:
                self._ended_logged = True
                log.info("Call ended")
            if now - self._last_serial_poll > 0.5:
                self._last_serial_poll = now
                self._handle(self.engine.on_serial_status(self.serial.status(now)))
            self._speak(now)

    def process_frame(self, frame: np.ndarray, now: float) -> None:
        self.latest_frame = frame
        if self.live:
            shield_ok = self.shield.update(frame, now)
        else:
            # Hand-held phone: the view changes constantly, so never trust a cached board position.
            self.shield.update(frame, now, force=True)
            if len(set(self.shield.visible_ids) & set(self.layout.markers)) < 3:
                self.shield.H_board_from_img = self.shield.H_img_from_board = None
            shield_ok = self.shield.ready
        hands = self.hands.process(frame, now)
        # Board position + hands for recent frames, so a captured photo can be cropped with the
        # geometry of *that* frame (AutoCapture may pick one a few frames back).
        self._recent.append((frame, self.shield.H_img_from_board, self.shield.H_board_from_img, hands))

        hand = None
        if hands:
            def score(h):  # prefer the hand whose fingertip is over the board
                if not shield_ok:
                    return 0.0
                b = self.shield.to_board(h.fingertip_px)
                return -self.layout.nearest_port(b)[1] if b else -1e9
            hand = max(hands, key=score)
        self.fingertip_px = hand.fingertip_px if hand else None
        fp_board = self.shield.to_board(hand.fingertip_px) if (hand and shield_ok) else None
        fp = self.smoother.add(fp_board)

        g = max(hands, key=lambda h: h.gesture_score) if hands else None
        gesture = self.gestures.update(g.gesture if g else None, g.gesture_score if g else 0.0, now)

        self._handle(self.engine.on_observation(Observation(now, shield_ok, fp, gesture)))
        if self._capture is not None:
            self._feed_capture(frame, now)

        state = self.engine.view_state()
        self.latest_annotated = draw_guidance(frame, self.shield, state, self.fingertip_px, now)
        # The agent's video is where captions live on FaceTime (Chert has no captions API).
        self.transport.send_video_frame(self.latest_annotated)
        self._last_frame_t = now

    def _idle_video(self, now: float) -> None:
        """No caller frames (phone face down / camera off): keep the caption visible anyway."""
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
            elif isinstance(a, Beep):
                if self.engine.mode == "voice":
                    if self.beeper:
                        self.beeper.set(a.closeness)
                    else:
                        self.transport.send_beep(tone_wav(500 + 1000 * a.closeness))
            elif isinstance(a, (RequestIdentify, RequestPlugCheck, RequestAnswer)):
                self._request_photo(a)
            elif isinstance(a, Finished):
                self._finish(a.recap)
        if self.beeper and self.engine.phase.value != "guide":
            self.beeper.set(None)

    def _say(self, a: Say) -> None:
        self.transcript_log.append(a.text)
        self.transport.send_caption(a.text)
        if self.engine.mode != "visual":  # visual mode: captions only
            self.scheduler.submit(a, time.monotonic())

    # ------------------------------------------------------------ photo on demand
    def _request_photo(self, req: PhotoRequest) -> None:
        if self.live:
            # Phone is on the stand: the current frame is the photo (wait a moment for the hand to clear).
            delay = 0.7 if isinstance(req, RequestPlugCheck) else 0.0
            self._pool.submit(self._job_after_delay, req, delay)
            return
        purpose = "plug" if isinstance(req, RequestPlugCheck) else "identify" if isinstance(req, RequestIdentify) else "question"
        ports = req.ports if isinstance(req, RequestPlugCheck) else []
        if self._capture is not None:  # a newer request replaces an unfinished one
            self._fail(self._capture[1])
        self._capture = (AutoCapture(purpose, ports, time.monotonic(), self.s.capture_timeout_s), req)
        target = self.layout.ports[ports[0]].spoken if len(ports) == 1 else "the I 2 C ports" if ports else ""
        self._say(Say(capture_prompt(purpose, target), "info"))

    def _feed_capture(self, frame: np.ndarray, now: float) -> None:
        cap, req = self._capture
        upd = cap.feed(frame, now, self.shield)
        if upd.status == "waiting":
            if upd.hint:
                self._say(Say(upd.hint, "steer", upd.hint_key))
            return
        self._capture = None
        if upd.status == "timeout" or upd.frame is None:
            self._say(Say("I didn't get a clear photo.", "info"))
            self._fail(req)
            return
        self._say(Say("Got it. You can put the phone down.", "urgent", "photo_ok"))
        photo = self._prepare_photo(req, upd.frame, upd.labeled)
        self._pool.submit(self._run_job, req, photo, upd.labeled)

    def _fail(self, req: PhotoRequest) -> None:
        if isinstance(req, RequestIdentify):
            self._results.put(("identify", ModuleID("unknown", False, "no photo")))
        elif isinstance(req, RequestPlugCheck):
            self._results.put(("plug", PlugCheck(None, None, "no photo")))
        else:
            self._results.put(("answer", None))

    def _job_after_delay(self, req: PhotoRequest, delay: float) -> None:
        time.sleep(delay)
        frame = self.latest_frame
        if frame is None:
            self._fail(req)
            return
        labeled = self.shield.ready
        self._run_job(req, self._prepare_photo(req, frame, labeled), labeled)

    def _frame_geometry(self, frame: np.ndarray):
        """(shield with that frame's homography, that frame's hands), or (None, None) if unknown."""
        try:
            recent = list(self._recent)  # live mode calls this from a worker thread
        except RuntimeError:
            recent = []
        for f, h_img, h_board, hands in reversed(recent):
            if f is frame:
                snap = copy.copy(self.shield)
                snap.H_img_from_board, snap.H_board_from_img = h_img, h_board
                return snap, hands
        return None, None

    def _prepare_photo(self, req: PhotoRequest, frame: np.ndarray, labeled: bool) -> np.ndarray:
        """Label and crop the photo for Claude. Fewer pixels = fewer tokens; crops keep full detail."""
        shield, hands = self._frame_geometry(frame)
        if isinstance(req, RequestPlugCheck):
            if labeled and shield is not None and shield.H_img_from_board is not None:
                photo = draw_port_labels(frame, shield, highlight=req.ports)
                ok, jpg = cv2.imencode(".jpg", photo)
                if ok:
                    self.snapshots.append(jpg.tobytes())  # full labelled photo for the recap
                photo, cropped = crop_to_shield(photo, shield)
                self._crop_log("plug", frame, photo, cropped)
                return photo
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                self.snapshots.append(jpg.tobytes())
            return frame
        if isinstance(req, RequestIdentify) and hands:
            photo, cropped = crop_to_hands(frame, hands)
            self._crop_log("identify", frame, photo, cropped)
            return photo
        return frame  # questions can be about anything in view: never crop

    @staticmethod
    def _crop_log(kind: str, frame: np.ndarray, photo: np.ndarray, cropped: bool) -> None:
        if cropped:
            fa, pa = frame.shape[0] * frame.shape[1], photo.shape[0] * photo.shape[1]
            log.info("Claude %s photo cropped %dx%d -> %dx%d (%.0f%% of the area)",
                     kind, frame.shape[1], frame.shape[0], photo.shape[1], photo.shape[0], 100 * pa / fa)

    def _run_job(self, req: PhotoRequest, photo: np.ndarray, labeled: bool) -> None:
        """`photo` is already labelled/cropped by _prepare_photo."""
        if self.verifier is None:
            self._fail(req)
            return
        if isinstance(req, RequestIdentify):
            self._results.put(("identify", self.verifier.identify_module(photo, req.candidates)))
        elif isinstance(req, RequestPlugCheck):
            self._results.put(("plug", self.verifier.check_plug(photo, req.module, req.ports, labeled=labeled)))
        else:
            self._results.put(("answer", self.verifier.answer_question(photo, req.question, req.context,
                                                                      self.engine.mode)))

    # ------------------------------------------------------------ output
    def _speak(self, now: float) -> None:
        busy = self.tts.busy if self.transport.local_audio else now < self._remote_busy_until
        say = self.scheduler.pop_ready(now, busy=busy)
        if say is None:
            return
        if self.transport.local_audio:
            self.tts.say(say.text)
        else:
            self._remote_busy_until = now + 0.5 + len(say.text.split()) / 2.8
            self._pool.submit(lambda t=say.text: self.transport.send_speech(t, self.tts.render(t)))

    def _finish(self, recap: str) -> None:
        if self.finished:
            return
        self.finished = True
        if self.beeper:
            self.beeper.set(None)
        out_dir = self.s.recap_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recap.txt").write_text(recap, encoding="utf-8")
        images = list(self.snapshots)
        if self.latest_annotated is not None:
            ok, jpg = cv2.imencode(".jpg", self.latest_annotated)
            if ok:
                images.append(jpg.tobytes())
        for i, img in enumerate(images):
            (out_dir / f"step{i + 1}.jpg").write_bytes(img)
        # Blind mode gets audio-friendly text only; visual mode also gets the plug-check photos.
        self.transport.send_message(recap, images if self.engine.mode == "visual" else None)
        log.info("Recap saved to %s", out_dir)

    # ------------------------------------------------------------ dashboard
    def status(self) -> dict:
        st = self.serial.status()
        return {
            **self.engine.view_state(),
            "guidance": "live" if self.live else "snapshot",
            "call_active": self._call_started is not None and not self._stop.is_set()
            and not getattr(self.transport, "call_ended", False),
            "call_status": "on call" if self._call_started is not None and not self._stop.is_set()
            and not getattr(self.transport, "call_ended", False) else
            "ended" if self._call_started is not None else "waiting",
            "call_elapsed_s": max(0, int(time.monotonic() - self._call_started))
            if self._call_started is not None else 0,
            "session_running": getattr(self, "_thread", None) is not None and self._thread.is_alive(),
            "capturing": self._capture[0].purpose if self._capture else None,
            "claude_usage": dict(getattr(self.verifier, "usage", {}) or {}),
            "fps": round(self.fps, 1),
            "shield_ok": self.shield.ready,
            "markers": self.shield.visible_ids,
            "verifier": self.verifier is not None,
            "serial": {"connected": st.connected, "receiving": st.receiving, "dist": st.last_dist, "lcd_ok": st.lcd_ok},
            "log": list(self.transcript_log),
        }
