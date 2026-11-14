"""
One session per call: frames in -> fast geometric path -> guidance -> speech/overlay out

The vision LLM (slow path) runs on a worker pool; its results are fed back to
the engine on the next frame, so guidance never waits on it.
"""
from __future__ import annotations

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
from backend.guidance.engine import (Beep, Finished, GuidanceEngine, Observation, Project,
                                     RequestIdentify, RequestPlugCheck, Say, parse_command)
from backend.guidance.overlay import draw_guidance, draw_port_labels
from backend.guidance.speech import LocalBeeper, SpeechScheduler, TTSWorker, common_phrases, tone_wav
from backend.hardware.serial_monitor import SerialMonitor
from backend.transport import CallTransport
from backend.vision.hands import GestureDebouncer, HandTracker, PointSmoother
from backend.vision.shield import ShieldLayout, ShieldTracker
from backend.vision.verify import ModuleID, PlugCheck, Verifier, claude_available

log = logging.getLogger(__name__)

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

        if verifier is None and self.s.verify_enabled and claude_available():
            verifier = Verifier(self.s.claude_model, self.s.claude_effort)
        self.verifier = verifier
        if self.verifier is None:
            log.warning("Vision LLM disabled (no Anthropic credentials or PINPOINT_VERIFY=0): "
                        "module ID is skipped and plug checks rely on the serial test.")

        self.engine = GuidanceEngine(
            self.project, self.layout, mode=self.s.mode, verifier_available=self.verifier is not None,
            place_then_check=self.s.place_then_check, sonify=self.s.sonify,
            hand_gone_check_s=self.s.hand_gone_check_s,
        )
        self.scheduler = SpeechScheduler(self.s.speech_min_interval_s, self.s.speech_repeat_after_s)
        self.tts = TTSWorker(ROOT / "cache" / "tts", speak_locally=transport.local_audio)
        if not transport.local_audio:
            self.tts.prerender(common_phrases(list(self.layout.ports)))
        self.beeper = LocalBeeper() if transport.local_audio else None
        self.serial = serial or SerialMonitor(self.s.serial_port, self.s.serial_baud)

        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="verify")
        self._results: queue.Queue = queue.Queue()
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._remote_busy_until = 0.0
        self._last_serial_poll = 0.0

        self.latest_frame: np.ndarray | None = None
        self.latest_annotated: np.ndarray | None = None
        self.fingertip_px: tuple[float, float] | None = None
        self.snapshots: list[bytes] = []
        self.transcript_log: deque[str] = deque(maxlen=12)
        self.fps = 0.0
        self.finished = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "Session":
        self.transport.start()
        self.serial.start()
        self._handle(self.engine.start())
        threading.Thread(target=self._loop, name="session", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.serial.stop()
        self.transport.close()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def command(self, text: str) -> str | None:
        """Thread-safe: queue a spoken/typed/button command."""
        cmd = parse_command(text)
        if cmd:
            self._commands.put(cmd)
        return cmd

    # ------------------------------------------------------------ main loop
    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            for text in self.transport.poll_transcripts():
                self.command(text)
            while not self._commands.empty():
                self._handle(self.engine.on_command(self._commands.get_nowait()))
            while not self._results.empty():
                kind, res = self._results.get_nowait()
                self._handle(self.engine.on_identify_result(res) if kind == "identify" else self.engine.on_plug_result(res))

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
            if now - self._last_serial_poll > 0.5:
                self._last_serial_poll = now
                self._handle(self.engine.on_serial_status(self.serial.status(now)))
            self._speak(now)

    def process_frame(self, frame: np.ndarray, now: float) -> None:
        self.latest_frame = frame
        shield_ok = self.shield.update(frame, now)
        hands = self.hands.process(frame, now)

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

        state = self.engine.view_state()
        self.latest_annotated = draw_guidance(frame, self.shield, state, self.fingertip_px, now)
        if self.engine.mode == "visual":
            self.transport.send_video_frame(self.latest_annotated)

    # ------------------------------------------------------------ actions
    def _handle(self, actions) -> None:
        for a in actions:
            if isinstance(a, Say):
                self.transcript_log.append(a.text)
                if self.engine.mode != "visual":  # visual mode: captions only
                    self.scheduler.submit(a, time.monotonic())
            elif isinstance(a, Beep):
                if self.engine.mode == "voice":
                    if self.beeper:
                        self.beeper.set(a.closeness)
                    else:
                        self.transport.send_beep(tone_wav(500 + 1000 * a.closeness))
            elif isinstance(a, RequestIdentify):
                self._pool.submit(self._identify_job, a)
            elif isinstance(a, RequestPlugCheck):
                self._pool.submit(self._plug_job, a)
            elif isinstance(a, Finished):
                self._finish(a.recap)
        if self.beeper and self.engine.phase.value != "guide":
            self.beeper.set(None)

    def _identify_job(self, req: RequestIdentify) -> None:
        frame = self.latest_frame
        if frame is None or self.verifier is None:
            self._results.put(("identify", ModuleID("unknown", False, "no frame")))
            return
        self._results.put(("identify", self.verifier.identify_module(frame, req.candidates)))

    def _plug_job(self, req: RequestPlugCheck) -> None:
        time.sleep(0.7)  # let the hand clear the port
        frame = self.latest_frame
        if frame is None or self.verifier is None:
            self._results.put(("plug", PlugCheck(None, None, "no verifier")))
            return
        annotated = draw_port_labels(frame, self.shield, highlight=req.ports)
        ok, jpg = cv2.imencode(".jpg", annotated)
        if ok:
            self.snapshots.append(jpg.tobytes())
        self._results.put(("plug", self.verifier.check_plug(annotated, req.module, req.ports)))

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
        # Blind mode gets audio-friendly text only; visual mode also gets annotated screenshots.
        self.transport.send_message(recap, images if self.engine.mode == "visual" else None)
        log.info("Recap saved to %s", out_dir)

    # ------------------------------------------------------------ dashboard
    def status(self) -> dict:
        st = self.serial.status()
        return {
            **self.engine.view_state(),
            "fps": round(self.fps, 1),
            "shield_ok": self.shield.ready,
            "markers": self.shield.visible_ids,
            "verifier": self.verifier is not None,
            "serial": {"connected": st.connected, "receiving": st.receiving, "dist": st.last_dist, "lcd_ok": st.lcd_ok},
            "log": list(self.transcript_log),
        }
