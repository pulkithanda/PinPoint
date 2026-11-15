"""
Speech in: the caller's audio -> utterances -> text, with local faster-whisper (open source).

Chert does not transcribe, so the adapter feeds raw PCM audio here. A simple
energy-based segmenter cuts the stream into utterances at ~0.6 s of silence,
and each utterance is transcribed on a worker thread (small model, CPU, ~0.3-1 s
for a short command).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

WHISPER_RATE = 16000


def to_float_16k(pcm16: np.ndarray, rate: int) -> np.ndarray:
    x = pcm16.astype(np.float32) / 32768.0
    if rate != WHISPER_RATE and len(x):
        n = int(len(x) * WHISPER_RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


class UtteranceSegmenter:
    """Collects speech between silences. feed() returns a finished utterance (float32 @16 kHz) or None."""

    def __init__(self, rms_threshold: float = 0.015, end_silence_s: float = 0.6,
                 min_speech_s: float = 0.25, max_utterance_s: float = 12.0):
        self.rms_threshold = rms_threshold
        self.end_silence_s = end_silence_s
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self._buf: list[np.ndarray] = []
        self._speech_s = 0.0
        self._silence_s = 0.0
        self._loud_run_s = 0.0
        self.barge_in_s = 0.35        # this much continuous speech counts as the caller starting to talk
        self.speech_started = False   # set once per utterance when barge_in_s is reached; caller clears it

    def feed(self, pcm16: np.ndarray, rate: int) -> np.ndarray | None:
        x = to_float_16k(pcm16, rate)
        if not len(x):
            return None
        dur = len(x) / WHISPER_RATE
        loud = float(np.sqrt(np.mean(x ** 2))) >= self.rms_threshold
        if loud:
            self._buf.append(x)
            self._speech_s += dur
            self._silence_s = 0.0
            before = self._loud_run_s
            self._loud_run_s += dur
            if before < self.barge_in_s <= self._loud_run_s:
                self.speech_started = True
        else:
            self._loud_run_s = 0.0
        if not loud and self._buf:
            self._buf.append(x)  # keep trailing silence so words aren't clipped
            self._silence_s += dur
        total = sum(len(b) for b in self._buf) / WHISPER_RATE
        if self._buf and (self._silence_s >= self.end_silence_s or total >= self.max_utterance_s):
            utt, enough = np.concatenate(self._buf), self._speech_s >= self.min_speech_s
            self._buf, self._speech_s, self._silence_s = [], 0.0, 0.0
            return utt if enough else None
        return None


class Transcriber:
    """Background faster-whisper worker. Call feed_pcm() with audio; on_text(text) fires per utterance."""

    def __init__(self, on_text: Callable[[str], None], model: str = "base.en",
                 on_speech_start: Callable[[], None] | None = None):
        self.on_text = on_text
        self.on_speech_start = on_speech_start
        self.model_name = model
        self.segmenter = UtteranceSegmenter()
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        threading.Thread(target=self._run, name="stt", daemon=True).start()

    def feed_pcm(self, pcm16: np.ndarray, rate: int) -> None:
        utt = self.segmenter.feed(pcm16, rate)
        if self.segmenter.speech_started:
            self.segmenter.speech_started = False
            if self.on_speech_start is not None:
                self.on_speech_start()
        if utt is not None:
            self._q.put(utt)

    def _run(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            log.info("Speech-to-text ready (faster-whisper %s)", self.model_name)
        except Exception as e:
            log.error("faster-whisper unavailable (%s); voice commands disabled", e)
            return
        while True:
            audio = self._q.get()
            try:
                segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                log.error("Transcription failed: %s", e)
                continue
            if text:
                log.info("Heard: %s", text)
                self.on_text(text)
