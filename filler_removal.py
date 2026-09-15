"""Whisper-based filler-word detection for the live pipeline (Stage 2).

Pure, pipeline-agnostic: given a list of 48 kHz stereo s16 PCM frames (the
CATCHUP backlog) and the absolute sequence number of the first frame,
`analyze()` returns the set of absolute frame sequence numbers that fall
inside a filler word ("um", "uh", …). The mixer drops those frames to
reclaim time, exactly like it drops over-long silence.

Design notes:
  * Runs on a background thread, never in the mixer's 50fps loop.
  * Model: faster-whisper "tiny", int8, CPU. ~0.2s to transcribe a few
    seconds of audio on Apple Silicon.
  * Fail-open: any error (model missing, transcription failure) returns an
    empty set, so the live audio path is never affected.
  * Conservative filler set by default — only non-lexical fillers, NOT
    "like"/"you know", to avoid cutting real words.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Set

import numpy as np

log = logging.getLogger("aspire.filler")

# "base" rather than "tiny". Measured: with tiny, only ~2 s of filler was found
# across a 16-minute talk even though the backlog sat at 5-15 s for ~35% of the
# run — so the limit was the model's transcription accuracy, not the amount of
# audio available to analyze. Non-lexical fillers ("um", "uh") are exactly the
# tokens a tiny model drops or mis-transcribes. base is ~2x the compute but
# still int8-on-CPU and runs off the real-time path, so it costs no latency.
# Override with ASPIRE_FILLER_MODEL (tiny | base | small).
MODEL_SIZE = os.environ.get("ASPIRE_FILLER_MODEL", "base")
FRAME_MS = 20                 # must match pipeline.FRAME_MS
SRC_RATE = 48_000            # capture/mixer rate
SRC_CHANNELS = 2
WHISPER_RATE = 16_000        # faster-whisper expects 16 kHz mono float32

# Non-lexical fillers only. "like" / "you know" are excluded by default
# because they're also legitimate words and cutting them corrupts meaning.
FILLER_WORDS = {"um", "uh", "uhm", "umm", "uhh", "er", "err",
                "ah", "ahh", "hmm", "hm", "mm", "mhm", "eh"}

_PUNCT = str.maketrans("", "", ".,!?;:\"'…-—–()[]")


def _normalize(word: str) -> str:
    return word.strip().lower().translate(_PUNCT)


class FillerRemover:
    """Lazy-loaded Whisper wrapper. One instance, reused across calls."""

    def __init__(self, model_size: str = MODEL_SIZE):
        self._model_size = model_size
        self._model = None
        self._unavailable = False  # set True if import/load fails — fail-open

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type="int8")
            log.info("[filler] whisper '%s' model loaded", self._model_size)
            return True
        except Exception as e:
            self._unavailable = True
            log.warning("[filler] whisper unavailable (%s) — filler removal "
                        "disabled, silence reclaim still active", e)
            return False

    def analyze(self, frames: List[bytes], base_seq: int) -> Set[int]:
        """Transcribe the backlog `frames` and return absolute frame seqs that
        land inside a filler word. `base_seq` is the absolute seq of frames[0].
        Returns an empty set on any problem (fail-open)."""
        if not frames or not self._ensure_model():
            return set()
        try:
            joined = b"".join(frames)
            stereo = np.frombuffer(joined, dtype=np.int16)
            if stereo.size == 0:
                return set()
            stereo = stereo.reshape(-1, SRC_CHANNELS)
            # Downmix to mono float32 in [-1, 1]
            mono = stereo.mean(axis=1).astype(np.float32) / 32768.0
            # Resample 48k -> 16k (exact 3:1 decimation via polyphase)
            from scipy.signal import resample_poly
            mono16 = resample_poly(mono, up=1, down=3).astype(np.float32)

            segments, _ = self._model.transcribe(
                mono16, language="en", word_timestamps=True,
                vad_filter=False)

            drop: Set[int] = set()
            frames_per_sec = 1000.0 / FRAME_MS  # 50
            for seg in segments:
                for w in (seg.words or []):
                    if _normalize(w.word) in FILLER_WORDS:
                        start_f = int(w.start * frames_per_sec)
                        end_f = int(w.end * frames_per_sec + 0.999)
                        for off in range(max(0, start_f), end_f):
                            drop.add(base_seq + off)
            if drop:
                log.info("[filler] flagged %d frames (%.1fs) across backlog "
                         "of %d frames", len(drop),
                         len(drop) * FRAME_MS / 1000.0, len(frames))
            return drop
        except Exception as e:
            log.warning("[filler] analyze failed: %s", e)
            return set()
