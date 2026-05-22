# pipeline.py
"""
LivePipeline — in-memory ffmpeg → slide detect → Haiku → Kokoro → mixer →
WebRTC audio track.

Mixer model (v0.5): pause-buffer-catchup, NOT ducking.

  Default                       speaker frames go straight to mixed_q.
  Slide change detected         pause speaker output. Buffer incoming PCM.
                                Play "Slide N. <description>" at full volume.
  TTS finishes                  resume from buffer, run silence-compression
                                so the listener catches back up to live.
  Caught up                     return to default.
  New slide while catching up   pause again, play, resume — buffer can grow.
  Buffer > 30 s                 drop oldest 50 % and warn.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue as _stdlib_queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image

from slide_detect import (DetectorParams, FrameState, STABILITY_HAMMING,
                          STABILITY_WINDOW_S, detect_change, ocr_text,
                          text_hash as ocr_text_hash)

# Shared diag-event writer (Phase 2 instrumentation). Behavior-neutral —
# every call below is a fire-and-forget record() that drops to a queue.
import diag_events

log = logging.getLogger("aspire.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Audio constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000          # 960
BYTES_PER_FRAME = SAMPLES_PER_FRAME * CHANNELS * 2          # 3840 bytes (s16 stereo)

# Capture / detect tunables
VIDEO_W, VIDEO_H = 1280, 720
VIDEO_FPS = 5
DETECT_INTERVAL_S = 1.0 / VIDEO_FPS

# Detection (matches the user's old proven pHash-only detector)
SAMPLE_INTERVAL_S = 0.4          # 400ms — matches old detector
CHANGE_HAMMING = 6               # pHash distance to count a frame as "different"
                                 # (was 10; live screen-captured slides produce
                                 # frame-to-frame deltas of 4-8 on real changes)
SUSTAIN_FRAMES = 1               # consecutive different frames to confirm
                                 # (was 3; live single-shot slide cuts produce
                                 # one big spike then immediate stability —
                                 # the 5s STABILITY_WINDOW_S below is the real
                                 # anti-flicker guard. Sustain >1 was blocking
                                 # commits because changed_run reset to 0 on
                                 # the very next sample after the cut.)
STABILITY_WINDOW_S = 5.0         # candidate must persist this long before commit
DEDUP_HAMMING = 6                # ≤6 = same slide as a prior committed one
UNIQUE_HASHES_MAX = 500
MAX_TTS_QUEUE = 3
RECENT_TOPICS_KEEP = 5

# Mixer state machine
SILENCE_THRESH_DBFS = -45.0
SILENCE_PEAK = int(10 ** (SILENCE_THRESH_DBFS / 20.0) * 32767)   # ≈ 184
# 30s cap — matches the user's stated tolerance ("up to 30 seconds more than
# original duration"). A shorter cap is also the structural fix for the
# cross-video contamination bug: when the user switches between source
# videos, the buffer can hold at most 30s of stale audio from the previous
# video, instead of the prior 120s. Anything older is dropped.
MAX_BUFFER_FRAMES = int(30_000 / FRAME_MS)                       # 30 s = 1500 frames

# Silence compression is OFF in this build. CATCHUP passes speaker frames
# through 1:1 — this was the configuration that produced 0 stutter marks
# across multiple phone tests. Anchor gating handles slide-description timing.

# TTS pause-coupling: a ready TTS clip is held in a "pending" slot until
# either the listener-perceived audio has been silent for at least
# TTS_PAUSE_TRIGGER_S, OR the clip has been waiting for TTS_HOLD_MAX_S.
# Fixes the "TTS plays before the speaker even starts talking" symptom by
# slotting descriptions into natural speech pauses rather than the moment
# Kokoro finishes synth. Max-hold prevents indefinite delay.
TTS_PAUSE_TRIGGER_S = 0.30
TTS_HOLD_MAX_S = 20.0
# Safety-net timeout. If a pending TTS clip's anchor still hasn't been
# reached after this long (wall-clock since enqueue), force-promote with
# reason="anchor_giveup". Expected to fire ~never; covers pathological
# states where buffer-overflow read_pos advance didn't catch up.
TTS_ANCHOR_GIVEUP_S = 120.0

MIXED_Q_MAX = 5     # legacy / probe-tone fallback
# speaker_q is unbounded — frames are 3.8KB each, even 60s @ 50fps is ~11MB.
# Mixer pulls fast, so the queue stays small in steady state.

# Audio history ring buffer (for per-peer playheads + scrub-back).
# 60 s × 50 fps = 3000 frames × 3840 bytes ≈ 11 MB.
RING_HISTORY_S = 60
RING_HISTORY_FRAMES = RING_HISTORY_S * 1000 // FRAME_MS

# How far behind the producer a fresh peer starts (frames). 3 = 60ms.
PEER_INITIAL_LAG_FRAMES = 3
# If a peer's playhead falls more than this many frames behind, snap back.
PEER_UNDERRUN_FRAMES = 200   # 4 s

# How long between ffmpeg auto-respawn attempts (capped exponential backoff)
RESPAWN_BACKOFF_INIT_S = 0.5
RESPAWN_BACKOFF_MAX_S = 30.0

EWMA_WINDOW_S = 2.0   # for video_fps / audio_fps reporting

# HTTP audio segment output (HLS / AAC-in-MPEGTS, 4s segments)
# HLS is universally supported: native on iOS+macOS Safari, via hls.js on
# Chrome/Firefox/Android. WebM/Opus over MSE was rejected by iOS Safari.
AUDIO_SEG_DIR = Path("sessions/live/audio")
AUDIO_SEG_DURATION_S = 4
AUDIO_SEG_RETAIN = 90        # 90 × 4s = 6 min window. Generous headroom so a
                             # listener that briefly stalls or scrubs back
                             # never runs into a segment that just rotated out.
AUDIO_AAC_KBPS = 64
AUDIO_OUT_CHANNELS = 1       # mono is fine for speech

# Video (+ legacy audio) device index for the ffmpeg-avfoundation capture.
# Format: "video_idx:audio_idx". Audio half is only used by the BlackHole
# path; the SCK audio path ignores it.
# Continuity Camera / iPhone connections can shift screen-capture indices.
# To find the current index:
#     ffmpeg -hide_banner -f avfoundation -list_devices true -i ''
# Look for "Capture screen 0".
AVFOUNDATION_INPUT = os.environ.get(
    "ASPIRE_AVFOUNDATION_INPUT",
    os.environ.get("AVFOUNDATION_INPUT", "3:0"),  # legacy name honored
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resample_to_48k_stereo_int16(samples: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono float32 → 48k stereo int16 (interleaved)."""
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SAMPLE_RATE)
        up, down = SAMPLE_RATE // g, sr // g
        samples = resample_poly(samples, up, down).astype(np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    s16 = (samples * 32767.0).astype(np.int16)
    stereo = np.stack([s16, s16], axis=1)  # (N, 2)
    return stereo


def _frame_to_jpeg(frame_bgr: np.ndarray, max_side: int = 1024) -> bytes:
    h, w, _ = frame_bgr.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _frame_peak_int16(buf: bytes) -> int:
    """Peak abs value of an int16-stereo packed frame."""
    arr = np.frombuffer(buf, dtype=np.int16)
    if arr.size == 0:
        return 0
    return int(np.max(np.abs(arr)))


def make_sine_frame_bytes(phase: float, freq_hz: float = 440.0,
                          dbfs: float = -20.0) -> Tuple[bytes, float]:
    """Generate one 20ms 48k stereo s16 frame of a continuous sine.
    Returns (bytes, new_phase). dbfs sets amplitude."""
    n = SAMPLES_PER_FRAME
    amp = 10 ** (dbfs / 20.0)
    t = (np.arange(n, dtype=np.float64) + phase) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq_hz * t) * amp
    s16 = np.clip(wave * 32767.0, -32768, 32767).astype(np.int16)
    stereo = np.stack([s16, s16], axis=1)
    new_phase = (phase + n) % SAMPLE_RATE
    return stereo.tobytes(), new_phase


# ─────────────────────────────────────────────────────────────────────────────
class _EncoderWriter:
    """Background-thread writer that owns the encoder ffmpeg's stdin so the
    mixer never blocks on a slow encoder.

    Mixer pushes frames non-blockingly via submit(). A bounded queue
    (~4 s of audio) absorbs transient encoder slowdowns. If the queue fills
    we drop frames — a 20ms gap is undetectable, a multi-second mixer stall
    is not.

    The writer thread also detects encoder death (BrokenPipe / dead proc)
    and respawns it with exponential backoff (0.5s → 30s cap), then replays
    the last ~2 s of frames from a recent ring so the new encoder produces
    audible output immediately rather than 4 s of silence.
    """

    QUEUE_MAX = 200      # 200 × 20ms = 4 s of frames at 50fps
    SEED_FRAMES = 100    # 100 × 20ms = 2 s replayed on respawn

    def __init__(self, pipeline: "LivePipeline"):
        self.pipeline = pipeline
        self.q: "_stdlib_queue.Queue[bytes]" = _stdlib_queue.Queue(
            maxsize=self.QUEUE_MAX)
        self.recent: Deque[bytes] = deque(maxlen=self.SEED_FRAMES)
        self.dropped_total: int = 0
        self.respawns_total: int = 0
        # Rolling 60s window of (ts, write_ms) for blocking-time diagnostic.
        self.write_ms_window: Deque[Tuple[float, float]] = deque(maxlen=4000)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="encoder-writer")
        self._thread.start()

    def submit(self, buf: bytes) -> None:
        """Mixer-side: NEVER blocks. Drops the frame if the queue is full."""
        try:
            self.q.put_nowait(buf)
        except _stdlib_queue.Full:
            self.dropped_total += 1
            diag_events.record("encoder", "encoder_dropped",
                               bytes=len(buf), reason="submit_queue_full")

    def stop(self) -> None:
        self._stop.set()

    def block_ms_in_window(self, seconds: float) -> float:
        cutoff = time.time() - seconds
        while self.write_ms_window and self.write_ms_window[0][0] < cutoff:
            self.write_ms_window.popleft()
        return sum(ms for _, ms in self.write_ms_window)

    def _run(self) -> None:
        # Lift priority if we can. macOS lets you go negative without root in
        # some shells; otherwise we just stay at default.
        try:
            os.nice(-10)
            log.info("[encoder-writer] priority raised (nice -10)")
        except (PermissionError, OSError) as e:
            log.info("[encoder-writer] cannot raise priority (%s); default ok", e)

        backoff = 0.5
        while not self._stop.is_set():
            try:
                buf = self.q.get(timeout=0.2)
            except _stdlib_queue.Empty:
                continue
            self.recent.append(buf)
            if self._try_write(buf):
                backoff = 0.5
                continue
            # Encoder is gone. Respawn (with backoff) then seed.
            if self._stop.is_set() or not self.pipeline._running:
                return
            self._respawn_and_seed(backoff)
            backoff = min(30.0, backoff * 2)

    def _try_write(self, buf: bytes) -> bool:
        proc = self.pipeline._ff_encoder
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        try:
            t0 = time.perf_counter()
            proc.stdin.write(buf)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.write_ms_window.append((time.time(), elapsed_ms))
            diag_events.record("encoder", "encoder_write_attempt",
                               bytes=len(buf),
                               blocked_us=int(elapsed_ms * 1000))
            return True
        except (BrokenPipeError, ValueError, OSError) as e:
            log.warning("[encoder-writer] write failed: %s — encoder dead", e)
            diag_events.record("encoder", "encoder_dropped",
                               bytes=len(buf),
                               reason=f"write_error:{type(e).__name__}")
            return False

    def _respawn_and_seed(self, backoff: float) -> None:
        log.warning("[encoder] died, respawning in %.1fs (attempt #%d)",
                    backoff, self.respawns_total + 1)
        diag_events.record("encoder", "encoder_respawn",
                           backoff_s=round(backoff, 2),
                           attempt=self.respawns_total + 1)
        time.sleep(backoff)
        if self._stop.is_set() or not self.pipeline._running:
            return
        try:
            self.pipeline._spawn_encoder_ffmpeg(reset_disk=False)
            self.respawns_total += 1
            log.info("[encoder] respawn success — seeding %d frames",
                     len(self.recent))
            for f in list(self.recent):
                if not self._try_write(f):
                    log.warning("[encoder] seed write failed mid-replay")
                    break
        except Exception as e:
            log.warning("[encoder] respawn failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
class MixerRing:
    """Lock-free single-producer / multi-reader ring of 20ms s16-stereo
    frames as a contiguous numpy buffer. A peer keeps an integer playhead;
    each recv() does ring[playhead % N].copy() and advances. No lock — int
    reads/writes are atomic in CPython, and a one-frame tear is acceptable
    for 20ms audio (worst case: a reader gets the previous frame at a
    boundary).

    Buffer shape: (max_frames, samples_per_frame, channels) int16.
    Storage in int16 keeps memory at ~11 MB for 60 s @ 50 fps stereo.
    """

    def __init__(self, max_frames: int = RING_HISTORY_FRAMES,
                 samples_per_frame: int = SAMPLES_PER_FRAME,
                 channels: int = CHANNELS):
        self._N = int(max_frames)
        self._samples = int(samples_per_frame)
        self._channels = int(channels)
        # Shape (N, samples, channels). int16 packed.
        self._buf = np.zeros((self._N, self._samples, self._channels), dtype=np.int16)
        # Single int counter — head is the index of the NEXT slot to write to.
        # head - 1 is the most-recently-written frame index.
        self._head = 0
        self._silence_bytes = (
            np.zeros((self._samples, self._channels), dtype=np.int16).tobytes()
        )
        # Rolling 60s window of write timestamps for rate diagnostics
        self._write_ts: Deque[float] = deque(maxlen=4000)

    @property
    def silence_frame(self) -> bytes:
        return self._silence_bytes

    @property
    def capacity(self) -> int:
        return self._N

    def append(self, frame_bytes: bytes) -> int:
        """Producer-only. Writes the frame at head%N and returns the index of
        the just-written frame."""
        idx = self._head
        slot = idx % self._N
        arr = np.frombuffer(frame_bytes, dtype=np.int16).reshape(
            self._samples, self._channels
        )
        self._buf[slot, :, :] = arr
        self._head = idx + 1
        self._write_ts.append(time.time())
        return idx

    def writes_in_window(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        while self._write_ts and self._write_ts[0] < cutoff:
            self._write_ts.popleft()
        return len(self._write_ts)

    def head(self) -> int:
        # Index of the most-recently-written frame (head - 1), or -1 if empty
        return self._head - 1

    def oldest(self) -> int:
        h = self._head - 1
        if h < 0:
            return 0
        return max(0, h - self._N + 1)

    def get(self, idx: int) -> Optional[bytes]:
        """Reader-side. Returns frame bytes at idx, or None if too old, or
        b"" if in the future. Lock-free — copy out before head advances."""
        head = self._head - 1
        if head < 0:
            return None
        if idx > head:
            return b""
        oldest = max(0, head - self._N + 1)
        if idx < oldest:
            return None
        slot = idx % self._N
        # .tobytes() on a numpy slice copies — safe even if the producer
        # advances during/after this call.
        return self._buf[slot].tobytes()


@dataclass
class TTSClip:
    samples_int16_stereo: np.ndarray   # (N, 2) int16
    text: str
    slide_no: int
    enqueued_at: float = field(default_factory=time.time)
    # Set when the slide commit fires; used to measure commit→audible latency
    commit_time: float = 0.0
    haiku_ms: float = 0.0
    tts_synth_ms: float = 0.0
    # Absolute speaker-frame index (matches audio_frames_in / _mixer_speaker_pos
    # units) recorded at the moment the slide was committed. The mixer cannot
    # promote this clip from pending → active until _mixer_speaker_pos has
    # reached anchor_pos — i.e. until the listener has heard the speaker audio
    # surrounding the slide's appearance.
    anchor_pos: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class LivePipeline:
    def __init__(self):
        self.video_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        # speaker_q unbounded — see SPEAKER_Q_MAX comment above
        self.speaker_q: asyncio.Queue = asyncio.Queue()
        self.mixed_q: asyncio.Queue = asyncio.Queue(maxsize=MIXED_Q_MAX)
        self.tts_q: "asyncio.Queue[TTSClip]" = asyncio.Queue(maxsize=MAX_TTS_QUEUE)

        self.ring = MixerRing()
        # Legacy dedup containers removed in v10 — unique_hashes deque is the
        # single source of truth now. The classes RollingDedup / TextDedup
        # remain in this file for reference but are no longer instantiated.
        self.recent_descriptions: Deque[Tuple[float, str]] = deque(maxlen=10)
        # slide_counter increments only when a slide is actually described
        # (committed past the 5s gate AND past dedup), not on every candidate.
        self.slide_counter = 0
        self.recent_topics: Deque[str] = deque(maxlen=RECENT_TOPICS_KEEP)

        # Detector state (pHash-only, matches old proven detector)
        self._prev_phash = None
        self._changed_run = 0
        self._candidate_phash = None
        self._candidate_first_seen_ts: Optional[float] = None
        self._candidate_frame_bgr = None
        # Session-wide deque of every committed pHash (used for dedup)
        self.unique_hashes: Deque = deque(maxlen=UNIQUE_HASHES_MAX)
        # Most recent OCR text from the most-recent commit (for the listener UI)
        self.last_ocr_for_speaking: str = ""

        # ── instrumentation (Phase A) ─────────────────────────────────────
        # Last 5 (prev_phash - current_phash) Hamming values, newest first
        self._last_phash_dists: Deque[int] = deque(maxlen=5)
        # Last 5 min(committed_phash - candidate_phash) at commit-check time
        self._last_dedup_dists: Deque[int] = deque(maxlen=5)
        # Frames sampled by detector + max changed_run, both with 60s rolling window
        self._frames_sampled_window: Deque[float] = deque(maxlen=2000)
        self._changed_run_window: Deque[Tuple[float, int]] = deque(maxlen=2000)
        self._last_change_below_log_ts: float = 0.0
        self._last_detect_state_log_ts: float = 0.0
        self._candidate_replaced_count: int = 0

        # Public state read by the listener UI / WebRTC tracks
        self.currently_speaking_text: str = ""
        self.currently_speaking_slide: int = 0
        self.currently_speaking_started_at: float = 0.0

        # CPU/mem health (filled in by _cpu_monitor)
        self.cpu_percent: float = 0.0
        self.mem_gb: float = 0.0

        # In-flight async tasks: phash_str -> (task, text_hash_or_None, phash_obj)
        # Used to roll back dedup state if the TTS work is dropped.
        self._inflight: Dict[str, Tuple[asyncio.Task, Optional[str], Any]] = {}

        # Audio capture source — env var selects:
        #   "sck_cli"   → SystemAudioDump (ScreenCaptureKit) + ffmpeg resampler (default)
        #   "blackhole" → legacy ffmpeg-avfoundation device ":0"
        self._audio_capture_source = os.environ.get(
            "ASPIRE_AUDIO_CAPTURE", "sck_cli"
        )
        # Two child subprocesses for capture — video is always ffmpeg-avfoundation;
        # audio is either ffmpeg-avfoundation (BlackHole) or an SCKAudioCapture
        # instance that quacks like a Popen.
        self._ff_video: Optional[subprocess.Popen] = None
        self._ff_audio = None  # subprocess.Popen | SCKAudioCapture | None
        # HTTP audio encoder: third ffmpeg subprocess, fed by the mixer via
        # a dedicated writer thread (_encoder_writer). Writes rotating
        # HLS/AAC segments to disk for the listener.
        self._ff_encoder: Optional[subprocess.Popen] = None
        self._encoder_writer: Optional[_EncoderWriter] = None
        self._reader_threads: List[threading.Thread] = []
        self._respawn_threads: List[threading.Thread] = []
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Mixer state machine, exposed via /webrtc/status
        self.mixer_state = "IDLE"          # LIVE | PAUSED_FOR_TTS | CATCHUP | IDLE
        self.speaker_buffer_ms = 0
        # Silence compression is permanently OFF in this build. CATCHUP
        # state pops the buffer and emits frames unchanged. This is the
        # configuration that produced 0 stutter marks in phone tests.
        self._compression_disabled: bool = True
        self.compression_mode: str = "OFF"

        # EWMA / sliding-window byte+frame counters
        self._video_frames_window: Deque[float] = deque(maxlen=400)   # timestamps
        self._audio_frames_window: Deque[float] = deque(maxlen=2000)
        self._video_bytes_window: Deque[Tuple[float, int]] = deque(maxlen=400)
        self._audio_bytes_window: Deque[Tuple[float, int]] = deque(maxlen=2000)

        self._stats = {
            "started_at": None,
            "video_frames_in": 0,
            "audio_frames_in": 0,
            "events": 0,
            "tts_played": 0,
            "tts_dropped": 0,
            "vision_calls": 0,
            "vision_p50_ms": 0.0,
            "tts_synth_p50_ms": 0.0,
            "buffer_overflows": 0,
            "silence_skipped": 0,
            "mixed_overflow_drops": 0,
            "video_respawns": 0,
            "audio_respawns": 0,
            "described_total": 0,
            "dedup_skipped": 0,
            "tts_queue_drops": 0,
            "candidates_seen": 0,
            "candidates_committed": 0,
            "tts_clips_played_complete": 0,
            "tts_clips_interrupted": 0,
            "vision_empty_responses": 0,
            "audio_segments_written": 0,
            "audio_segments_pruned": 0,
        }
        # Audio-path timing windows
        self._commit_to_audible_ms: Deque[float] = deque(maxlen=50)
        self._ring_writes_per_sec_window: Deque[int] = deque(maxlen=60)
        # Per-frame mixer push_mixed timing (Phase 1 instrumentation)
        # Each entry: (wallclock_ts, microseconds spent in push_mixed)
        self._mixer_write_us_window: Deque[Tuple[float, float]] = deque(maxlen=4000)
        # Frames dropped when speaker_buffer hit the 120s cap (Phase 1)
        self._buffer_drop_frames_total: int = 0
        # Absolute monotonic count of speaker frames the mixer has EMITTED to
        # the listener (in the same units as audio_frames_in). Used as the
        # "have you heard up to this capture position yet?" reference for
        # anchor-gated TTS promotion. Advanced by:
        #   - LIVE branch when emitting a real speaker frame
        #   - CATCHUP branch when popping from speaker_buffer (incl. silence-skip pops)
        #   - Buffer overflow drops — treated as "consumed but not heard"
        # NOT advanced for silence pads, TTS frames, or capture-side processing.
        self._mixer_speaker_pos: int = 0
        # Total speaker frames dropped by smart silence compression. Surfaced
        # in status() so an operator can see how much time has been reclaimed.
        self._silence_dropped_total_frames: int = 0
        # HLS segment file appearance times — used to log inter-seg deltas
        self._last_seg_seen_idx: int = -1
        self._last_seg_seen_ts: float = 0.0
        self._late_segments_total: int = 0
        # Per-peer recv-time deques and underrun counts: peer_id -> ...
        self.peer_recv_times: Dict[str, Deque[float]] = {}
        self.peer_underruns: Dict[str, Deque[float]] = {}
        self._vision_times: Deque[float] = deque(maxlen=50)
        self._tts_times: Deque[float] = deque(maxlen=50)

        # ── Diagnostic windows (Phase 1 v2) ────────────────────────────────
        # Per-core CPU samples (60s window) and python_threads/tasks (60s)
        self._cpu_per_core_window: Deque[Tuple[float, List[float]]] = deque(maxlen=120)
        self._cpu_total_window: Deque[Tuple[float, float]] = deque(maxlen=600)
        self._mem_gb_window: Deque[Tuple[float, float]] = deque(maxlen=600)
        # mixer_state sampled every 1s for distribution (60-min window)
        self._mixer_state_window: Deque[Tuple[float, str]] = deque(maxlen=3600)
        # In-progress flags for tts/vision (set by task in _detect_loop)
        self.tts_in_progress: bool = False
        self.vision_in_progress: bool = False

    # ── status helpers ─────────────────────────────────────────────────────
    def _ewma_fps(self, window: Deque[float]) -> float:
        now = time.time()
        cutoff = now - EWMA_WINDOW_S
        # Cheap O(n) prune from the left
        while window and window[0] < cutoff:
            window.popleft()
        return len(window) / EWMA_WINDOW_S

    def _bytes_in_last(self, window: Deque[Tuple[float, int]], seconds: float) -> int:
        now = time.time()
        cutoff = now - seconds
        while window and window[0][0] < cutoff:
            window.popleft()
        return sum(b for _, b in window)

    def _count_in_window(self, window: Deque[float], seconds: float) -> int:
        cutoff = time.time() - seconds
        while window and window[0] < cutoff:
            window.popleft()
        return len(window)

    def _peer_recv_p50(self) -> float:
        all_t = []
        for q in self.peer_recv_times.values():
            all_t.extend(list(q)[-1500:])
        return float(np.median(all_t)) if all_t else 0.0

    def _peer_recv_p95(self) -> float:
        all_t = []
        for q in self.peer_recv_times.values():
            all_t.extend(list(q)[-1500:])
        return float(np.quantile(all_t, 0.95)) if len(all_t) >= 2 else 0.0

    def _peer_underruns_count(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        n = 0
        for q in self.peer_underruns.values():
            n += sum(1 for t in q if t >= cutoff)
        return n

    def _max_changed_run_in_window(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        while self._changed_run_window and self._changed_run_window[0][0] < cutoff:
            self._changed_run_window.popleft()
        if not self._changed_run_window:
            return 0
        return max(r for _, r in self._changed_run_window)

    def status(self) -> dict:
        if self._vision_times:
            self._stats["vision_p50_ms"] = float(np.median(self._vision_times) * 1000)
        if self._tts_times:
            self._stats["tts_synth_p50_ms"] = float(np.median(self._tts_times) * 1000)
        try:
            from vision_ocr import stats as ocr_stats
            self._stats.update(ocr_stats())
        except Exception:
            pass
        v_alive = self._ff_video is not None and self._ff_video.poll() is None
        a_alive = self._ff_audio is not None and self._ff_audio.poll() is None
        cand_age = (time.time() - self._candidate_first_seen_ts
                    if self._candidate_first_seen_ts else 0.0)
        return {
            "running": self._running,
            **self._stats,
            "video_fps": self._ewma_fps(self._video_frames_window),
            "audio_fps": self._ewma_fps(self._audio_frames_window),
            "video_bytes_in_last_5s": self._bytes_in_last(self._video_bytes_window, 5.0),
            "audio_bytes_in_last_5s": self._bytes_in_last(self._audio_bytes_window, 5.0),
            "ffmpeg_video_pid": self._ff_video.pid if self._ff_video else None,
            "ffmpeg_audio_pid": self._ff_audio.pid if self._ff_audio else None,
            "ffmpeg_video_alive": v_alive,
            "ffmpeg_audio_alive": a_alive,
            "mixer_state": self.mixer_state,
            "compression_mode": self.compression_mode,
            "speaker_buffer_ms": self.speaker_buffer_ms,
            "speaker_q_size": self.speaker_q.qsize(),
            "speaker_q": self.speaker_q.qsize(),
            "video_q": self.video_q.qsize(),
            "mixed_q": self.mixed_q.qsize(),
            "mixed_q_size": self.mixed_q.qsize(),
            "tts_q_size": self.tts_q.qsize(),
            "tts_q": self.tts_q.qsize(),
            "unique_hashes_size": len(self.unique_hashes),
            "candidate_age_s": round(cand_age, 2),
            "candidate_phash_excerpt": (str(self._candidate_phash)[:8]
                                        if self._candidate_phash else ""),
            "last_5_phash_distances": [int(x) for x in self._last_phash_dists],
            "last_5_dedup_distances": [int(x) for x in self._last_dedup_dists],
            "candidate_replaced_count": self._candidate_replaced_count,
            "frames_sampled_in_last_60s": self._count_in_window(
                self._frames_sampled_window, 60.0),
            "changed_run_max_in_last_60s": self._max_changed_run_in_window(60.0),
            "currently_speaking_text": self.currently_speaking_text,
            "currently_speaking_slide": self.currently_speaking_slide,
            "currently_speaking_ocr": self.last_ocr_for_speaking[:300],
            "ring_head": self.ring.head(),
            "ring_oldest": self.ring.oldest(),
            "ring_history_s": RING_HISTORY_S,
            "cpu_percent": round(self.cpu_percent, 1),
            "mem_gb": round(self.mem_gb, 2),
            # Audio-path diagnostics (Phase A)
            "ring_writes_per_sec_p50": int(np.median(list(
                self._ring_writes_per_sec_window))) if self._ring_writes_per_sec_window else 0,
            "ring_writes_per_sec_min_in_last_60s": (
                min(self._ring_writes_per_sec_window)
                if self._ring_writes_per_sec_window else 0),
            "encoder_writer_dropped_total": (
                self._encoder_writer.dropped_total
                if self._encoder_writer else 0),
            "encoder_writer_respawns_total": (
                self._encoder_writer.respawns_total
                if self._encoder_writer else 0),
            "encoder_writer_q_size": (
                self._encoder_writer.q.qsize()
                if self._encoder_writer else 0),
            "encoder_alive": (self._ff_encoder is not None
                              and self._ff_encoder.poll() is None),
            "encoder_pid": (self._ff_encoder.pid
                            if self._ff_encoder else None),
            "buffer_drop_frames_total": self._buffer_drop_frames_total,
            "mixer_read_pos": self._mixer_speaker_pos,
            "mixer_write_pos": self._stats.get("audio_frames_in", 0),
            "mixer_lag_frames": max(
                0,
                self._stats.get("audio_frames_in", 0) - self._mixer_speaker_pos,
            ),
            # Alias for back-compat (renamed; older clients may still read this)
            "mixer_read_write_gap_frames": max(
                0,
                self._stats.get("audio_frames_in", 0) - self._mixer_speaker_pos,
            ),
            "silence_dropped_total_frames": self._silence_dropped_total_frames,
            "silence_dropped_total_ms":
                self._silence_dropped_total_frames * FRAME_MS,
            "late_segments_total": self._late_segments_total,
            "commit_to_audible_p50_ms": float(np.median(
                self._commit_to_audible_ms)) if self._commit_to_audible_ms else 0.0,
            "commit_to_audible_p95_ms": float(np.quantile(
                list(self._commit_to_audible_ms), 0.95)
            ) if len(self._commit_to_audible_ms) >= 2 else 0.0,
            "peer_recv_p50_ms": self._peer_recv_p50(),
            "peer_recv_p95_ms": self._peer_recv_p95(),
            "peer_underruns_in_last_60s": self._peer_underruns_count(60.0),
            "recent_descriptions": [
                {"t": t, "text": s} for t, s in list(self.recent_descriptions)[-5:]
            ],
            "recent_topics": list(self.recent_topics),
        }

    def diagnostic_snapshot(self) -> dict:
        """Point-in-time snapshot for /diagnostics/snapshot. Returns every
        queue depth, process state, mixer state, buffer ms, and recent (60s)
        jitter metrics. Designed for a human to read after correlating a
        user-marked stutter — NOT used in the steady-state log."""
        now = time.time()
        cutoff = now - 60.0

        # Mixer write_us jitter over last 60s
        xs = sorted(us for ts, us in self._mixer_write_us_window if ts >= cutoff)
        mixer = {
            "writes_60s": len(xs),
            "mean_us": (sum(xs) / len(xs)) if xs else 0.0,
            "max_us": xs[-1] if xs else 0.0,
            "p99_us": xs[max(0, int(len(xs) * 0.99) - 1)] if xs else 0.0,
        }

        # Encoder stdin block_ms jitter
        w = self._encoder_writer
        encoder_jitter = {
            "block_ms_60s": w.block_ms_in_window(60.0) if w else 0.0,
            "dropped_total": w.dropped_total if w else 0,
            "respawns_total": w.respawns_total if w else 0,
            "q_size": w.q.qsize() if w else 0,
            "q_max": _EncoderWriter.QUEUE_MAX,
        }

        # Per-core CPU latest sample
        per_core = (self._cpu_per_core_window[-1][1]
                    if self._cpu_per_core_window else [])

        # Process state
        def alive(p):
            return p is not None and p.poll() is None

        # Peer info (HLS has none server-side; WebRTC peers tracked in app)
        peer_info = []
        try:
            from app import _peer_sessions
            head = self.ring.head()
            for s in _peer_sessions.values():
                peer_info.append({
                    "peer_id": s.peer_id,
                    "playhead": s.playhead,
                    "behind_live_ms": ((head - s.playhead) * FRAME_MS
                                       if s.playhead >= 0 else None),
                })
        except Exception:
            pass

        return {
            "now": now,
            "started_at": self._stats.get("started_at"),
            "running": self._running,
            "mixer_state": self.mixer_state,
            "compression_mode": self.compression_mode,
            "speaker_buffer_ms": self.speaker_buffer_ms,
            "queues": {
                "video_q": self.video_q.qsize(),
                "speaker_q": self.speaker_q.qsize(),
                "mixed_q": self.mixed_q.qsize(),
                "tts_q": self.tts_q.qsize(),
                "encoder_writer_q": w.q.qsize() if w else 0,
            },
            "ring": {
                "head": self.ring.head(),
                "oldest": self.ring.oldest(),
                "writes_in_1s": self.ring.writes_in_window(1.0),
            },
            "ffmpeg": {
                "video_pid": self._ff_video.pid if self._ff_video else None,
                "video_alive": alive(self._ff_video),
                "audio_pid": self._ff_audio.pid if self._ff_audio else None,
                "audio_alive": alive(self._ff_audio),
                "encoder_pid": self._ff_encoder.pid if self._ff_encoder else None,
                "encoder_alive": alive(self._ff_encoder),
            },
            "mixer_jitter_60s": mixer,
            "encoder_jitter_60s": encoder_jitter,
            "buffer_drop_frames_total": self._buffer_drop_frames_total,
            "mixer_read_pos": self._mixer_speaker_pos,
            "mixer_write_pos": self._stats.get("audio_frames_in", 0),
            "mixer_lag_frames": max(
                0,
                self._stats.get("audio_frames_in", 0) - self._mixer_speaker_pos,
            ),
            # Alias for back-compat (renamed; older clients may still read this)
            "mixer_read_write_gap_frames": max(
                0,
                self._stats.get("audio_frames_in", 0) - self._mixer_speaker_pos,
            ),
            "silence_dropped_total_frames": self._silence_dropped_total_frames,
            "silence_dropped_total_ms":
                self._silence_dropped_total_frames * FRAME_MS,
            "late_segments_total": self._late_segments_total,
            "audio_segments_written": self._stats.get("audio_segments_written", 0),
            "cpu_total_latest": self.cpu_percent,
            "cpu_per_core_latest": list(per_core),
            "mem_gb": self.mem_gb,
            "tts_in_progress": self.tts_in_progress,
            "vision_in_progress": self.vision_in_progress,
            "peers": peer_info,
        }

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stats["started_at"] = time.time()
        self.mixer_state = "LIVE"
        self._loop = asyncio.get_running_loop()

        # Phase-2 diagnostic event writer. Truncates the JSONL each start.
        diag_events.start()
        diag_events.record("system", "pipeline_started",
                           compression_disabled=self._compression_disabled)
        log.info("[mixer] silence compression: OFF "
                 "(CATCHUP passes speaker frames 1:1)")

        # Start capture subprocesses + reader threads.
        log.info("[capture] video source: avfoundation device %s",
                 AVFOUNDATION_INPUT)
        self._spawn_video_ffmpeg()
        if self._audio_capture_source == "sck_cli":
            log.info("[capture] audio source: sck_cli "
                     "(SystemAudioDump via ScreenCaptureKit, 24kHz->48kHz resampled)")
            self._spawn_audio_sck()
        else:
            log.info("[capture] audio source: blackhole (legacy avfoundation)")
            self._spawn_audio_ffmpeg()
        self._spawn_encoder_ffmpeg()

        self._tasks = [
            asyncio.create_task(self._fps_logger(), name="fps_logger"),
            asyncio.create_task(self._detect_loop(), name="detect"),
            asyncio.create_task(self._mixer_loop(), name="mixer"),
            asyncio.create_task(self._cpu_monitor(), name="cpu_monitor"),
            asyncio.create_task(self._ring_logger(), name="ring_logger"),
            asyncio.create_task(self._segment_pruner(), name="seg_pruner"),
            asyncio.create_task(self._audio_diag_logger(), name="audio_diag"),
            asyncio.create_task(self._summary_logger(), name="summary"),
        ]
        log.info("[pipeline] started (video pid=%s audio pid=%s)",
                 self._ff_video.pid if self._ff_video else None,
                 self._ff_audio.pid if self._ff_audio else None)

    async def stop(self) -> None:
        if not self._running:
            return
        diag_events.record("system", "pipeline_stopping")
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        for proc in (self._ff_video, self._ff_audio):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
        if self._encoder_writer is not None:
            try:
                self._encoder_writer.stop()
            except Exception:
                pass
            self._encoder_writer = None
        if self._ff_encoder:
            try:
                if self._ff_encoder.stdin and not self._ff_encoder.stdin.closed:
                    self._ff_encoder.stdin.close()
            except Exception:
                pass
            try:
                self._ff_encoder.terminate()
                self._ff_encoder.wait(timeout=2)
            except Exception:
                try: self._ff_encoder.kill()
                except Exception: pass
            self._ff_encoder = None
        self._ff_video = None
        self._ff_audio = None
        # Reader threads will exit when poll() returns or pipe closes
        for th in self._reader_threads:
            th.join(timeout=2)
        self._reader_threads.clear()
        self.mixer_state = "IDLE"
        diag_events.record("system", "pipeline_stopped",
                           dropped_events_total=diag_events.dropped_count())
        diag_events.stop()
        log.info("[pipeline] stopped")

    # ── peer-attach hook ───────────────────────────────────────────────────
    def drain_mixed_for_new_peer(self) -> int:
        """Empty stale frames so a new peer starts at the current 'now'."""
        dropped = 0
        while not self.mixed_q.empty():
            try:
                self.mixed_q.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            log.info("[mixer] drained %d stale frames for new peer", dropped)
        return dropped

    async def enqueue_tts_text(self, text: str, slide_no: Optional[int] = None) -> None:
        """Synthesize `text` via Kokoro and push as a TTS clip. Used by the
        late-joiner re-announce and by the /webrtc/debug-slide endpoint."""
        from tts_kokoro import synth as kokoro_synth
        try:
            t0 = time.time()
            samples, sr = await kokoro_synth(text)
            self._tts_times.append(time.time() - t0)
        except Exception as e:
            log.warning("[tts] direct synth failed: %s", e)
            return
        stereo = _resample_to_48k_stereo_int16(samples, sr)
        if slide_no is None:
            # Don't bump slide_counter here — that counter is reserved for
            # actual committed slide descriptions. Use a sentinel.
            slide_no = self.slide_counter
        clip = TTSClip(samples_int16_stereo=stereo, text=text, slide_no=slide_no)
        if self.tts_q.full():
            try:
                dropped = self.tts_q.get_nowait()
                self._stats["tts_dropped"] += 1
                log.info("[mixer] tts queue full, dropping older clip: %r", dropped.text[:60])
            except asyncio.QueueEmpty:
                pass
        await self.tts_q.put(clip)
        self.recent_descriptions.append((time.time(), text))

    # ── ffmpeg capture (split into two subprocesses) ───────────────────────
    def _video_cmd(self) -> List[str]:
        # Just the screen index, no audio. Drop input framerate to 15 to save
        # CPU; output fps via -r to get a clean constant 5fps.
        screen_idx = AVFOUNDATION_INPUT.split(":")[0]
        return [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "+genpts",
            "-thread_queue_size", "1024",
            "-probesize", "10M",
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-pixel_format", "uyvy422",
            "-framerate", "15",
            "-i", screen_idx,
            "-map", "0:v",
            "-vf", f"scale={VIDEO_W}:{VIDEO_H}",
            "-r", str(VIDEO_FPS),
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

    def _audio_cmd(self) -> List[str]:
        # Just the audio device index, no video.
        audio_idx = AVFOUNDATION_INPUT.split(":")[1] if ":" in AVFOUNDATION_INPUT \
                    else AVFOUNDATION_INPUT
        return [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "+genpts",
            "-thread_queue_size", "1024",
            "-f", "avfoundation",
            "-i", f":{audio_idx}",
            "-map", "0:a",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-af", "aresample=async=1000:first_pts=0",
            "-f", "s16le",
            "pipe:1",
        ]

    def _spawn_video_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH")
        cmd = self._video_cmd()
        log.info("[pipeline] launching VIDEO ffmpeg: %s", " ".join(cmd))
        self._ff_video = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        # Start a stderr drainer so warnings hit our log instead of blocking
        threading.Thread(target=self._drain_stderr,
                         args=(self._ff_video, "video"),
                         daemon=True).start()
        # Frame reader thread
        th = threading.Thread(
            target=self._video_reader_thread,
            args=(self._ff_video,),
            daemon=True, name="video-reader",
        )
        self._reader_threads.append(th)
        th.start()

    def _spawn_audio_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH")
        cmd = self._audio_cmd()
        log.info("[pipeline] launching AUDIO ffmpeg: %s", " ".join(cmd))
        self._ff_audio = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._drain_stderr,
                         args=(self._ff_audio, "audio"),
                         daemon=True).start()
        th = threading.Thread(
            target=self._audio_reader_thread,
            args=(self._ff_audio,),
            daemon=True, name="audio-reader",
        )
        self._reader_threads.append(th)
        th.start()

    def _spawn_audio_sck(self) -> None:
        """SCK path: SystemAudioDump (Swift CLI using ScreenCaptureKit) piped
        through ffmpeg for 24k→48k resampling. Presents a subprocess-like
        surface so the existing _audio_reader_thread can read .stdout exactly
        like it does for ffmpeg-avfoundation.

        SCKAudioCapture drains both child stderrs itself, so we do NOT call
        _drain_stderr for this capture object.
        """
        from audio_capture_sck import SCKAudioCapture
        cap = SCKAudioCapture()
        cap.start()  # raises RuntimeError with remediation steps on failure
        self._ff_audio = cap
        th = threading.Thread(
            target=self._audio_reader_thread,
            args=(cap,),
            daemon=True, name="audio-reader",
        )
        self._reader_threads.append(th)
        th.start()

    def _spawn_encoder_ffmpeg(self, reset_disk: bool = True) -> None:
        """Third ffmpeg subprocess: reads s16le 48k stereo from its stdin
        (fed by the dedicated _EncoderWriter thread) and writes rotating
        HLS/AAC segments + a live playlist to disk. HLS is natively supported
        by iOS+macOS Safari and plays via hls.js on every other browser.

        reset_disk=True wipes prior segments + playlist (initial spawn).
        reset_disk=False keeps them (respawn) so listeners see seamless
        continuation as soon as a new segment lands.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH")
        AUDIO_SEG_DIR.mkdir(parents=True, exist_ok=True)
        if reset_disk:
            # Wipe any leftover from a previous run
            for old in list(AUDIO_SEG_DIR.glob("*.ts")) + \
                       list(AUDIO_SEG_DIR.glob("*.m3u8")) + \
                       list(AUDIO_SEG_DIR.glob("*.m4s")) + \
                       list(AUDIO_SEG_DIR.glob("*.mp4")) + \
                       list(AUDIO_SEG_DIR.glob("*.webm")):
                try: old.unlink()
                except OSError: pass

        playlist_path = str(AUDIO_SEG_DIR / "live.m3u8")
        seg_pattern = str(AUDIO_SEG_DIR / "seg_%05d.aac")
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-i", "pipe:0",
            "-ac", str(AUDIO_OUT_CHANNELS),
            "-c:a", "aac", "-b:a", f"{AUDIO_AAC_KBPS}k",
            "-f", "hls",
            "-hls_time", str(AUDIO_SEG_DURATION_S),
            "-hls_list_size", str(AUDIO_SEG_RETAIN),
            "-hls_flags", "delete_segments+append_list+independent_segments",
            "-hls_segment_type", "fmp4",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", str(AUDIO_SEG_DIR / "seg_%05d.m4s"),
            playlist_path,
        ]
        log.info("[pipeline] launching ENCODER ffmpeg: %s", " ".join(cmd))
        self._ff_encoder = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._drain_stderr,
                         args=(self._ff_encoder, "encoder"),
                         daemon=True).start()
        # Start the dedicated writer thread the first time only. The same
        # writer survives encoder respawns — it just routes to the new proc.
        if self._encoder_writer is None:
            self._encoder_writer = _EncoderWriter(self)

    def _encoder_write(self, frame_bytes: bytes) -> None:
        """Submit a 20ms s16le stereo frame to the encoder writer thread.
        NEVER blocks the mixer — drops the frame if the writer's queue is
        full. The writer thread does the actual stdin.write and handles
        encoder death + respawn behind the scenes."""
        w = self._encoder_writer
        if w is None:
            return
        w.submit(frame_bytes)

    def _drain_stderr(self, proc: subprocess.Popen, label: str) -> None:
        try:
            for line in iter(proc.stderr.readline, b""):
                if not line:
                    break
                msg = line.decode("utf-8", errors="replace").rstrip()
                if msg:
                    log.info("[ffmpeg/%s] %s", label, msg)
        except Exception as e:
            log.info("[ffmpeg/%s] stderr drain ended: %s", label, e)

    def _put_threadsafe(self, q: asyncio.Queue, item) -> None:
        loop = self._loop
        if loop is None:
            return
        def _do_put():
            if q.full():
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass
            try: q.put_nowait(item)
            except asyncio.QueueFull: pass
        loop.call_soon_threadsafe(_do_put)

    @staticmethod
    def _read_exact(stdout, n: int) -> Optional[bytes]:
        """Read exactly n bytes from a binary pipe. Returns None on real EOF.
        Pipe reads can short — loop until full frame or pipe closes."""
        buf = bytearray()
        while len(buf) < n:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                # True EOF (pipe closed). Return what we have if anything.
                return bytes(buf) if buf else None
            buf.extend(chunk)
        return bytes(buf)

    def _video_reader_thread(self, proc: subprocess.Popen) -> None:
        frame_bytes = VIDEO_W * VIDEO_H * 3
        stdout = proc.stdout
        eof = False
        last_read_ts = 0.0
        frame_idx = 0
        try:
            while self._running:
                if proc.poll() is not None:
                    eof = True
                    break
                buf = self._read_exact(stdout, frame_bytes)
                if buf is None or len(buf) < frame_bytes:
                    eof = True
                    break
                self._stats["video_frames_in"] += 1
                now = time.time()
                self._video_frames_window.append(now)
                self._video_bytes_window.append((now, len(buf)))
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((VIDEO_H, VIDEO_W, 3))
                self._put_threadsafe(self.video_q, arr)
                frame_idx += 1
                since_ms = (now - last_read_ts) * 1000.0 if last_read_ts else 0.0
                last_read_ts = now
                diag_events.record("capture", "video_capture_read",
                                   frame_index=frame_idx,
                                   since_last_read_ms=round(since_ms, 2))
        except Exception as e:
            log.warning("[ffmpeg/video] reader exception: %s", e)
            eof = True
        finally:
            log.warning("[ffmpeg/video] reader exited (proc rc=%s, eof=%s)",
                        proc.poll(), eof)
            if self._running and eof:
                self._schedule_respawn("video", RESPAWN_BACKOFF_INIT_S)

    def _audio_reader_thread(self, proc: subprocess.Popen) -> None:
        stdout = proc.stdout
        eof = False
        last_read_ts = 0.0
        try:
            while self._running:
                if proc.poll() is not None:
                    eof = True
                    break
                buf = self._read_exact(stdout, BYTES_PER_FRAME)
                if buf is None or len(buf) < BYTES_PER_FRAME:
                    bytes_got = 0 if buf is None else len(buf)
                    diag_events.record("capture", "audio_capture_underrun",
                                       bytes_got=bytes_got,
                                       expected=BYTES_PER_FRAME)
                    eof = True
                    break
                self._stats["audio_frames_in"] += 1
                now = time.time()
                self._audio_frames_window.append(now)
                self._audio_bytes_window.append((now, len(buf)))
                self._put_threadsafe(self.speaker_q, buf)
                since_ms = (now - last_read_ts) * 1000.0 if last_read_ts else 0.0
                last_read_ts = now
                q_size = self.speaker_q.qsize()
                diag_events.record("capture", "audio_capture_read",
                                   bytes_read=len(buf),
                                   expected_bytes=BYTES_PER_FRAME,
                                   since_last_read_ms=round(since_ms, 3))
                diag_events.record("queue", "speaker_q_put",
                                   q_size_after=q_size)
        except Exception as e:
            log.warning("[ffmpeg/audio] reader exception: %s", e)
            eof = True
        finally:
            log.warning("[ffmpeg/audio] reader exited (proc rc=%s, eof=%s)",
                        proc.poll(), eof)
            if self._running and eof:
                self._schedule_respawn("audio", RESPAWN_BACKOFF_INIT_S)

    def _schedule_respawn(self, which: str, backoff: float) -> None:
        """Background-respawn a dead ffmpeg with capped exponential backoff."""
        def _runner():
            wait = backoff
            attempts = 0
            while self._running:
                attempts += 1
                log.warning("[respawn/%s] attempt #%d after %.1fs", which, attempts, wait)
                time.sleep(wait)
                if not self._running:
                    return
                try:
                    if which == "video":
                        self._stats["video_respawns"] += 1
                        self._spawn_video_ffmpeg()
                    else:
                        self._stats["audio_respawns"] += 1
                        if self._audio_capture_source == "sck_cli":
                            self._spawn_audio_sck()
                        else:
                            self._spawn_audio_ffmpeg()
                    log.info("[respawn/%s] success", which)
                    return
                except Exception as e:
                    log.warning("[respawn/%s] failed: %s", which, e)
                    wait = min(RESPAWN_BACKOFF_MAX_S, wait * 2)
        th = threading.Thread(target=_runner, daemon=True, name=f"respawn-{which}")
        self._respawn_threads.append(th)
        th.start()

    async def _cpu_monitor(self) -> None:
        """Sample total + per-core CPU and RSS every 1s. We need 1-Hz sampling
        for the diag/summary loggers to compute p99 over 30s windows.

        cpu_percent (self.cpu_percent) is the process CPU normalized to a
        single-core baseline (process_cpu / num_cores), matching Activity
        Monitor's display. Per-core values are system-wide and used to find
        a single saturated core (Haiku/Kokoro burst → one core pegs)."""
        try:
            import psutil
        except ImportError:
            log.warning("[health] psutil not available, cpu monitor disabled")
            return
        proc = psutil.Process()
        ncores = max(1, psutil.cpu_count() or 1)
        # Prime the moving average — first call returns 0.0 in psutil
        proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        log_tick = 0
        while self._running:
            await asyncio.sleep(1.0)
            now = time.time()
            cpu = proc.cpu_percent(interval=None) / ncores
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            rss = proc.memory_info().rss / 1e9
            self.cpu_percent = cpu
            self.mem_gb = rss
            self._cpu_per_core_window.append((now, list(per_core)))
            self._cpu_total_window.append((now, cpu))
            self._mem_gb_window.append((now, rss))
            # Mixer state sample for distribution rollup
            self._mixer_state_window.append((now, self.mixer_state))
            # Per-10s legacy summary line (kept for human grep convenience)
            log_tick += 1
            if log_tick >= 10:
                log_tick = 0
                log.info("[health] cpu=%.0f%% mem=%.2fGB unique_hashes=%d "
                         "described=%d dedup=%d",
                         cpu, rss, len(self.unique_hashes),
                         self._stats["described_total"], self._stats["dedup_skipped"])

    async def _segment_pruner(self) -> None:
        """ffmpeg's `hls_flags delete_segments` does the actual pruning. This
        task reports counts + disk usage to /webrtc/status, and on every poll
        logs the inter-segment delta (target ≈4 s) — a proxy for HLS write
        latency, since we can't directly measure frame→disk time."""
        # Tighter polling so we catch every new segment within ~1 s
        while self._running:
            await asyncio.sleep(1.0)
            try:
                segs = sorted(AUDIO_SEG_DIR.glob("seg_*.m4s"))
            except OSError:
                continue
            for p in segs:
                try:
                    idx = int(p.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if idx > self._last_seg_seen_idx:
                    now = time.time()
                    delta_ms = (now - self._last_seg_seen_ts) * 1000.0 \
                        if self._last_seg_seen_ts > 0 else 0.0
                    if self._last_seg_seen_ts > 0:
                        target_ms = AUDIO_SEG_DURATION_S * 1000
                        is_late = delta_ms > target_ms * 1.5
                        log.info(
                            "[hls] seg=%d since_prev=%.0fms target=%dms %s",
                            idx, delta_ms, target_ms,
                            "LATE" if is_late else "OK")
                        if is_late:
                            self._late_segments_total += 1
                            log.warning(
                                "[hls] seg=%d LATE — encoder may have "
                                "stalled (delta=%.0fms)", idx, delta_ms)
                    diag_events.record("encoder", "hls_segment_written",
                                       seg_index=idx,
                                       since_last_seg_ms=round(delta_ms, 1))
                    self._stats["audio_segments_written"] += 1
                    self._last_seg_seen_idx = idx
                    self._last_seg_seen_ts = now
            try:
                files = (list(AUDIO_SEG_DIR.glob("*.m4s"))
                         + list(AUDIO_SEG_DIR.glob("*.mp4")))
                total = sum(p.stat().st_size for p in files)
                self._stats["audio_segments_total_mb"] = round(total / 1e6, 2)
            except OSError:
                pass

    async def _audio_diag_logger(self) -> None:
        """Per-1s and per-5s structured diagnostic lines. See module-level
        docstring at the top of the file (or PHASE 1 spec) for the field list.
        Format is fixed so /tmp/aspire-backend.log lines are greppable.
        """
        prev_buffer_drops = 0
        prev_encoder_dropped = (self._encoder_writer.dropped_total
                                if self._encoder_writer else 0)
        prev_audio_frames_in = self._stats.get("audio_frames_in", 0)
        five_sec_tick = 0
        prev_seg_count = self._stats.get("audio_segments_written", 0)
        while self._running:
            await asyncio.sleep(1.0)
            now = time.time()

            # ── [capture] audio_q_size, audio_fps, drops_in_1s ────────────
            audio_fps = self._ewma_fps(self._audio_frames_window)
            cur_audio_frames = self._stats.get("audio_frames_in", 0)
            # "drops" at capture = frames not in window vs expected at 50fps.
            # We track buffer-cap drops separately; here use audio_q backlog
            # delta as a proxy for capture stalls. Cleaner: just report what
            # we measure — frames in last 1s (should be ~50) vs expected 50.
            frames_in_1s = cur_audio_frames - prev_audio_frames_in
            prev_audio_frames_in = cur_audio_frames
            capture_drops_in_1s = max(0, 50 - frames_in_1s)  # expected 50fps
            log.info(
                "[capture] audio_q_size=%d audio_fps=%.1f frames_in_1s=%d "
                "drops_in_1s=%d",
                self.speaker_q.qsize(), audio_fps, frames_in_1s,
                capture_drops_in_1s)

            # ── [mixer] state, buf_ms, writes_in_1s, mean/max/p99 write_us ─
            cutoff = now - 1.0
            while (self._mixer_write_us_window
                   and self._mixer_write_us_window[0][0] < cutoff):
                self._mixer_write_us_window.popleft()
            xs = [us for _, us in self._mixer_write_us_window]
            if xs:
                xs_sorted = sorted(xs)
                p99 = xs_sorted[max(0, int(len(xs_sorted) * 0.99) - 1)]
                log.info(
                    "[mixer] state=%s buf_ms=%d writes_in_1s=%d "
                    "mean_write_us=%.0f max_write_us=%.0f p99_write_us=%.0f",
                    self.mixer_state, self.speaker_buffer_ms, len(xs),
                    sum(xs) / len(xs), max(xs), p99)
            else:
                log.info(
                    "[mixer] state=%s buf_ms=%d writes_in_1s=0 "
                    "mean_write_us=0 max_write_us=0 p99_write_us=0",
                    self.mixer_state, self.speaker_buffer_ms)

            # ── [encoder] q_size, stdin_blocking_ms, dropped_in_1s ────────
            w = self._encoder_writer
            if w is not None:
                block_ms = w.block_ms_in_window(1.0)
                dropped_in_1s = w.dropped_total - prev_encoder_dropped
                prev_encoder_dropped = w.dropped_total
                log.info(
                    "[encoder] q_size=%d/%d stdin_blocking_ms_total_in_1s=%.1f "
                    "dropped_in_1s=%d dropped_total=%d respawns_total=%d",
                    w.q.qsize(), _EncoderWriter.QUEUE_MAX, block_ms,
                    dropped_in_1s, w.dropped_total, w.respawns_total)

            # Also keep the legacy buffer-cap drop line (it's distinct from
            # encoder-writer drops — this measures speaker_buffer overflow).
            buffer_drops_in_1s = (
                self._buffer_drop_frames_total - prev_buffer_drops)
            prev_buffer_drops = self._buffer_drop_frames_total
            if buffer_drops_in_1s > 0:
                log.info(
                    "[buffer] buffer_drop_frames_in_1s=%d total=%d "
                    "speaker_buffer_ms=%d state=%s",
                    buffer_drops_in_1s, self._buffer_drop_frames_total,
                    self.speaker_buffer_ms, self.mixer_state)

            # ── Per-5s health, encoder, tts, vision ───────────────────────
            five_sec_tick += 1
            if five_sec_tick >= 5:
                five_sec_tick = 0
                proc = self._ff_encoder
                alive = proc is not None and proc.poll() is None
                cur_seg_count = self._stats.get("audio_segments_written", 0)
                segs_in_5s = cur_seg_count - prev_seg_count
                prev_seg_count = cur_seg_count

                # wallclock_drift_ms — how far the encoder output is behind
                # wallclock. Each segment is AUDIO_SEG_DURATION_S long; we
                # expect (elapsed / dur) segments by now. Lag in ms.
                started_at = self._stats.get("started_at") or now
                elapsed = max(0.0, now - started_at)
                expected_segs = elapsed / AUDIO_SEG_DURATION_S
                drift_segs = expected_segs - cur_seg_count
                drift_ms = drift_segs * AUDIO_SEG_DURATION_S * 1000.0
                log.info(
                    "[encoder] alive=%s segments_written_in_5s=%d expected≈1 "
                    "total_segs=%d wallclock_drift_ms=%.0f",
                    "T" if alive else "F", segs_in_5s, cur_seg_count, drift_ms)

                # [health] expanded — per-core CPU + python threads/tasks
                try:
                    import psutil
                    py_threads = psutil.Process().num_threads()
                except Exception:
                    py_threads = -1
                try:
                    py_tasks = len([t for t in asyncio.all_tasks()
                                    if not t.done()])
                except Exception:
                    py_tasks = -1
                last_per_core = (self._cpu_per_core_window[-1][1]
                                 if self._cpu_per_core_window else [])
                core_repr = "[" + ",".join(
                    "%.0f" % c for c in last_per_core) + "]"
                log.info(
                    "[health] cpu_total=%.0f%% per_core=%s mem_gb=%.2f "
                    "python_threads=%d python_tasks=%d",
                    self.cpu_percent, core_repr, self.mem_gb,
                    py_threads, py_tasks)

                # [tts] queue_depth + p50 + in_progress
                tts_p50 = (float(np.median(self._tts_times)) * 1000.0
                           if self._tts_times else 0.0)
                log.info(
                    "[tts] queue_depth=%d synthesize_p50=%.0fms in_progress=%s",
                    self.tts_q.qsize(), tts_p50,
                    "T" if self.tts_in_progress else "F")

                # [vision] queue_depth + p50 + in_progress. There is no
                # dedicated vision queue — Haiku runs ad-hoc per commit.
                # Use a proxy: 1 if in-flight, 0 otherwise.
                vision_p50 = (float(np.median(self._vision_times)) * 1000.0
                              if self._vision_times else 0.0)
                vision_qd = 1 if self.vision_in_progress else 0
                log.info(
                    "[vision] queue_depth=%d haiku_p50=%.0fms in_progress=%s",
                    vision_qd, vision_p50,
                    "T" if self.vision_in_progress else "F")

    async def _summary_logger(self) -> None:
        """Per-30s rollup lines for at-a-glance diagnosis. Three lines per
        tick — encoder/mixer-write/segment counters, mixer state distribution,
        and CPU/mem peaks."""
        prev_encoder_dropped = (self._encoder_writer.dropped_total
                                if self._encoder_writer else 0)
        prev_encoder_respawns = (self._encoder_writer.respawns_total
                                 if self._encoder_writer else 0)
        prev_buffer_drops = self._buffer_drop_frames_total
        prev_late_segs = self._late_segments_total
        while self._running:
            await asyncio.sleep(30.0)
            now = time.time()
            w = self._encoder_writer

            cur_drop = w.dropped_total if w else 0
            cur_resp = w.respawns_total if w else 0
            d_drop = cur_drop - prev_encoder_dropped
            d_resp = cur_resp - prev_encoder_respawns
            d_buf = self._buffer_drop_frames_total - prev_buffer_drops
            d_late = self._late_segments_total - prev_late_segs
            prev_encoder_dropped = cur_drop
            prev_encoder_respawns = cur_resp
            prev_buffer_drops = self._buffer_drop_frames_total
            prev_late_segs = self._late_segments_total

            # Max + p99 mixer write_us over last 30s
            cutoff = now - 30.0
            while (self._mixer_write_us_window
                   and self._mixer_write_us_window[0][0] < cutoff):
                self._mixer_write_us_window.popleft()
            xs = sorted(us for _, us in self._mixer_write_us_window)
            if xs:
                max_us = xs[-1]
                p99 = xs[max(0, int(len(xs) * 0.99) - 1)]
            else:
                max_us = p99 = 0.0
            log.info(
                "[SUMMARY 30s] encoder_dropped=%d encoder_respawns=%d "
                "max_write_us=%.0f p99_write_us=%.0f late_segments=%d "
                "audio_drops=%d",
                d_drop, d_resp, max_us, p99, d_late, d_buf)

            # Mixer state distribution
            states = [s for ts, s in self._mixer_state_window
                      if ts >= cutoff]
            if states:
                total = len(states)
                from collections import Counter
                c = Counter(states)
                pct = {k: round(v * 100.0 / total, 1) for k, v in c.items()}
                log.info(
                    "[SUMMARY 30s] mixer_state_distribution=%s", pct)
            else:
                log.info("[SUMMARY 30s] mixer_state_distribution={}")

            # CPU p99 + mem peak over the 30s window
            cpu_xs = sorted(c for ts, c in self._cpu_total_window
                            if ts >= cutoff)
            mem_xs = [m for ts, m in self._mem_gb_window if ts >= cutoff]
            if cpu_xs:
                cpu_p99 = cpu_xs[max(0, int(len(cpu_xs) * 0.99) - 1)]
            else:
                cpu_p99 = 0.0
            mem_peak = max(mem_xs) if mem_xs else 0.0
            log.info(
                "[SUMMARY 30s] cpu_p99=%.0f%% mem_peak_gb=%.2f",
                cpu_p99, mem_peak)

    async def _ring_logger(self) -> None:
        """Log ring producer rate + max peer lag once per second."""
        while self._running:
            await asyncio.sleep(1.0)
            writes = self.ring.writes_in_window(1.0)
            self._ring_writes_per_sec_window.append(writes)
            head = self.ring.head()
            peer_count = 0
            lag_frames = 0
            try:
                from app import _peer_sessions
                peer_count = len(_peer_sessions)
                for s in _peer_sessions.values():
                    if s.playhead >= 0:
                        lag_frames = max(lag_frames, head - s.playhead)
            except Exception:
                pass
            lag_ms = lag_frames * FRAME_MS
            log.info("[ring] head=%d writes_in_1s=%d expected=50 "
                     "oldest_peer_lag_ms=%d peer_count=%d",
                     head, writes, lag_ms, peer_count)

    async def _fps_logger(self) -> None:
        while self._running:
            await asyncio.sleep(1.0)
            vfps = self._ewma_fps(self._video_frames_window)
            afps = self._ewma_fps(self._audio_frames_window)
            v_alive = self._ff_video is not None and self._ff_video.poll() is None
            a_alive = self._ff_audio is not None and self._ff_audio.poll() is None
            log.info("[capture] video_fps=%.2f audio_fps=%.2f V=%s A=%s "
                     "speaker_q=%d mixed_q=%d buf=%dms state=%s",
                     vfps, afps, "✓" if v_alive else "✗", "✓" if a_alive else "✗",
                     self.speaker_q.qsize(), self.mixed_q.qsize(),
                     self.speaker_buffer_ms, self.mixer_state)

    # ── slide detection (old proven pHash-only logic) ────────────────────
    @staticmethod
    def _phash_for(frame_bgr: np.ndarray):
        h, w, _ = frame_bgr.shape
        cropped = frame_bgr[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]
        return imagehash.phash(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))

    def _clear_candidate(self) -> None:
        self._candidate_phash = None
        self._candidate_first_seen_ts = None
        self._candidate_frame_bgr = None

    def _rollback_unique_hash_for(self, slide_no: int) -> None:
        """When Haiku returns empty, drop the unique_hash entry that was
        added at COMMIT time so the same slide can be retried on its next
        appearance. Also undo the slide_counter so numbering stays dense."""
        if self.unique_hashes:
            try:
                self.unique_hashes.pop()
            except IndexError:
                pass
        # Roll back the counter only if it matches what we'd just bumped
        if self.slide_counter == slide_no:
            self.slide_counter -= 1
        self._stats["candidates_committed"] = max(
            0, self._stats.get("candidates_committed", 0) - 1)
        self._stats["described_total"] = max(
            0, self._stats.get("described_total", 0) - 1)

    @staticmethod
    def _extract_topic(desc: str) -> str:
        m = re.match(r"^\s*slide\s+\d+\s*[.:]\s*", desc, flags=re.IGNORECASE)
        body = desc[m.end():] if m else desc
        first = body.split(".")[0].strip()
        return first[:60]

    async def _detect_loop(self) -> None:
        """pHash-only detector matching the user's old proven approach.

        Per sampled frame (every SAMPLE_INTERVAL_S):
          1. Centre-crop, compute pHash.
          2. Compare to prev_phash; if Hamming > CHANGE_HAMMING, increment
             changed_run, else reset.
          3. If changed_run >= SUSTAIN_FRAMES: install/refresh candidate.
          4. If candidate held STABILITY_WINDOW_S: dedup against
             unique_hashes (Hamming ≤ DEDUP_HAMMING = same), and if novel,
             COMMIT — run OCR ONCE on the committed frame, kick Haiku+TTS.
        """
        from vision_haiku import describe_slide
        from tts_kokoro import synth as kokoro_synth
        from vision_ocr import extract_text_simple

        last_run = 0.0

        async def _describe_and_speak(frame_bgr, slide_no: int,
                                      ocr_text_str: str,
                                      commit_time: float,
                                      anchor_pos: int) -> None:
            recent_topics = list(self.recent_topics)
            try:
                jpeg = await asyncio.to_thread(_frame_to_jpeg, frame_bgr)
            except Exception as e:
                log.warning("[detect] jpeg encode failed: %s", e)
                return
            self.vision_in_progress = True
            diag_events.record("vision", "haiku_request_start",
                               slide_no=slide_no)
            try:
                t0 = time.time()
                desc = await describe_slide(jpeg, slide_number=slide_no,
                                            ocr_text=ocr_text_str,
                                            recent_topics=recent_topics)
                haiku_dt = time.time() - t0
                self._vision_times.append(haiku_dt)
                self._stats["vision_calls"] += 1
                diag_events.record("vision", "haiku_request_end",
                                   slide_no=slide_no,
                                   took_ms=int(haiku_dt * 1000),
                                   response_len=len(desc or ""))
            except Exception as e:
                log.warning("[vision] failed: %s", e)
                diag_events.record("vision", "haiku_request_end",
                                   slide_no=slide_no,
                                   took_ms=int((time.time() - t0) * 1000),
                                   response_len=0,
                                   error=str(e)[:120])
                self._rollback_unique_hash_for(slide_no)
                return
            finally:
                self.vision_in_progress = False
            if not desc:
                self._stats["vision_empty_responses"] = (
                    self._stats.get("vision_empty_responses", 0) + 1)
                log.warning("[vision] empty response after retry, slide#%d skipped, dedup rolled back",
                            slide_no)
                self._rollback_unique_hash_for(slide_no)
                return

            # Streaming Kokoro: yield per-sentence chunks. First chunk arrives
            # in ~300-500ms; subsequent chunks synthesize in parallel.
            from tts_kokoro import synth_stream
            chunk_idx = 0
            tts_t0 = time.time()
            self.tts_in_progress = True
            diag_events.record("tts", "kokoro_synth_start",
                               slide_no=slide_no)
            try:
                chunk_start = time.time()
                async for samples, sr in synth_stream(desc):
                    chunk_idx += 1
                    chunk_dt = time.time() - chunk_start
                    is_first = (chunk_idx == 1)
                    if is_first:
                        first_chunk_dt = time.time() - tts_t0
                        self._tts_times.append(first_chunk_dt)
                    stereo_s16 = _resample_to_48k_stereo_int16(samples, sr)
                    audio_dur_s = stereo_s16.shape[0] / SAMPLE_RATE
                    diag_events.record("tts", "kokoro_synth_end",
                                       slide_no=slide_no,
                                       chunk_idx=chunk_idx,
                                       took_ms=int(chunk_dt * 1000),
                                       audio_duration_s=round(audio_dur_s, 3))
                    clip = TTSClip(
                        samples_int16_stereo=stereo_s16,
                        text=desc if is_first else "",
                        slide_no=slide_no,
                        # Only the FIRST chunk carries commit-time so the
                        # commit→audible metric measures the right thing
                        commit_time=commit_time if is_first else 0.0,
                        haiku_ms=haiku_dt * 1000.0 if is_first else 0.0,
                        tts_synth_ms=(first_chunk_dt * 1000.0
                                       if is_first else 0.0),
                        # All chunks for the same slide share the anchor —
                        # they should all wait for the listener to reach the
                        # commit-time speaker position before playing.
                        anchor_pos=anchor_pos,
                    )
                    if self.tts_q.full():
                        try:
                            dropped = self.tts_q.get_nowait()
                            self._stats["tts_dropped"] += 1
                            self._stats["tts_queue_drops"] += 1
                            log.warning("[tts] queue full, dropped %r",
                                        dropped.text[:40])
                        except asyncio.QueueEmpty:
                            pass
                    await self.tts_q.put(clip)
                    read_pos_at_queue = self._mixer_speaker_pos
                    diag_events.record(
                        "queue", "tts_q_put",
                        slide_no=slide_no,
                        clip_duration_s=round(audio_dur_s, 3),
                        q_size_after=self.tts_q.qsize(),
                        anchor_pos=anchor_pos,
                        mixer_read_pos_at_queue=read_pos_at_queue,
                        anchor_gap_frames=max(0, anchor_pos - read_pos_at_queue),
                    )
                    chunk_start = time.time()
            except Exception as e:
                log.warning("[tts] stream failed at chunk %d: %s",
                            chunk_idx, e)
                diag_events.record("tts", "kokoro_synth_end",
                                   slide_no=slide_no,
                                   chunk_idx=chunk_idx,
                                   error=str(e)[:120])
                return
            finally:
                self.tts_in_progress = False

            self.recent_descriptions.append((time.time(), desc))
            self.last_ocr_for_speaking = ocr_text_str or ""
            topic = self._extract_topic(desc)
            if topic:
                self.recent_topics.append(topic)

        async def _try_commit(now: float) -> None:
            if (self._candidate_phash is None
                    or self._candidate_first_seen_ts is None
                    or self._candidate_frame_bgr is None):
                return
            if (now - self._candidate_first_seen_ts) < STABILITY_WINDOW_S:
                return

            cand_ph = self._candidate_phash
            cand_frame = self._candidate_frame_bgr

            # Dedup: min Hamming distance to any committed pHash
            min_d = None
            for uh in self.unique_hashes:
                d = uh - cand_ph
                if min_d is None or d < min_d:
                    min_d = d
            self._last_dedup_dists.appendleft(min_d if min_d is not None else -1)

            decision = "COMMITTED"
            if min_d is not None and min_d <= DEDUP_HAMMING:
                self._stats["dedup_skipped"] += 1
                decision = "SKIPPED_DEDUP"
                log.info("[detect] commit-check phash=%s min_dist_to_unique=%d "
                         "threshold=%d → %s",
                         str(cand_ph)[:8], min_d, DEDUP_HAMMING, decision)
                self._clear_candidate()
                return

            log.info("[detect] commit-check phash=%s min_dist_to_unique=%s "
                     "threshold=%d → %s",
                     str(cand_ph)[:8],
                     "n/a" if min_d is None else str(min_d),
                     DEDUP_HAMMING, decision)

            # COMMIT
            self.unique_hashes.append(cand_ph)
            self.slide_counter += 1
            slide_no = self.slide_counter
            commit_time = time.time()
            # Snapshot the speaker write position at commit time. This is the
            # "anchor" the listener has to reach before this slide's TTS may
            # play. audio_frames_in is the running capture-frame counter in
            # the same units as _mixer_speaker_pos (20ms s16 stereo frames).
            anchor_pos = int(self._stats.get("audio_frames_in", 0))
            self._stats["candidates_committed"] += 1
            self._stats["described_total"] += 1
            self._stats["events"] += 1
            cand_age = (commit_time - self._candidate_first_seen_ts
                        if self._candidate_first_seen_ts else 0.0)
            diag_events.record("detect", "candidate_committed",
                               slide_no=slide_no,
                               phash_excerpt=str(cand_ph)[:8],
                               candidate_age_s=round(cand_age, 2),
                               anchor_pos=anchor_pos,
                               mixer_read_pos_at_commit=self._mixer_speaker_pos)
            self._clear_candidate()

            # Single OCR call here, post-commit, on the winning frame.
            try:
                ocr_text_str = await asyncio.to_thread(
                    extract_text_simple, cand_frame
                )
            except Exception as e:
                log.warning("[ocr] post-commit error: %s", e)
                ocr_text_str = ""

            log.info("[detect] COMMIT slide #%d, OCR=%r anchor=%d",
                     slide_no, (ocr_text_str or "")[:60], anchor_pos)

            asyncio.create_task(
                _describe_and_speak(cand_frame, slide_no, ocr_text_str,
                                    commit_time, anchor_pos)
            )

        # ── per-frame loop ────────────────────────────────────────────────
        while self._running:
            try:
                frame = await asyncio.wait_for(self.video_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                await _try_commit(time.time())
                continue

            now = time.time()
            if now - last_run < SAMPLE_INTERVAL_S * 0.9:
                continue
            last_run = now

            # pHash off the event loop (cheap, but cv2/PIL is sync)
            try:
                ph = await asyncio.to_thread(self._phash_for, frame)
            except Exception as e:
                log.warning("[detect] phash error: %s", e)
                continue

            self._stats["candidates_seen"] += 1
            self._frames_sampled_window.append(now)

            if self._prev_phash is None:
                self._prev_phash = ph
                diag_events.record("detect", "frame_sampled",
                                   phash_excerpt=str(ph)[:8],
                                   dist_to_prev=-1,
                                   sustain_run=0)
                continue

            dist = self._prev_phash - ph
            self._last_phash_dists.appendleft(dist)
            if dist > CHANGE_HAMMING:
                self._changed_run += 1
            else:
                self._changed_run = 0
                # Log "change below threshold" but throttle to once / 5 s
                # (only logs when there WAS a non-trivial frame-to-frame
                # change but it didn't cross the line).
                if dist >= max(2, CHANGE_HAMMING - 6) and (
                    now - self._last_change_below_log_ts >= 5.0
                ):
                    log.info("[detect] change-below-threshold dist=%d "
                             "threshold=%d (slide may be too similar to prior)",
                             dist, CHANGE_HAMMING)
                    self._last_change_below_log_ts = now

            self._changed_run_window.append((now, self._changed_run))

            diag_events.record("detect", "frame_sampled",
                               phash_excerpt=str(ph)[:8],
                               dist_to_prev=int(dist),
                               sustain_run=int(self._changed_run))

            if self._changed_run >= SUSTAIN_FRAMES:
                # Confirmed visual change — install or refresh candidate
                if (self._candidate_phash is None
                        or (self._candidate_phash - ph) > CHANGE_HAMMING):
                    if self._candidate_phash is not None:
                        self._candidate_replaced_count += 1
                        diag_events.record("detect", "candidate_replaced",
                                           old_phash=str(self._candidate_phash)[:8],
                                           new_phash=str(ph)[:8],
                                           reason="superseded_by_change")
                    self._candidate_phash = ph
                    self._candidate_first_seen_ts = now
                    self._candidate_frame_bgr = frame
                    log.info("[detect] new candidate phash=%s, holding 5s",
                             str(ph)[:8])
                    diag_events.record("detect", "candidate_set",
                                       phash_excerpt=str(ph)[:8])
                else:
                    # Same candidate — refresh frame for downstream JPEG/OCR
                    self._candidate_frame_bgr = frame
                self._changed_run = 0
                self._prev_phash = ph
            else:
                self._prev_phash = ph

            # Once-per-second live state log
            if now - self._last_detect_state_log_ts >= 1.0:
                self._last_detect_state_log_ts = now
                cand_age = (now - self._candidate_first_seen_ts
                            if self._candidate_first_seen_ts else 0.0)
                log.info("[detect] dist_to_prev=%d changed_run=%d cand=%s "
                         "cand_age=%.1fs seen=%d committed=%d dedup_skipped=%d",
                         dist, self._changed_run,
                         str(self._candidate_phash)[:8] if self._candidate_phash else "none",
                         cand_age,
                         self._stats["candidates_seen"],
                         self._stats["candidates_committed"],
                         self._stats["dedup_skipped"])

            await _try_commit(now)

    # ── mixer (pause-buffer-catchup state machine) ─────────────────────────
    async def _mixer_loop(self) -> None:
        speaker_buffer: Deque[bytes] = deque()
        active_tts: Optional[np.ndarray] = None
        active_text: str = ""
        active_slide: int = 0
        skip_next_silent = False
        prev_state: str = self.mixer_state  # for diag state-change detection
        # TTS pause-coupling locals:
        #   pending_clip is a TTSClip pulled from the queue but not yet
        #   promoted to active_tts because we're still waiting for a natural
        #   pause in the listener-perceived audio.
        #   last_audible_ts tracks the wall-clock time of the most recent
        #   non-silent frame we EMITTED (so it reflects what the listener
        #   actually heard, including CATCHUP playback of buffered audio).
        pending_clip: Optional[TTSClip] = None
        last_audible_ts: float = time.time()  # init "just-active"; prevents firing in the first 300ms
        # Wall-clock when _mixer_speaker_pos first reached pending_clip.anchor_pos.
        # None means the anchor hasn't been reached yet for the current pending
        # clip. The 20s post-anchor pause-search timer is computed from this,
        # NOT from clip.enqueued_at — so a slide whose anchor is 30s in the
        # future correctly waits 30s + (pause search) before playing.
        anchor_reached_ts: Optional[float] = None

        # Wallclock pacing — produce 1 frame per FRAME_MS so PTS stays aligned
        # with real time even if speaker_q is sometimes empty.
        next_emit = time.time()

        async def get_speaker_frame_nowait() -> Optional[bytes]:
            q_size_before = self.speaker_q.qsize()
            try:
                buf = self.speaker_q.get_nowait()
                diag_events.record("queue", "speaker_q_get_attempt",
                                   q_size_before=q_size_before, got=True)
                return buf
            except asyncio.QueueEmpty:
                diag_events.record("queue", "speaker_q_get_attempt",
                                   q_size_before=q_size_before, got=False)
                return None

        def push_mixed(buf: bytes) -> None:
            # Append to the shared ring (per-peer playheads read from it).
            t0 = time.perf_counter()
            self.ring.append(buf)
            # Hand to the encoder writer thread — non-blocking submit.
            self._encoder_write(buf)
            # Also keep mixed_q populated for legacy / fallback consumers
            # (it's tiny and self-draining).
            if self.mixed_q.full():
                try:
                    self.mixed_q.get_nowait()
                    self._stats["mixed_overflow_drops"] += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                self.mixed_q.put_nowait(buf)
            except asyncio.QueueFull:
                pass
            self._mixer_write_us_window.append(
                (time.time(), (time.perf_counter() - t0) * 1e6))

        silence_buf = b"\x00" * BYTES_PER_FRAME

        while self._running:
            tick_t0 = time.perf_counter()
            # Pace the loop to 50 fps wall-clock
            sleep_for = next_emit - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            next_emit += FRAME_MS / 1000.0
            # If we fell way behind (e.g. coroutine starved), reset cadence
            if time.time() - next_emit > 0.5:
                next_emit = time.time() + FRAME_MS / 1000.0

            # Diag: state-change detection (top-of-iteration so we always
            # observe the state used for THIS tick's decisions)
            if self.mixer_state != prev_state:
                diag_events.record("mixer", "mixer_state_change",
                                   from_state=prev_state,
                                   to_state=self.mixer_state,
                                   reason="loop_top",
                                   buf_ms=self.speaker_buffer_ms)
                prev_state = self.mixer_state

            tick_source = "none"  # set below depending on which branch emits

            # ── Pause-coupled TTS pickup ──────────────────────────────────
            # Stage 1: move a clip from the asyncio queue into pending_clip
            #   so we can hold it while waiting for a natural pause.
            if active_tts is None and pending_clip is None and not self.tts_q.empty():
                try:
                    pending_clip = self.tts_q.get_nowait()
                    diag_events.record("queue", "tts_q_get",
                                       slide_no=pending_clip.slide_no,
                                       q_size_after=self.tts_q.qsize())
                    diag_events.record("mixer", "tts_pending_pause_start",
                                       slide_no=pending_clip.slide_no,
                                       enqueued_at=pending_clip.enqueued_at)
                except asyncio.QueueEmpty:
                    pending_clip = None
            # Stage 2: promote pending_clip to active_tts.
            # Gate order:
            #   1. ANCHOR — _mixer_speaker_pos must reach the clip's anchor_pos
            #      (i.e. the listener has heard the speaker audio surrounding
            #      the moment the slide was committed). Until this is true, do
            #      not even consider the pause/timeout logic.
            #   2. PAUSE — once anchor is reached, wait for a natural silence
            #      ≥ TTS_PAUSE_TRIGGER_S in the emitted audio, or fall back to
            #      TTS_HOLD_MAX_S timeout counted from anchor-reached.
            #   3. SAFETY NET — if a clip has been pending > TTS_ANCHOR_GIVEUP_S
            #      (wall-clock since enqueue) AND its anchor is still not
            #      reached, force-promote with reason="anchor_giveup".
            if active_tts is None and pending_clip is not None:
                now_t = time.time()
                anchor_pos = pending_clip.anchor_pos
                read_pos = self._mixer_speaker_pos
                anchor_reached = read_pos >= anchor_pos
                wait_since_enqueue = now_t - pending_clip.enqueued_at

                # First-time anchor reached transition: log + diag event.
                if anchor_reached and anchor_reached_ts is None:
                    anchor_reached_ts = now_t
                    anchor_wait_ms = int(wait_since_enqueue * 1000)
                    diag_events.record(
                        "mixer", "slide_anchor_reached",
                        slide_no=pending_clip.slide_no,
                        anchor_pos=anchor_pos,
                        read_pos_at_reach=read_pos,
                        anchor_wait_ms=anchor_wait_ms,
                    )
                    log.info(
                        "[mixer] slide#%d anchor reached "
                        "(anchor=%d read=%d wait=%dms)",
                        pending_clip.slide_no, anchor_pos, read_pos,
                        anchor_wait_ms)

                # Safety net: anchor never reached after TTS_ANCHOR_GIVEUP_S.
                # Force-promote so the queue can't deadlock.
                forced_giveup = (
                    not anchor_reached
                    and wait_since_enqueue >= TTS_ANCHOR_GIVEUP_S
                )

                # Eligibility:
                promote = False
                trigger = None
                pause_wait_ms = 0
                anchor_wait_ms_final = 0
                silence_dur = now_t - last_audible_ts

                if forced_giveup:
                    promote = True
                    trigger = "anchor_giveup"
                    gap_frames = max(0, anchor_pos - read_pos)
                    diag_events.record(
                        "mixer", "slide_anchor_giveup",
                        slide_no=pending_clip.slide_no,
                        anchor_pos=anchor_pos,
                        mixer_read_pos=read_pos,
                        gap_frames=gap_frames,
                        total_wait_ms=int(wait_since_enqueue * 1000),
                    )
                    log.warning(
                        "[mixer] slide#%d ANCHOR GIVEUP — anchor=%d read=%d "
                        "gap=%d frames wait=%.1fs (force-promoting)",
                        pending_clip.slide_no, anchor_pos, read_pos,
                        gap_frames, wait_since_enqueue)
                elif anchor_reached:
                    # anchor_reached_ts is guaranteed non-None here.
                    post_anchor_dur = now_t - anchor_reached_ts
                    if silence_dur >= TTS_PAUSE_TRIGGER_S:
                        promote = True
                        trigger = "natural_pause"
                    elif post_anchor_dur >= TTS_HOLD_MAX_S:
                        promote = True
                        trigger = "forced_timeout"
                    pause_wait_ms = int(post_anchor_dur * 1000)
                    anchor_wait_ms_final = int(
                        (anchor_reached_ts - pending_clip.enqueued_at) * 1000
                    )

                if promote:
                    diag_events.record(
                        "mixer", "slide_promotion",
                        slide_no=pending_clip.slide_no,
                        trigger=trigger,
                        anchor_wait_ms=anchor_wait_ms_final,
                        pause_wait_ms=pause_wait_ms,
                    )
                    # Keep the old event around for back-compat with existing
                    # analyzer code that counts tts_promoted_to_active.
                    diag_events.record(
                        "mixer", "tts_promoted_to_active",
                        slide_no=pending_clip.slide_no,
                        hold_dur_s=round(wait_since_enqueue, 3),
                        silence_dur_s=round(silence_dur, 3),
                        trigger=trigger,
                    )
                    log.info(
                        "[mixer] slide#%d promoted: trigger=%s "
                        "anchor_wait=%dms pause_wait=%dms",
                        pending_clip.slide_no, trigger,
                        anchor_wait_ms_final, pause_wait_ms)
                    clip = pending_clip
                    pending_clip = None
                    anchor_reached_ts = None
                    active_tts = clip.samples_int16_stereo
                    active_slide = clip.slide_no
                    active_total_samples = active_tts.shape[0]
                    active_played_samples = 0
                    self._stats["tts_played"] += 1
                    audible_time = time.time()
                    # First chunk of a new slide carries text + commit_time;
                    # subsequent streaming chunks have text="" and reuse the
                    # current slide's caption / state.
                    is_first_chunk = bool(clip.text)
                    if clip.commit_time:
                        c2a_ms = (audible_time - clip.commit_time) * 1000
                        queue_wait_ms = max(0.0,
                                            c2a_ms - clip.haiku_ms - clip.tts_synth_ms)
                        self._commit_to_audible_ms.append(c2a_ms)
                        log.info("[latency] slide#%d commit→audible=%.0fms "
                                 "(haiku=%.0fms tts=%.0fms queue_wait=%.0fms)",
                                 active_slide, c2a_ms, clip.haiku_ms,
                                 clip.tts_synth_ms, queue_wait_ms)
                    log.info("[mixer] start clip slide#%d duration=%.2fs queue_depth=%d%s",
                             active_slide,
                             active_tts.shape[0] / SAMPLE_RATE,
                             self.tts_q.qsize(),
                             "" if is_first_chunk else " (cont.)")
                    self.mixer_state = "PAUSED_FOR_TTS"
                    if is_first_chunk:
                        active_text = clip.text
                        self.currently_speaking_text = active_text
                        self.currently_speaking_slide = active_slide
                        self.currently_speaking_started_at = audible_time

            # ── State: PAUSED_FOR_TTS ─────────────────────────────────────
            if active_tts is not None:
                # Drain any speaker frame that arrived this tick into the buffer
                spk = await get_speaker_frame_nowait()
                if spk is not None:
                    speaker_buffer.append(spk)

                # Cap buffer at 30s
                if len(speaker_buffer) > MAX_BUFFER_FRAMES:
                    drop = len(speaker_buffer) // 2
                    for _ in range(drop):
                        speaker_buffer.popleft()
                    self._stats["buffer_overflows"] += 1
                    self._buffer_drop_frames_total += drop
                    # Advance read_pos so anchors that fell inside the dropped
                    # range still get reached. Treat the drops as "consumed
                    # but not heard" — the listener experiences a jump, which
                    # is what the cap is designed for. (Fix 2-a per spec.)
                    self._mixer_speaker_pos += drop
                    log.info("[buffer] dropped %d frames due to overflow, "
                             "advancing read_pos to %d",
                             drop, self._mixer_speaker_pos)
                    log.warning("[mixer] speaker buffer >30s — dropped oldest %d frames", drop)

                # Emit one TTS frame
                take = min(SAMPLES_PER_FRAME, active_tts.shape[0])
                tts_slice = active_tts[:take]
                active_tts = active_tts[take:]
                active_played_samples += take
                if take < SAMPLES_PER_FRAME:
                    pad = np.zeros((SAMPLES_PER_FRAME - take, CHANNELS), dtype=np.int16)
                    tts_slice = np.concatenate([tts_slice, pad], axis=0)
                push_mixed(tts_slice.tobytes())
                tick_source = "tts"

                if active_tts.shape[0] == 0:
                    played_s = active_played_samples / SAMPLE_RATE
                    total_s = active_total_samples / SAMPLE_RATE
                    interrupted = active_played_samples < active_total_samples
                    if interrupted:
                        self._stats["tts_clips_interrupted"] += 1
                    else:
                        self._stats["tts_clips_played_complete"] += 1
                    log.info("[mixer] %s clip slide#%d played=%.2fs/%.2fs reason=%s",
                             "INTERRUPTED" if interrupted else "finish",
                             active_slide, played_s, total_s,
                             "preempted" if interrupted else "natural_end")

                    # If the next queued clip belongs to the same slide
                    # (streaming continuation), DON'T transition to CATCHUP
                    # — the next loop iteration will pick it up and keep
                    # PAUSED_FOR_TTS.
                    next_clip_same_slide = False
                    try:
                        peek = self.tts_q._queue[0] if self.tts_q.qsize() else None
                        next_clip_same_slide = (
                            peek is not None and peek.slide_no == active_slide
                        )
                    except Exception:
                        pass

                    active_tts = None
                    if next_clip_same_slide:
                        # Stay PAUSED_FOR_TTS; consume the continuation next tick
                        pass
                    else:
                        log.info("[mixer] TTS done. buffer=%dms → %s",
                                 len(speaker_buffer) * FRAME_MS,
                                 "CATCHUP" if speaker_buffer else "LIVE")
                        self.mixer_state = "CATCHUP" if speaker_buffer else "LIVE"
                        self.currently_speaking_text = ""
                        self.currently_speaking_slide = 0
                        self.last_ocr_for_speaking = ""

                self.speaker_buffer_ms = len(speaker_buffer) * FRAME_MS
                diag_events.record("mixer", "mixer_tick",
                                   state=self.mixer_state,
                                   used_source=tick_source,
                                   buf_ms=self.speaker_buffer_ms,
                                   compression_mode=self.compression_mode,
                                   took_us=int((time.perf_counter() - tick_t0) * 1e6))
                continue

            # ── State: CATCHUP ────────────────────────────────────────────
            if speaker_buffer:
                # Drain newly-arrived speaker frames into buffer first
                while True:
                    spk = await get_speaker_frame_nowait()
                    if spk is None:
                        break
                    speaker_buffer.append(spk)
                if len(speaker_buffer) > MAX_BUFFER_FRAMES:
                    drop = len(speaker_buffer) // 2
                    for _ in range(drop):
                        speaker_buffer.popleft()
                    self._stats["buffer_overflows"] += 1
                    self._buffer_drop_frames_total += drop
                    # See PAUSED_FOR_TTS branch — same Fix 2-a rationale.
                    self._mixer_speaker_pos += drop
                    log.info("[buffer] dropped %d frames due to overflow, "
                             "advancing read_pos to %d",
                             drop, self._mixer_speaker_pos)
                    log.warning("[mixer] speaker buffer >30s — dropped oldest %d frames", drop)

                # No silence compression. Pop one buffered frame, advance
                # the read counter, emit it unchanged. This is the path
                # that produced 0 stutter marks in phone tests.
                buf = speaker_buffer.popleft()
                self._mixer_speaker_pos += 1
                push_mixed(buf)
                tick_source = "speaker_buffered"
                # Pause-coupling: update last_audible_ts if what we emitted
                # is above the silence threshold. CATCHUP plays buffered
                # speaker frames; their loudness is the listener-perceived
                # audio level.
                if _frame_peak_int16(buf) >= SILENCE_PEAK:
                    last_audible_ts = time.time()

                self.speaker_buffer_ms = len(speaker_buffer) * FRAME_MS
                # Caught up?
                if not speaker_buffer and self.speaker_q.qsize() <= 2:
                    self.mixer_state = "LIVE"
                    log.info("[mixer] CATCHUP done, back to LIVE")
                diag_events.record("mixer", "mixer_tick",
                                   state="CATCHUP",
                                   used_source=tick_source,
                                   buf_ms=self.speaker_buffer_ms,
                                   compression_mode=self.compression_mode,
                                   took_us=int((time.perf_counter() - tick_t0) * 1e6))
                continue

            # ── State: LIVE ───────────────────────────────────────────────
            self.mixer_state = "LIVE"
            spk = await get_speaker_frame_nowait()
            if spk is None:
                # No speaker audio right now — emit silence so the track never
                # starves and PTS stays continuous. Silence pads do NOT
                # advance _mixer_speaker_pos (nothing from the capture stream
                # actually reached the listener).
                push_mixed(silence_buf)
                tick_source = "silence"
            else:
                push_mixed(spk)
                tick_source = "speaker"
                # Real speaker frame emitted → advance the read-pos counter.
                self._mixer_speaker_pos += 1
                # Pause-coupling: speaker activity ⇒ no pause yet.
                if _frame_peak_int16(spk) >= SILENCE_PEAK:
                    last_audible_ts = time.time()
            diag_events.record("mixer", "mixer_tick",
                               state="LIVE",
                               used_source=tick_source,
                               buf_ms=0,
                               compression_mode=self.compression_mode,
                               took_us=int((time.perf_counter() - tick_t0) * 1e6))


# Module-level singleton
_pipeline: Optional[LivePipeline] = None


def get_pipeline() -> LivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePipeline()
    return _pipeline
