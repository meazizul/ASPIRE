#!/usr/bin/env python3
"""
diag_input_audio.py — standalone diagnostic for the BlackHole → ffmpeg → Python
audio capture path. Mirrors the live pipeline's ffmpeg command EXACTLY, then
tight-loops reading one 20ms s16-stereo frame (3840 bytes) at a time and
records wall-clock inter-read intervals.

Hypotheses tested:
  H4 (avfoundation input buffer too small) → max-interval / gap_gt30ms_count
  H5 (Python read cadence drift)           → p99_interval, max_interval
  H6 (avfoundation PTS jitter)             → mean_interval / drift

If audio data arrives smoothly from BlackHole at 48kHz, each 20ms frame should
land ~20ms apart. Significant deviation (large p99, frequent gaps >30ms) is
direct evidence of input-side jitter.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Must match pipeline.py constants exactly
SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000          # 960
BYTES_PER_FRAME = SAMPLES_PER_FRAME * CHANNELS * 2          # 3840
DURATION_S = 60
AUDIO_DEVICE_IDX = os.environ.get("AUDIO_IDX", "0")          # BlackHole 2ch

LOG_PATH = Path("/tmp/aspire-diag-input.log")


def _ff_cmd() -> list:
    """Identical to pipeline._audio_cmd() — same flags, same filters, same buffer
    behavior. If we capture jitter here, the live pipeline captures the same."""
    return [
        "ffmpeg",
        "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts",
        "-thread_queue_size", "1024",
        "-f", "avfoundation",
        "-i", f":{AUDIO_DEVICE_IDX}",
        "-map", "0:a",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-af", "aresample=async=1000:first_pts=0",
        "-f", "s16le",
        "pipe:1",
    ]


def _read_exact(stdout, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stdout.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _emit(line: str, fh) -> None:
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main() -> int:
    cmd = _ff_cmd()
    print(f"[diag] launching: {' '.join(cmd)}", flush=True)
    print(f"[diag] reading {BYTES_PER_FRAME} bytes/frame, "
          f"expecting one frame every {FRAME_MS}ms for {DURATION_S}s", flush=True)
    print(f"[diag] log → {LOG_PATH}", flush=True)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )

    # Drain ffmpeg stderr to the diag log without blocking reads.
    import threading
    def _drain_stderr():
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not raw:
                    break
                sys.stderr.write("[ffmpeg] " + raw.decode("utf-8", "replace"))
        except Exception:
            pass
    threading.Thread(target=_drain_stderr, daemon=True).start()

    intervals_all: list = []   # ms between consecutive reads
    bytes_total = 0
    started = time.perf_counter()
    next_log = started + 1.0
    second_intervals: list = []
    second_idx = 0
    last_read_ts = None
    fh = LOG_PATH.open("w")

    try:
        while True:
            now = time.perf_counter()
            if now - started >= DURATION_S:
                break
            buf = _read_exact(proc.stdout, BYTES_PER_FRAME)
            if buf is None:
                _emit("[diag ERROR] ffmpeg stdout closed early (likely a "
                      "permission or device-busy error — check stderr above)",
                      fh)
                break
            read_ts = time.perf_counter()
            bytes_total += len(buf)
            if last_read_ts is not None:
                dt_ms = (read_ts - last_read_ts) * 1000.0
                second_intervals.append(dt_ms)
                intervals_all.append(dt_ms)
            last_read_ts = read_ts

            if read_ts >= next_log and second_intervals:
                vs = sorted(second_intervals)
                n = len(vs)
                mean = sum(vs) / n
                p50 = vs[int(n * 0.5)]
                p95 = vs[min(n - 1, int(n * 0.95))]
                p99 = vs[min(n - 1, int(n * 0.99))]
                mx = vs[-1]
                gap_gt30 = sum(1 for v in vs if v > 30.0)
                gap_lt10 = sum(1 for v in vs if v < 10.0)
                second_idx += 1
                _emit(
                    f"[diag t={second_idx}s] reads_in_1s={n} expected=50 "
                    f"mean_interval_ms={mean:.2f} p50={p50:.2f} p95={p95:.2f} "
                    f"p99={p99:.2f} max={mx:.2f} gap_gt30ms={gap_gt30} "
                    f"gap_lt10ms={gap_lt10} total_bytes_read={bytes_total}",
                    fh)
                second_intervals = []
                next_log = read_ts + 1.0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try: proc.kill()
            except Exception: pass

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    if not intervals_all:
        _emit("[diag SUMMARY] no frames captured — check BlackHole is the "
              "default output and that an app is playing audio to it.", fh)
        _emit("[diag VERDICT] NO_DATA", fh)
        fh.close()
        return 2

    vs = sorted(intervals_all)
    n = len(vs)
    mean = sum(vs) / n
    p50 = vs[int(n * 0.5)]
    p95 = vs[min(n - 1, int(n * 0.95))]
    p99 = vs[min(n - 1, int(n * 0.99))]
    mx = vs[-1]
    gap_gt30 = sum(1 for v in vs if v > 30.0)
    gap_gt100 = sum(1 for v in vs if v > 100.0)

    _emit(
        f"[diag SUMMARY] total_reads={n} mean_interval_ms={mean:.2f} "
        f"p50={p50:.2f} p95={p95:.2f} p99_interval_ms={p99:.2f} "
        f"max_interval_ms={mx:.2f} gap_gt30ms_count={gap_gt30} "
        f"gap_gt100ms_count={gap_gt100} total_bytes_read={bytes_total}",
        fh)

    # ── VERDICT ──────────────────────────────────────────────────────────────
    if p99 < 25.0 and gap_gt30 == 0:
        verdict = "CLEAN"
    elif (25.0 <= p99 < 40.0) or (0 < gap_gt30 <= 5):
        verdict = "MILD_JITTER"
    elif (40.0 <= p99 < 80.0) or (5 < gap_gt30 <= 30):
        verdict = "SIGNIFICANT_JITTER"
    else:
        verdict = "SEVERE_JITTER"
    _emit(f"[diag VERDICT] {verdict}", fh)

    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
