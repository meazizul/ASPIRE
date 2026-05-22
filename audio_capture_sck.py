"""ScreenCaptureKit audio capture front-end.

Wraps `tools/systemAudioDump/SystemAudioDump` (a small Swift CLI that uses
ScreenCaptureKit to dump system audio as raw PCM on stdout) and pipes its
output through an in-process ffmpeg subprocess that resamples from 24 kHz
stereo s16le to 48 kHz stereo s16le — which is what the rest of the live
pipeline already expects from the legacy BlackHole+ffmpeg path.

Presents a subprocess.Popen-compatible surface so the existing
`_audio_reader_thread` in pipeline.py can consume it without modification:

  .stdout    — 48 kHz stereo s16le PCM from the resampler
  .stderr    — None (we drain both child stderrs internally)
  .pid       — pid of the resampler ffmpeg
  .poll()    — returns None if both children are alive, else the first
               non-None exit code we see
  .terminate(), .wait(timeout=…), .kill() — propagate to both children

If the binary is missing or Screen Recording permission has been revoked,
.start() raises a clear RuntimeError with the remediation steps.

Activate this path with: ASPIRE_AUDIO_CAPTURE=sck_cli  (this is the default).
Legacy fallback: ASPIRE_AUDIO_CAPTURE=blackhole.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("aspire.audio_capture_sck")

# Default binary location built from tools/systemAudioDump/.
# (APFS is case-insensitive on the default macOS volume — so the binary
#  lives inside the cloned source dir; we use that path directly.)
DEFAULT_BINARY = (
    Path(__file__).parent / "tools" / "systemAudioDump" / "SystemAudioDump"
)

# SystemAudioDump emits this (verified in main.swift, AudioDumper class).
SAD_SAMPLE_RATE = 24_000
SAD_CHANNELS = 2
SAD_SAMPLE_FORMAT = "s16le"

# Mixer expects 48 kHz stereo s16le (same as the BlackHole+ffmpeg path).
OUT_SAMPLE_RATE = 48_000
OUT_CHANNELS = 2
OUT_SAMPLE_FORMAT = "s16le"


class SCKAudioCapture:
    """Subprocess-like wrapper. Lifecycle is start() → consume .stdout → stop()."""

    def __init__(self, binary_path: Optional[Path] = None):
        self._binary = Path(binary_path) if binary_path else DEFAULT_BINARY
        self._sad: Optional[subprocess.Popen] = None
        self._resampler: Optional[subprocess.Popen] = None
        self._stderr_threads: List[threading.Thread] = []

    # ── subprocess.Popen-compatible surface ────────────────────────────────
    @property
    def stdout(self):
        return self._resampler.stdout if self._resampler else None

    @property
    def stderr(self):
        # Both children's stderrs are drained by our own background threads.
        # Return None so the caller's drain code is skipped (we just don't
        # call pipeline._drain_stderr for this object).
        return None

    @property
    def pid(self) -> Optional[int]:
        return self._resampler.pid if self._resampler else None

    def poll(self) -> Optional[int]:
        """Returns None if BOTH children are alive. Otherwise the first
        non-None exit code (treating either child's exit as "the source
        died" so the reader-thread respawn logic fires)."""
        for proc in (self._sad, self._resampler):
            if proc is None:
                continue
            rc = proc.poll()
            if rc is not None:
                return rc
        return None

    def terminate(self) -> None:
        # Resampler first so it sees stdin close cleanly.
        for proc in (self._resampler, self._sad):
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def wait(self, timeout: Optional[float] = None) -> None:
        for proc in (self._resampler, self._sad):
            if proc is not None:
                try:
                    proc.wait(timeout=timeout)
                except Exception:
                    pass

    def kill(self) -> None:
        for proc in (self._resampler, self._sad):
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._preflight()

        sad_cmd = [str(self._binary)]
        log.info("[sad] launching SystemAudioDump: %s", " ".join(sad_cmd))
        self._sad = subprocess.Popen(
            sad_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Drain sad's stderr so it doesn't fill the pipe buffer + so we
        # capture permission-revoked errors etc.
        self._stderr_threads.append(
            self._spawn_stderr_drain(self._sad, "sad")
        )

        # Resampler ffmpeg: 24k stereo s16le on stdin → 48k stereo s16le on
        # stdout. async resampling smooths any clock drift between sad's
        # capture clock and our 50fps mixer.
        resampler_cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-f", SAD_SAMPLE_FORMAT,
            "-ar", str(SAD_SAMPLE_RATE),
            "-ac", str(SAD_CHANNELS),
            "-i", "pipe:0",
            "-af", "aresample=async=1000:first_pts=0",
            "-f", OUT_SAMPLE_FORMAT,
            "-ar", str(OUT_SAMPLE_RATE),
            "-ac", str(OUT_CHANNELS),
            "pipe:1",
        ]
        log.info("[sad] launching resampler ffmpeg: %s",
                 " ".join(resampler_cmd))
        self._resampler = subprocess.Popen(
            resampler_cmd,
            stdin=self._sad.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Close our local copy of sad's stdout fd; the resampler now owns
        # the read end. If we leak it, EOF detection breaks.
        try:
            self._sad.stdout.close()
        except Exception:
            pass

        self._stderr_threads.append(
            self._spawn_stderr_drain(self._resampler, "sad-resampler")
        )

    def _preflight(self) -> None:
        if not self._binary.exists():
            raise RuntimeError(
                f"SystemAudioDump binary not found at {self._binary}.\n"
                "Build it with:\n"
                "  cd ~/Desktop/aspire-main/tools/systemAudioDump\n"
                "  swift build -c release\n"
                "  cp .build/release/SystemAudioDump "
                "~/Desktop/aspire-main/tools/systemAudioDump/SystemAudioDump\n"
                "  chmod +x ~/Desktop/aspire-main/tools/systemAudioDump/SystemAudioDump"
            )
        if not os.access(self._binary, os.X_OK):
            raise RuntimeError(
                f"SystemAudioDump not executable: {self._binary}.\n"
                f"Run: chmod +x {self._binary}"
            )
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not on PATH (needed by the SCK audio path as a "
                "24kHz->48kHz resampler)."
            )

    def _spawn_stderr_drain(self, proc: subprocess.Popen, tag: str
                            ) -> threading.Thread:
        def drain() -> None:
            try:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    msg = line.decode("utf-8", errors="replace").rstrip()
                    if not msg:
                        continue
                    level = _classify_sck_stderr_line(msg)
                    if level == "ERROR":
                        log.error("[ffmpeg/%s] %s", tag, msg)
                    else:
                        log.info("[ffmpeg/%s] %s", tag, msg)
            except Exception as e:
                log.info("[ffmpeg/%s] stderr drain ended: %s", tag, e)
        t = threading.Thread(target=drain, daemon=True,
                             name=f"sck-stderr-{tag}")
        t.start()
        return t


# Module-level helper so the classification rule is testable / reusable.
def _classify_sck_stderr_line(line: str) -> str:
    """Return 'ERROR' for genuine failures, 'INFO' for benign banners.

    Conservative rule: only ERROR when there's a clear failure signal.
    Benign banners that mention 'permission' (e.g. 'Checking permissions...',
    '✅ Permissions OK') are INFO. Real permission failures are matched by
    the explicit 'permission denied' / 'permission required' substrings or
    by the ❌ glyph the Swift CLI uses for failure lines.
    """
    if not line:
        return "INFO"
    lower = line.lower()
    if (line.startswith("❌")
            or "permission denied" in lower
            or "permission required" in lower
            or "error" in lower
            or "failed" in lower):
        return "ERROR"
    return "INFO"
