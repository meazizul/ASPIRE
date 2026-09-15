# tts_kokoro.py
"""
Local Kokoro TTS wrapper. The model is loaded once at process start (~1s)
and reused for every synth call. synth() is awaited from the event loop and
runs the actual ONNX inference on a background thread so we never block.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("aspire.tts")

DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_SPEED = float(os.environ.get("KOKORO_SPEED", "1.5"))
DEFAULT_LANG = os.environ.get("KOKORO_LANG", "en-us")

_MODEL_DIR = Path(__file__).parent / "models" / "kokoro"
_MODEL_PATH = _MODEL_DIR / "kokoro-v1.0.onnx"
_VOICES_PATH = _MODEL_DIR / "voices-v1.0.bin"

_kokoro = None  # type: ignore
# kokoro-onnx is NOT thread-safe — concurrent .create() calls segfault the
# underlying ONNX runtime. Serialize all synth calls through this lock.
_kokoro_lock = asyncio.Lock()


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        if not _MODEL_PATH.exists() or not _VOICES_PATH.exists():
            raise RuntimeError(
                f"Kokoro model files missing in {_MODEL_DIR}; "
                "run the install step from the upgrade guide."
            )
        t0 = time.time()
        _kokoro = Kokoro(str(_MODEL_PATH), str(_VOICES_PATH))
        log.info("[tts] kokoro loaded in %.2fs", time.time() - t0)
    return _kokoro


import re as _re_num

# Kokoro's text frontend treats a period as a sentence break, so "261.5" is
# spoken as "two hundred sixty-one." <pause> "five", and a thousands separator
# splits "1,053" into "one" <pause> "fifty-three". Both were reported as
# unintelligible by listeners. Normalizing the text before synthesis removes the
# ambiguity — the decimal point becomes the spoken word "point", and grouping
# commas are simply dropped so the number reads as a single quantity.
_DECIMAL_RE = _re_num.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_RE = _re_num.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize_numbers_for_speech(text: str) -> str:
    """Make numerals unambiguous for the TTS frontend.

    "EUR 261.5 billion" -> "EUR 261 point 5 billion"
    "1,053 patients"    -> "1053 patients"
    """
    if not text:
        return text
    out = _THOUSANDS_RE.sub("", text)
    out = _DECIMAL_RE.sub(" point ", out)
    return out


def _synth_blocking(text: str, voice: str, speed: float, lang: str
                    ) -> Tuple[np.ndarray, int]:
    k = get_kokoro()
    text = normalize_numbers_for_speech(text)
    samples, sr = k.create(text, voice=voice, speed=speed, lang=lang)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    return samples, int(sr)


async def synth(text: str,
                voice: str = DEFAULT_VOICE,
                speed: float = DEFAULT_SPEED,
                lang: str = DEFAULT_LANG) -> Tuple[np.ndarray, int]:
    """Async TTS. Returns (mono float32 in [-1,1], sample_rate_hz).
    Serialized via _kokoro_lock — kokoro-onnx is not thread-safe."""
    t0 = time.time()
    async with _kokoro_lock:
        samples, sr = await asyncio.to_thread(_synth_blocking, text, voice, speed, lang)
    log.info("[tts] kokoro %.2fs synth, %.2fs audio @ %dHz",
             time.time() - t0, len(samples) / sr, sr)
    return samples, sr


# ─────────────────────────────────────────────────────────────────────────────
# Streaming synth: yields per-sentence (samples, sr) chunks. The first chunk
# arrives as soon as the first sentence finishes synthesizing — typically
# 200–500 ms vs. ~1.5 s for a full 20-word description. Subsequent chunks
# synthesize in parallel via asyncio tasks.
# ─────────────────────────────────────────────────────────────────────────────
import re as _re

_SENT_SPLIT = _re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, max_chars_per_chunk: int = 90):
    """Sentence-split, then merge tiny fragments so we don't synth 5-word stubs."""
    raw = [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]
    if not raw:
        return []
    out, cur = [], ""
    for s in raw:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars_per_chunk:
            cur = cur + " " + s
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


async def synth_stream(text: str,
                       voice: str = DEFAULT_VOICE,
                       speed: float = DEFAULT_SPEED,
                       lang: str = DEFAULT_LANG):
    """Async generator yielding (samples, sr) per sentence. SERIALIZED
    through _kokoro_lock because kokoro-onnx is not thread-safe — concurrent
    .create() calls segfault the ONNX runtime.

    Latency benefit is preserved because the FIRST sentence (the short one)
    finishes synthesizing in ~300-500 ms and is yielded immediately, while
    the mixer starts playing it. The second sentence then synthesizes
    serially during the first one's playback, so by the time the mixer
    finishes the first chunk the second is ready — no audible gap."""
    sentences = _split_sentences(text)
    if not sentences:
        return

    log.info("[tts] streaming %d sentences (serialized)", len(sentences))
    t_kick = time.time()
    for i, s in enumerate(sentences):
        t0 = time.time()
        async with _kokoro_lock:
            samples, sr = await asyncio.to_thread(
                _synth_blocking, s, voice, speed, lang)
        log.info("[tts] stream chunk %d/%d ready: %.2fs synth, %.2fs audio (since kick %.2fs)",
                 i + 1, len(sentences), time.time() - t0, len(samples) / sr,
                 time.time() - t_kick)
        yield samples, sr


def warm_up() -> None:
    """Force the model load now (call from FastAPI startup) so the first user
    request isn't penalized by the ~1s cold start."""
    get_kokoro()
