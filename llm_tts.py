# llm_tts.py
"""
LLM (vision) + TTS helpers.

This version adds ASYNC variants (analyze_slide_async, tts_async) so that
multiple slides within a single chunk can be processed in parallel, cutting
end-to-end latency roughly in half for chunks that contain 2+ new slides.

The sync versions are retained for backward compatibility with the existing
prerecorded-video flow.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")
TTS_MODEL    = os.environ.get("TTS_MODEL",    "gpt-4o-mini-tts")
TTS_VOICE    = os.environ.get("TTS_VOICE",    "alloy")

# ── Lazy singletons ─────────────────────────────────────────────────────────
_client      = None   # sync  OpenAI client
_client_async = None  # async OpenAI client


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def get_async_client():
    global _client_async
    if _client_async is None:
        from openai import AsyncOpenAI
        _client_async = AsyncOpenAI()
    return _client_async


# ── Vision prompt ───────────────────────────────────────────────────────────
ANALYZE_PROMPT = (
    "You are a slide analyzer for a blind listener. Return STRICT JSON:\n"
    "{\n"
    '  "title": "<the main heading/subject — 3 to 6 words>",\n'
    '  "bullets": [],\n'
    '  "summary": "<ONE fragment, at most 8 words, on the core topic>"\n'
    "}\n"
    "Focus ONLY on the central content the presenter is discussing. IGNORE:\n"
    "- logos, watermarks, backgrounds, decorative graphics\n"
    "- headers, footers, page/slide numbers, section markers\n"
    "- navigation bars, progress bars, player UI, browser chrome\n"
    "- webcam overlays, participant lists, chat boxes\n"
    "- any text smaller than the main body content\n"
    "If it's an image/chart, name it briefly (e.g. 'bar chart of sales'). "
    "Hard limit: title + summary together must fit 15 words so the spoken "
    "form finishes in under 3 seconds. Be terse. No markdown, no preamble.\n"
)


def _b64_data_url(pil_img: Image.Image) -> str:
    buf = BytesIO()
    # Downsize large screenshots to keep payload small (huge speedup on 4K monitors).
    if max(pil_img.size) > 1280:
        pil_img = pil_img.copy()
        pil_img.thumbnail((1280, 1280))
    pil_img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _clean_json(s: str) -> str:
    s = s.strip()
    m = re.search(r"\{.*\}", s, re.S)
    return m.group(0) if m else s


def _pack_analysis(raw: str) -> Dict:
    try:
        data = json.loads(_clean_json(raw))
    except Exception:
        data = {"title": "", "bullets": [], "summary": raw.strip()[:140]}

    title   = (data.get("title") or "").strip()
    bullets = [b.strip() for b in (data.get("bullets") or []) if isinstance(b, str)]
    summary = (data.get("summary") or "").strip()

    # Signature must be stable against LLM wording drift — the same slide shown
    # twice may produce slightly different summary phrasings, so we hash the
    # normalized title alone (alphanumeric only, lowercased). If the title is
    # empty, fall back to the first 40 alphanumerics of the summary.
    norm_title = re.sub(r"[^a-z0-9]+", "", title.lower())
    if norm_title:
        sig_src = norm_title
    else:
        sig_src = re.sub(r"[^a-z0-9]+", "", summary.lower())[:40]
    if sig_src:
        signature = hashlib.sha1(sig_src.encode("utf-8")).hexdigest()[:24]
    else:
        # No content to hash — give a unique sig so this slide is not incorrectly
        # deduped with other empty-analysis slides.
        signature = "empty-" + uuid.uuid4().hex[:18]

    return {"title": title, "bullets": bullets, "summary": summary, "signature": signature}


# ── Sync (kept for prerecorded mode) ────────────────────────────────────────
def analyze_slide(pil_img: Image.Image) -> Dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYZE_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": _b64_data_url(pil_img), "detail": "low"}},
            ],
        }],
        temperature=0.2,
        max_tokens=220,
    )
    return _pack_analysis(resp.choices[0].message.content or "{}")


def tts(text: str, out_path: Path) -> int:
    """Synthesize TTS to MP3. Returns duration in ms (0 on failure)."""
    from pydub import AudioSegment

    out_path = out_path.with_suffix(".mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client()
    resp = client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, speed=1.75,
    )
    with open(out_path, "wb") as f:
        f.write(resp.content)

    try:
        return len(AudioSegment.from_file(out_path, format="mp3"))
    except Exception:
        return 0


# ── Async (used by the real-time segment flow) ──────────────────────────────
async def analyze_slide_async(pil_img: Image.Image) -> Dict:
    client = get_async_client()
    resp = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYZE_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": _b64_data_url(pil_img), "detail": "low"}},
            ],
        }],
        temperature=0.2,
        max_tokens=220,
    )
    return _pack_analysis(resp.choices[0].message.content or "{}")


async def tts_async(text: str, out_path: Path) -> int:
    """Async TTS. Returns duration in ms (0 on failure)."""
    from pydub import AudioSegment

    out_path = out_path.with_suffix(".mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_async_client()
    resp = await client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, speed=1.75,
    )
    # The response body is bytes; write asynchronously where possible.
    content = resp.content if isinstance(resp.content, (bytes, bytearray)) else await resp.aread()
    await asyncio.to_thread(_write_bytes, out_path, content)

    try:
        return await asyncio.to_thread(_audio_len, out_path)
    except Exception:
        return 0


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _audio_len(path: Path) -> int:
    from pydub import AudioSegment
    return len(AudioSegment.from_file(path, format="mp3"))


# ── Shared TTS-text builder ─────────────────────────────────────────────────
def build_tts_text(slide_no: int, title: str, bullets: List[str], summary: str) -> str:
    parts = [f"Slide {slide_no}. {title.strip('.')}."] if title else [f"Slide {slide_no}."]
    if summary:
        parts.append(summary.strip())
    return " ".join(parts)
