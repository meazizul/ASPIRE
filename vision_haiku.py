# vision_haiku.py
"""
Async Claude Haiku 4.5 vision wrapper for live slide descriptions.

Returns a single short spoken-form announcement that obeys a strict
"Slide N. [Title.] Sentence." template, with OCR text used as ground
truth for the title and last-spoken topics passed only as anti-repetition
hints (NOT as content to extend).
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import List, Optional

from anthropic import AsyncAnthropic
from PIL import Image

log = logging.getLogger("aspire.vision")

MODEL = os.environ.get("VISION_MODEL_CLAUDE", "claude-haiku-4-5")
MAX_LONGEST_SIDE = 1024

_client: Optional[AsyncAnthropic] = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


SYSTEM_PROMPT = """You produce short spoken slide announcements for a blind audience listening to a live lecture or presentation. Output is read aloud by TTS — keep it punchy.

CONTENT TO DESCRIBE:
- The educational/presentation content area only (slides, diagrams, charts, code, documents being presented).
- Describe what is genuinely visible. If you cannot see educational content, describe the scene briefly.

CONTENT TO IGNORE:
- App windows, video player UI (VLC, QuickTime, browser controls), desktop, dock, menu bar, browser tabs/URL bar, mouse cursor, file manager, webcam tile of the speaker, system notifications, OS chrome.
- Do NOT mention "the speaker is shown", "video plays in VLC", "the screen shows a window with…" — readers know they are hearing a livestream.

OUTPUT FORMAT (strict):
- If a clear slide title is visible: "Slide {N}. {Title}. {≤12-word summary of body content}."
- If no title is visible but content is: "Slide {N}. {≤15-word visual description starting with the subject}."
- Hard ceiling: 20 words total. One sentence, one period, optionally a second short sentence for the body.
- Present tense. Active voice. No filler: never write "we can see", "this slide shows", "the image depicts", "in this image", "the slide presents".
- If OCR-extracted text is provided, you MUST use the literal title text as the title. Do NOT translate or rephrase it. Do NOT invent a title that is not in the OCR text.
- If the OCR text contains a clear title (a short capitalized phrase at the top of the text block), use that title verbatim. Do NOT include OCR text that looks like UI elements ('Submit', 'Next', 'Page X of Y', menu items, button labels, navigation breadcrumbs).
- If OCR text is empty or contains no clear title, OMIT the title segment entirely — go straight to the visual description.

ANTI-REPETITION:
- A list of recently-spoken topics may be supplied. Avoid repeating those exact topics. If the new slide IS a continuation, describe what is NEW or DIFFERENT from those topics.

EXAMPLES:
Good (with title): "Slide 4. Linear Regression. A scatter plot of housing prices fits a straight trend line."
Good (no title): "Slide 7. Three-tier architecture diagram with web, application, and database layers connected by arrows."
Good (image only): "Slide 12. A bar chart compares accuracy across four models, with model B highest at 94 percent."
Bad: "This slide shows a diagram of a system with several components." (filler, vague)
Bad: "Slide 5. Introduction. Continuing from the previous architecture topic…" (inheriting from prior context)
Bad: "Slide 3. The user is watching a video in VLC player." (describing app chrome)
"""

# Banned filler prefixes — stripped post-hoc if the model emits them anyway
_BANNED_PREFIXES = [
    "this slide shows",
    "this slide depicts",
    "this slide presents",
    "this slide displays",
    "this slide",
    "we can see",
    "the image depicts",
    "the image shows",
    "the image presents",
    "the image",
    "in this image",
    "the screen shows",
    "the screen displays",
    "the slide shows",
    "the slide presents",
    "the slide",
]

_WORD_HARD_CEILING = 25  # truncate after first sentence if exceeded


def _encode_jpeg(image_bytes_or_pil) -> str:
    if isinstance(image_bytes_or_pil, (bytes, bytearray)):
        img = Image.open(BytesIO(image_bytes_or_pil))
    else:
        img = image_bytes_or_pil
    img = img.convert("RGB")
    if max(img.size) > MAX_LONGEST_SIDE:
        img.thumbnail((MAX_LONGEST_SIDE, MAX_LONGEST_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _strip_banned_prefix(text: str) -> str:
    low = text.lower()
    for p in _BANNED_PREFIXES:
        if low.startswith(p):
            stripped = text[len(p):].lstrip(" ,:;.")
            # Only strip if it leaves at least 6 words — otherwise we'd
            # produce a useless stub. Better imperfect than empty.
            if len(re.findall(r"\b\w+\b", stripped)) < 6:
                return text
            if stripped:
                stripped = stripped[0].upper() + stripped[1:]
            return stripped
    return text


def _normalize_slide_prefix(text: str, slide_number: int) -> str:
    """Force the leading 'Slide N.' to match the actual slide_number."""
    m = re.match(r"^\s*slide\s+\d+[.:\s]*", text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():].lstrip()
    text = f"Slide {slide_number}. {text}"
    return text


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _truncate_to_first_sentence(text: str) -> str:
    """Truncate to the first sentence — but operate on the BODY only, never
    on the leading 'Slide N.' marker which would otherwise be returned alone
    as the first 'sentence'."""
    s = text.strip()
    m = re.match(r"^(\s*slide\s+\d+\.\s*)", s, flags=re.IGNORECASE)
    prefix = ""
    if m:
        prefix = m.group(1)
        s = s[m.end():]
    parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)
    body = parts[0] if parts else s
    return (prefix + body).strip()


def _is_empty_or_stub(text: str) -> bool:
    """True if the response is empty, near-empty, or just 'Slide N.'/'Slide N'."""
    s = (text or "").strip()
    if len(s) < 8:
        return True
    if re.fullmatch(r"slide\s+\d+\.?", s, flags=re.IGNORECASE):
        return True
    # "Slide N. " followed by < 4 chars of body
    body = re.sub(r"^\s*slide\s+\d+\.\s*", "", s, flags=re.IGNORECASE)
    return len(body.strip()) < 4


async def describe_slide(image_bytes: bytes, slide_number: int,
                         ocr_text: str = "",
                         recent_topics: List[str] | None = None) -> Optional[str]:
    """Call Haiku vision; return one strictly-formatted spoken sentence.

    `image_bytes`  : JPEG/PNG bytes (or a PIL.Image).
    `slide_number` : the session-wide 1-based counter (only incremented on
                     committed descriptions, not candidates).
    `ocr_text`     : OCR text from the slide (used as ground-truth title).
    `recent_topics`: anti-repetition hints, NOT context to continue from.
    """
    b64 = _encode_jpeg(image_bytes)
    topics = (recent_topics or [])[-5:]

    user_text_parts = [f"Slide number: {slide_number}"]
    if ocr_text and ocr_text.strip():
        user_text_parts.append(f"OCR-extracted text from slide:\n{ocr_text.strip()}")
    else:
        user_text_parts.append("OCR found no readable text on this slide.")
    if topics:
        user_text_parts.append(
            "Recently spoken topics (avoid repeating these): " + "; ".join(topics)
        )
    user_text_parts.append("Produce the announcement following the strict format.")
    user_text = "\n\n".join(user_text_parts)

    client = get_client()

    async def _one_call(extra_user_text: str = "") -> tuple[str, float]:
        t0 = time.time()
        full_text = user_text + (("\n\n" + extra_user_text) if extra_user_text else "")
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=180,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": full_text},
                ],
            }],
        )
        raw = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                raw += block.text
        return raw.strip(), time.time() - t0

    raw, dt = await _one_call()
    text = _validate(raw, slide_number)

    if _is_empty_or_stub(text):
        log.warning("[vision] empty/stub response after validate: raw=%r → %r — retrying once",
                    raw[:100], text)
        ocr_hint = (ocr_text or "").strip()
        retry_msg = (
            "Your previous response was empty or just the slide number. "
            "Please describe what you see in the image — even a brief sentence. "
            + (f"OCR text from the slide: {ocr_hint!r}. Use it as a starting point."
               if ocr_hint
               else "Describe the image's main visual element.")
            + " Stay UNDER 20 words total."
        )
        try:
            raw2, dt2 = await _one_call(retry_msg)
            text2 = _validate(raw2, slide_number)
            if not _is_empty_or_stub(text2):
                log.info("[vision] retry succeeded: %r", text2[:140])
                return text2
            log.warning("[vision] retry also empty raw=%r → %r — skipping slide",
                        raw2[:100], text2)
            return None
        except Exception as e:
            log.warning("[vision] retry failed: %s — skipping slide", e)
            return None

    log.info("[vision] haiku %.2fs: %r", dt, text[:140])
    return text


def _validate(raw: str, slide_number: int) -> str:
    """Banned-prefix strip → slide-prefix normalize → first-sentence truncate
    on body only → trailing period. Returns "" if the model gave nothing usable."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = _strip_banned_prefix(text)
    text = _normalize_slide_prefix(text, slide_number)
    if _word_count(text) > _WORD_HARD_CEILING:
        truncated = _truncate_to_first_sentence(text)
        log.warning("[vision] response %d words > ceiling, truncated to: %r",
                    _word_count(text), truncated[:120])
        text = truncated
    if not text.endswith("."):
        text = text + "."
    return text
