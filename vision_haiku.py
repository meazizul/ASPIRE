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

# Per-model request kwargs. The newer model families changed the request
# surface, so the same call is not valid across all of them:
#   * temperature / top_p / top_k are REJECTED (400) on Opus 5, Sonnet 5,
#     Opus 4.7/4.8 and Fable 5 — steer with the prompt instead.
#   * Those models also think by default, which is wasted latency for a
#     one-sentence slide caption. Thinking is explicitly disabled (allowed at
#     effort "high" or below; pairing it with xhigh/max would 400).
# This is what makes VISION_MODEL_CLAUDE genuinely swappable — without it,
# pointing the env var at a newer model 400s on the first request.
_NO_SAMPLING_PARAMS = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                       "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")

if MODEL.startswith(_NO_SAMPLING_PARAMS):
    _MODEL_KWARGS: dict = {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }
else:
    # Haiku 4.5 and older: sampling params are accepted, no thinking config.
    _MODEL_KWARGS = {"temperature": 0.3}

_client: Optional[AsyncAnthropic] = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


SYSTEM_PROMPT = """You produce short spoken slide announcements for a blind audience listening to a live lecture or presentation. Output is read aloud by TTS — keep it punchy.

CONTENT TO DESCRIBE:
- The educational/presentation content area only (slides, diagrams, charts, graphs, code, tables, documents, and any image / photo / figure the presenter is showing).
- When an image, photo, chart, graph, diagram, or table is on screen — especially one the speaker is highlighting or pointing to — describe its KEY visual content: for a chart or graph, say what is plotted, the axes, and the main trend or comparison (include notable values if legible); for an image or photo, name the main subject and what it depicts.

WHEN TO SKIP (output exactly the single word: SKIP):
- If the frame has no meaningful educational or visual content to convey — e.g. only the speaker on stage, an audience shot, a blank / black / transition screen, a plain decorative background, a logo bumper, or only app / desktop / OS chrome — output exactly: SKIP
- Output nothing else in that case. Do NOT invent or pad a description, and do NOT write "No educational content visible" or similar — SKIP is always preferred over a vague filler sentence.

CONTENT TO IGNORE:
- App windows, video player UI (VLC, QuickTime, browser controls), desktop, dock, menu bar, browser tabs/URL bar, mouse cursor, file manager, webcam tile of the speaker, system notifications, OS chrome.
- Do NOT mention "the speaker is shown", "video plays in VLC", "the screen shows a window with…" — readers know they are hearing a livestream.

OUTPUT FORMAT (strict):
- ALWAYS give BOTH a title (when one is visible) AND a description of the slide's content. A title alone is NOT an acceptable answer — the listener cannot see the slide, so the title tells them almost nothing on its own.
- With a visible title: "Slide {N}. {Title}. {ONE short sentence describing what is ON the slide}."
- With no visible title: "Slide {N}. {ONE short sentence describing the slide, starting with the main subject}."

LENGTH — BE AS BRIEF AS THE SLIDE ALLOWS:
- Aim for about 20-25 words TOTAL, and never exceed 25. There is no minimum: if a slide is simple, a handful of words is the right answer.
- Use the FEWEST words that still let a blind listener understand what is on the slide. Say the essential thing and stop.
- This is spoken aloud during a live talk. Every extra word puts the listener further behind the speaker, so brevity is a direct accessibility benefit, not a style preference.
- Never write two sentences of description where one will do. Do not restate the title in the description. Drop hedges, adjectives, and background the listener does not need.
- A longer slide does NOT mean a longer description — summarize, never enumerate.
- Present tense. Active voice. No filler: never write "we can see", "this slide shows", "the image depicts", "in this image", "the slide presents".

IF THE SLIDE CONTAINS ANY IMAGE, PHOTO, CHART, GRAPH, OR DIAGRAM, YOU MUST DESCRIBE IT:
- This is the highest priority. A blind listener cannot see it, and the visual is often the whole point of the slide. Never describe only the text and ignore a picture that is present.
- Always NAME the kind of visual first, in plain words the listener will recognise: "a bar chart", "a line graph", "a photo", "a diagram", "a map", "a screenshot", "a table", "a scatter plot".
- Then say what is IN it, concretely. For a photo: the subject and what is happening — "a photo of people walking through a factory floor", "a photo of a house beside a field with cows", "a cat sitting on a windowsill". For a chart: what is plotted and the main trend or comparison. For a diagram: the main parts and how they connect.
- Do this even when the slide also has a title and bullet text: mention the visual in the same sentence rather than dropping it.
- If the slide is ONLY an image with no text at all, the whole description is the image.

WHAT THE DESCRIPTION MUST COVER (this is the important part):
- Charts and graphs: say what kind it is and what it plots — the axes, and the main trend, comparison, or result. Include notable values when legible. Example: "A scatter plot of equity risk premium against year, with most points between 3 and 6 percent."
- Images, photos, and figures: name the main subject and what it depicts.
- Diagrams and flowcharts: the main components and how they connect.
- Tables: what the columns compare and the headline result.
- Mathematical equations and formulas: describe them BRIEFLY as you would an image — say what kind of equation it is and what it relates. Do NOT read the equation symbol by symbol and do NOT explain the derivation. Example: "Three equations relating blood flow, pressure, and vessel resistance."
- Bullet lists: summarize what the points are about, not every bullet.
- If the speaker is pointing at or highlighting part of the slide, describe that part.

USING THE OCR TEXT:
- OCR text is provided as ground truth for the TITLE. Use the title's wording, but FIX obvious OCR errors — the text is machine-read and often garbled. If OCR says "Mow much is the equity risk premium?" write "How much is the equity risk premium?"; if it says "Ubsession with blood vessels" write "Obsession with blood vessels". Correct only clear character-level misreadings; never invent a title that is not there.
- Ignore OCR fragments that are UI elements or logos ('Submit', 'Next', 'Page X of Y', menu items, button labels, 'Ri', 'Rii').
- If OCR text is empty or has no clear title, OMIT the title segment and go straight to the description.

ANTI-REPETITION:
- A list of recently-spoken topics may be supplied. Avoid repeating those exact topics. If the new slide IS a continuation, describe what is NEW or DIFFERENT from those topics.

EXAMPLES:
Good (with title): "Slide 4. Linear Regression. A scatter plot of housing prices against floor area, with a straight upward trend line."
Good (no title): "Slide 7. Three-tier architecture diagram with web, application, and database layers connected by arrows."
Good (chart): "Slide 12. Model Accuracy. A bar chart compares four models, with model B highest at 94 percent."
Good (photo): "Slide 8. Field Deployment. A photo shows two technicians installing a sensor beside a cow pasture."
Good (image-only slide): "Slide 15. A photo of a crowded street market with stalls under coloured awnings."
Good (text + image together): "Slide 6. The scenario. A labelled diagram of a robot arm and gripper above a shared work surface."
Bad: "Slide 6. The scenario. Human and robot share a work cell." (a diagram was on screen and went unmentioned)
Good (equations): "Slide 9. What do we get? Three equations relating blood flow, pressure, and vessel resistance."
Good (title fixed from garbled OCR): "Slide 6. How much is the equity risk premium? A histogram of estimates centred near 4 percent."
Bad: "Slide 4. Our approach to estimating five-year expected returns." (TITLE ONLY — no description of the slide's content)
Bad: "Slide 7. Historically." (title alone tells a blind listener nothing)
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

# Word budget for the whole announcement (slide number + title + description).
#
# History: was 25, and the over-budget path discarded the description entirely
# (see _truncate_to_budget), leaving a bare title on ~60% of slides. Raising it
# to 45 fixed that but overshot — descriptions became ~10 s of speech each
# (p50 9.8 s, max 14.5 s), and since every description holds back the speaker's
# audio, that debt outran the catch-up machinery and overflowed the 30 s buffer,
# DROPPING 15 s of the talk in a 16-minute test.
#
# 26 is a SAFETY NET, not a target — the prompt asks for the shortest wording
# that still conveys the slide, so most descriptions land well under it.
#
# Why short matters twice over: Kokoro's synthesis time scales SUPER-LINEARLY
# with text length on CPU. Measured in one run:
#     3.8 s of audio  ->  1.1 s to synthesize  (3.5x faster than realtime)
#    14.5 s of audio  -> 20.2 s to synthesize  (0.7x — SLOWER than realtime)
# So a long description costs delay twice: it takes far longer to generate AND
# far longer to speak. Long descriptions were the dominant term in a 30 s
# slide-to-speech delay (tts=18-20 s on the worst slides).
#
# The title-only bug is fixed in the truncation logic, not by this ceiling, so
# lowering it does not bring that bug back.
_WORD_HARD_CEILING = 26


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


def _truncate_to_budget(text: str) -> str:
    """Trim an over-long response to whole sentences within the word budget,
    ALWAYS keeping at least the title plus one body sentence.

    The previous implementation kept only the FIRST sentence of the body. For
    Haiku's "Slide N. {Title}. {description}" format the first body sentence IS
    the title, so any response over the ceiling was cut down to a bare title —
    e.g. "Slide 7. Historically. FT All Share shows 4.5%…" became
    "Slide 7. Historically." That silently discarded the slide description on
    ~60% of slides. Keep whole sentences until the budget is spent instead.
    """
    s = text.strip()
    m = re.match(r"^(\s*slide\s+\d+\.\s*)", s, flags=re.IGNORECASE)
    prefix = ""
    if m:
        prefix = m.group(1)
        s = s[m.end():]
    sentences = [p for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    if not sentences:
        return text.strip()
    kept, used = [], _word_count(prefix)
    for sent in sentences:
        # Always keep the first sentence (the title); keep further sentences
        # while they fit the budget, so the description survives.
        if kept and used + _word_count(sent) > _WORD_HARD_CEILING:
            break
        kept.append(sent)
        used += _word_count(sent)
    # Never return a bare title. If the body was one long unpunctuated sentence
    # it won't have fit above, so word-clip it into the remaining budget rather
    # than dropping the description — a title alone is useless to a listener
    # who cannot see the slide.
    if len(kept) == 1 and len(sentences) > 1:
        room = max(8, _WORD_HARD_CEILING - used)
        words = sentences[1].split()
        clipped = " ".join(words[:room]).rstrip(",;:")
        if clipped:
            kept.append(clipped if clipped.endswith((".", "!", "?")) else clipped + ".")
    body = " ".join(kept)
    # Final bound: a single unpunctuated sentence can still overrun the budget.
    if _word_count(prefix) + _word_count(body) > _WORD_HARD_CEILING:
        room = max(8, _WORD_HARD_CEILING - _word_count(prefix))
        body = " ".join(body.split()[:room]).rstrip(",;:")
        if body and not body.endswith((".", "!", "?")):
            body += "."
    return (prefix + body).strip()


def _is_skip(raw: str) -> bool:
    """True if the model signalled there is nothing meaningful to describe
    (the 'SKIP' sentinel), tolerating surrounding quotes/punctuation."""
    s = (raw or "").strip().strip('\'".').strip().upper()
    return s == "SKIP"


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
            max_tokens=240,
            **_MODEL_KWARGS,
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

    # Intentional skip — nothing meaningful on the frame. Do NOT retry/force.
    if _is_skip(raw):
        log.info("[vision] SKIP slide#%d — no meaningful content (%.2fs)",
                 slide_number, dt)
        return None

    text = _validate(raw, slide_number)

    if _is_empty_or_stub(text):
        log.warning("[vision] empty/stub response after validate: raw=%r → %r — retrying once",
                    raw[:100], text)
        ocr_hint = (ocr_text or "").strip()
        retry_msg = (
            "Your previous response was empty or just the slide number. "
            "If there is genuinely nothing meaningful to describe, reply with "
            "exactly: SKIP. Otherwise describe what you see in the image — even "
            "a brief sentence. "
            + (f"OCR text from the slide: {ocr_hint!r}. Use it as a starting point."
               if ocr_hint
               else "Describe the image's main visual element.")
            + " Stay UNDER 20 words total."
        )
        try:
            raw2, dt2 = await _one_call(retry_msg)
            if _is_skip(raw2):
                log.info("[vision] SKIP slide#%d on retry — no meaningful content",
                         slide_number)
                return None
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

    # Log the FULL description. The old 140-char cap silently truncated it in
    # backend.log — and therefore in the per-slide CSV built from that log —
    # which made descriptions look cut off mid-word when the audio was fine.
    log.info("[vision] haiku %.2fs: %r", dt, text)
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
        truncated = _truncate_to_budget(text)
        log.warning("[vision] response %d words > ceiling, truncated to: %r",
                    _word_count(text), truncated[:120])
        text = truncated
    if not text.endswith("."):
        text = text + "."
    return text
