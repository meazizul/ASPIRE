# slide_detect.py
"""
Text-first slide detector.

  Primary signal:  Apple Vision OCR text on the frame, normalized + sorted
                   words → SHA1 → 16-hex content_signature. Same text =
                   same slide regardless of cursor / animation / bullet
                   reorder. New text = new slide candidate.

  Fallback:        for image-only slides (text < 20 chars OR < 3 useful
                   words), fall back to a pHash-based "img_<hex>"
                   signature. The visual hash is centre-80% cropped to
                   ignore webcam overlays.

The 5-second stability gate runs in pipeline.py — this module just produces
signatures.

The legacy pHash-only `SlideDetector(...).detect(video_path,...)` is
preserved for the legacy HLS chunk processor (still callable but unused
by the live path).
"""
from __future__ import annotations

import hashlib
import logging
import re
import string
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image

log = logging.getLogger("aspire.slide_detect")

# 5-second stability gate constants (read by pipeline.py)
STABILITY_WINDOW_S = 5.0

# Text-first signature thresholds
TEXT_MIN_CHARS = 20
TEXT_MIN_WORDS = 3
TEXT_MIN_WORD_LEN = 3
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "for",
    "on", "with", "by", "at", "from", "this", "that", "it", "as", "be",
    "are", "was", "were",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tunables (kept for the legacy file walker)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectorParams:
    change_thresh: int = 10
    sustain_frames: int = 2
    unique_thresh: int = 8
    hist_corr_cut: float = 0.55
    pixel_mae_cut: float = 28.0


# ─────────────────────────────────────────────────────────────────────────────
# Text → signature
# ─────────────────────────────────────────────────────────────────────────────
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize_words(text: str) -> List[str]:
    s = text.lower().translate(_PUNCT_TABLE)
    tokens = [t for t in re.split(r"\s+", s)
              if len(t) >= TEXT_MIN_WORD_LEN and t not in STOPWORDS and not t.isdigit()]
    return tokens


def is_text_substantive(text: str) -> bool:
    """True if the OCR result has enough text to use as the primary signal."""
    if not text or len(text) < TEXT_MIN_CHARS:
        return False
    words = _normalize_words(text)
    return len(words) >= TEXT_MIN_WORDS


def text_content_signature(text: str) -> str:
    """Stable SHA1 hex (16 chars) of the sorted unique normalized words.
    Sorted so cursor moves / bullet reorders / minor edits don't all create
    'new' signatures. Same overall content yields same signature."""
    words = sorted(set(_normalize_words(text)))
    payload = " ".join(words)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Image-only fallback (pHash)
# ─────────────────────────────────────────────────────────────────────────────
def _phash_centre(frame_bgr: np.ndarray) -> str:
    h, w, _ = frame_bgr.shape
    cropped = frame_bgr[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]
    img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(img))


def image_only_signature(frame_bgr: np.ndarray) -> str:
    return "img_" + _phash_centre(frame_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point used by pipeline._detect_loop
# ─────────────────────────────────────────────────────────────────────────────
def signature_for_frame(frame_bgr: np.ndarray, ocr_text_simple) -> Tuple[str, str, bool]:
    """Returns (signature, ocr_text, used_text).

    `ocr_text_simple` is a callable (frame_bgr) -> str — passed in so the
    detector doesn't import the Vision module (and so tests can stub it)."""
    text = ""
    try:
        text = ocr_text_simple(frame_bgr) or ""
    except Exception as e:
        log.warning("[ocr] error: %s", e)
        text = ""

    if is_text_substantive(text):
        return text_content_signature(text), text, True

    # Fallback: image-only slide
    return image_only_signature(frame_bgr), text, False


# ─────────────────────────────────────────────────────────────────────────────
# Legacy back-compat: stateless single-frame helper used by old code paths
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FrameState:
    phash: Optional[Any] = None
    hist: Optional[Any] = None
    small: Optional[Any] = None
    changed_run: int = 0


@dataclass
class ChangeEvent:
    kind: str
    phash: Any
    score: float


def _phash_legacy(frame_bgr):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return imagehash.phash(img)


def _hsv_hist(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                     [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h


def _small_gray(frame_bgr):
    return cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                      (64, 36), interpolation=cv2.INTER_AREA)


def detect_change(prev: FrameState, frame_bgr: np.ndarray,
                  params: DetectorParams = DetectorParams()
                  ) -> Tuple[Optional[ChangeEvent], FrameState]:
    """Legacy two-tier pHash detector. Unused by the live path; kept so the
    old HLS chunk processor still imports cleanly."""
    h, w, _ = frame_bgr.shape
    cropped = frame_bgr[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]
    ph = _phash_legacy(cropped)
    hist = _hsv_hist(frame_bgr)
    small = _small_gray(frame_bgr)

    if prev.phash is None:
        new = FrameState(phash=ph, hist=hist, small=small, changed_run=0)
        return ChangeEvent("seed", ph, 0.0), new

    corr = float(cv2.compareHist(prev.hist, hist, cv2.HISTCMP_CORREL))
    mae = float(np.mean(np.abs(prev.small.astype(np.int16) - small.astype(np.int16))))
    if corr < params.hist_corr_cut or mae > params.pixel_mae_cut:
        new = FrameState(phash=ph, hist=hist, small=small, changed_run=0)
        return ChangeEvent("hard", ph, corr), new

    dist = prev.phash - ph
    run = prev.changed_run + 1 if dist > params.change_thresh else 0
    if run >= params.sustain_frames:
        new = FrameState(phash=ph, hist=hist, small=small, changed_run=0)
        return ChangeEvent("soft", ph, float(dist)), new

    return None, FrameState(phash=ph, hist=hist, small=small, changed_run=run)


# Legacy file-based walker (HLS path). Untouched.
class SlideDetector:
    def __init__(self, sample_ms: int = 400, change_thresh: int = 10,
                 sustain_frames: int = 2, session=None,
                 unique_thresh: int = 8,
                 hist_corr_cut: float = 0.55,
                 pixel_mae_cut: float = 28.0):
        self.sample_ms = sample_ms
        self.params = DetectorParams(
            change_thresh=change_thresh, sustain_frames=sustain_frames,
            unique_thresh=unique_thresh, hist_corr_cut=hist_corr_cut,
            pixel_mae_cut=pixel_mae_cut,
        )
        self.session = session

    def _is_duplicate(self, h) -> bool:
        if self.session is None or not self.session.unique_hashes:
            return False
        return any((uh - h) <= self.params.unique_thresh
                   for uh in self.session.unique_hashes)

    def detect(self, video_path: str, start_ms: int, duration_ms: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        end_ms = start_ms + duration_ms
        prev = FrameState(
            phash=getattr(self.session, "last_phash", None),
            hist=getattr(self.session, "last_hist", None),
            small=getattr(self.session, "last_small", None),
            changed_run=0,
        )
        results = []
        t_ms = start_ms
        while t_ms < end_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ok, frame = cap.read()
            if not ok:
                break
            event, prev = detect_change(prev, frame, self.params)
            if event is not None:
                rel_ms = max(0, t_ms - start_ms)
                if not self._is_duplicate(event.phash):
                    results.append((rel_ms, event.phash))
                    self.session.unique_hashes.append(event.phash)
            t_ms += self.sample_ms
        cap.release()
        if self.session is not None and prev.phash is not None:
            self.session.last_phash = prev.phash
            self.session.last_hist = prev.hist
            self.session.last_small = prev.small
        return sorted({t: ph for t, ph in results}.items())


# ── Legacy OCR helpers retained as no-ops so existing callers don't break ──
def ocr_text(frame_bgr: np.ndarray) -> str:
    """Legacy entry point — no longer used by the live pipeline; returns ""."""
    return ""


def text_hash(normalized: str) -> str:
    return hashlib.sha1((normalized or "").encode("utf-8")).hexdigest()[:16]


def extract_title_line(normalized: str, max_chars: int = 80) -> str:
    return (normalized or "")[:max_chars]


# Constants kept for back-compat imports
STABILITY_HAMMING = 4
