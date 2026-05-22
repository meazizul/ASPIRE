# vision_ocr.py
"""
Apple Vision OCR wrapper. On-device, optimized for screen content,
~50–150 ms per frame on Apple Silicon.

Used by the slide detector as the PRIMARY signal for content change
(text-first detection). Vision is thread-safe — each call creates its
own VNImageRequestHandler, so this runs fine from the detect_loop's
asyncio.to_thread() worker.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Tuple

import cv2
import numpy as np

import Vision
import Quartz
from Cocoa import NSData

log = logging.getLogger("aspire.ocr")

# Rolling timing window for status reporting
_TIMINGS: Deque[float] = deque(maxlen=200)
_call_count = 0


# Downscale before OCR: Vision is roughly linear in pixel count. 720p → 540p
# halves OCR time while keeping slide text recognizable.
OCR_TARGET_LONGEST_SIDE = 960


def _ndarray_to_cgimage(frame_bgr: np.ndarray):
    """Convert a BGR numpy frame to a CGImageRef via JPEG bytes (faster than PNG)."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest > OCR_TARGET_LONGEST_SIDE:
        scale = OCR_TARGET_LONGEST_SIDE / longest
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    if src is None:
        raise RuntimeError("CGImageSourceCreateWithData failed")
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        raise RuntimeError("CGImageSourceCreateImageAtIndex failed")
    return img


def _record_time(elapsed_s: float) -> None:
    global _call_count
    _TIMINGS.append(elapsed_s)
    _call_count += 1
    if _call_count % 20 == 0:
        log.info("[ocr] vision avg=%.1fms p50=%.1fms p95=%.1fms (last %d)",
                 1000 * (sum(_TIMINGS)/len(_TIMINGS)),
                 1000 * float(np.median(_TIMINGS)),
                 1000 * float(np.quantile(_TIMINGS, 0.95)),
                 len(_TIMINGS))


def stats() -> dict:
    if not _TIMINGS:
        return {"ocr_p50_ms": 0.0, "ocr_p95_ms": 0.0, "ocr_calls": 0}
    return {
        "ocr_p50_ms": float(np.median(_TIMINGS)) * 1000,
        "ocr_p95_ms": float(np.quantile(_TIMINGS, 0.95)) * 1000,
        "ocr_calls": _call_count,
    }


def extract_text(frame_bgr: np.ndarray) -> List[dict]:
    """Run Vision and return a list of {text, confidence, bbox} dicts.
    bbox is (x, y, w, h) in normalized 0..1 with origin at the bottom-left
    (Vision's native coord system)."""
    t0 = time.time()
    cg = _ndarray_to_cgimage(frame_bgr)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    # Accurate level. We only call this once per committed slide (every 30+
    # seconds), so the ~150ms cost is irrelevant — and Fast was producing
    # "CIRud¢ SetthJg8" instead of "Claude Settings", corrupting Haiku's titles.
    try:
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    except Exception:
        pass
    try:
        request.setUsesLanguageCorrection_(True)
    except Exception:
        pass
    try:
        request.setRecognitionLanguages_(["en-US"])
    except Exception:
        pass
    # Skip very small text — speeds things up and cuts UI-chrome noise.
    try:
        request.setMinimumTextHeight_(0.015)
    except Exception:
        pass

    err = handler.performRequests_error_([request], None)
    elapsed = time.time() - t0
    _record_time(elapsed)

    results: List[dict] = []
    observations = request.results() or []
    for obs in observations:
        candidate = obs.topCandidates_(1)
        if not candidate or len(candidate) == 0:
            continue
        c = candidate[0]
        text = str(c.string())
        conf = float(c.confidence())
        bbox = obs.boundingBox()
        # bbox is a CGRect ((x, y), (w, h))
        try:
            x = float(bbox.origin.x); y = float(bbox.origin.y)
            w = float(bbox.size.width); h = float(bbox.size.height)
        except Exception:
            x = y = w = h = 0.0
        results.append({"text": text, "confidence": conf, "bbox": (x, y, w, h)})
    return results


def extract_text_simple(frame_bgr: np.ndarray, conf_thresh: float = 0.4) -> str:
    """Concatenate all recognized strings above confidence threshold in
    natural reading order (top-to-bottom, then left-to-right within rows).
    Vision's bbox y-axis is bottom-up, so we sort by 1-y.

    Returns a single-line string with words space-separated."""
    items = extract_text(frame_bgr)
    if not items:
        return ""
    # Filter low-confidence
    items = [it for it in items if it["confidence"] >= conf_thresh and it["text"].strip()]
    if not items:
        return ""
    # Convert to (top, left, text) where top is 1 - (y + h) so smaller = higher.
    rows: List[Tuple[float, float, str]] = []
    for it in items:
        x, y, w, h = it["bbox"]
        top = 1.0 - (y + h)
        rows.append((top, x, it["text"]))

    # Group into rows by approximate y; tolerance = 1.5% of frame height
    rows.sort(key=lambda r: (r[0], r[1]))
    grouped: List[List[Tuple[float, float, str]]] = []
    row_tol = 0.015
    for r in rows:
        if grouped and abs(r[0] - grouped[-1][0][0]) <= row_tol:
            grouped[-1].append(r)
        else:
            grouped.append([r])

    pieces: List[str] = []
    for row in grouped:
        row.sort(key=lambda r: r[1])
        pieces.append(" ".join(t for _, _, t in row))
    return " ".join(pieces).strip()
