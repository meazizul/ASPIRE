"""Text-based slide-region detection.

In a wide conference scene (hall + speaker + a small projected slide), the
slide is the rectangle where the readable text is concentrated. Apple Vision
already gives us per-line text bounding boxes; this module clusters them and
returns the bounding rectangle of the dominant text cluster — the slide.

Pure / pipeline-agnostic: `roi_from_text_items()` takes the list of OCR items
(as produced by vision_ocr.extract_text) and returns an ROI as
(x, y, w, h) fractions of the frame with a TOP-LEFT origin, or None when there
isn't enough text to decide (caller then falls back to the center crop).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def roi_from_text_items(items: List[dict],
                        conf_thresh: float = 0.40,
                        gap: float = 0.12,
                        pad: float = 0.03,
                        min_boxes: int = 2,
                        max_coverage: float = 0.55,
                        ) -> Optional[Tuple[float, float, float, float]]:
    """Cluster OCR text boxes; return the dominant cluster's bounding rect as
    (x, y, w, h) fractions (top-left origin), or None.

    `items`: list of {text, confidence, bbox=(x,y,w,h)} where bbox is
    normalized 0..1 with a BOTTOM-LEFT origin (Vision's native system)."""
    # Convert qualifying boxes to top-left rects (x0,y0,x1,y1).
    rects: List[Tuple[float, float, float, float]] = []
    for it in items:
        if it.get("confidence", 0.0) < conf_thresh:
            continue
        if not (it.get("text") or "").strip():
            continue
        x, y, w, h = it["bbox"]
        rects.append((x, 1.0 - (y + h), x + w, 1.0 - y))
    if len(rects) < min_boxes:
        return None

    # Union-find: connect boxes whose slightly-expanded rects overlap.
    n = len(rects)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    g = gap / 2.0
    exp = [(r[0] - g, r[1] - g, r[2] + g, r[3] + g) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(exp[i], exp[j]):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(rects[i])

    def total_area(c):
        return sum((r[2] - r[0]) * (r[3] - r[1]) for r in c)

    best = max(clusters.values(), key=total_area)
    if len(best) < min_boxes:
        return None

    x0 = min(r[0] for r in best)
    y0 = min(r[1] for r in best)
    x1 = max(r[2] for r in best)
    y1 = max(r[3] for r in best)

    # Pad a little, then clamp to [0.02, 0.98] so we always drop the extreme
    # screen edges (macOS menu bar / dock) even on full-screen slides.
    x0 = max(0.02, x0 - pad)
    y0 = max(0.02, y0 - pad)
    x1 = min(0.98, x1 + pad)
    y1 = min(0.98, y1 + pad)
    w = x1 - x0
    h = y1 - y0
    if w < 0.08 or h < 0.06:
        return None  # too small to be a real slide
    if w * h > max_coverage:
        # The dominant text cluster fills most of the frame — this is a
        # full-screen / text-everywhere source. Return None so the caller
        # keeps its proven center-80% crop (which also excludes the menu bar
        # and dock). Auto-ROI only engages for genuine sub-region slides.
        return None
    return (round(x0, 4), round(y0, 4), round(w, 4), round(h, 4))
