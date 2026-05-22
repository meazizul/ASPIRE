# chunker.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

import imagehash

CHUNK_MS = 5_000
LOOKAHEAD_MS = 2_000
TIME_BUFFER_MS = 5_000

# Cap unique_hashes to prevent unbounded O(N) growth in 1h+ sessions.
MAX_UNIQUE_HASHES = 500


@dataclass
class Session:
    session_id: str
    video_path: str
    duration_ms: int
    num_chunks: int
    ready: Dict[int, bool] = field(default_factory=dict)
    processing: Dict[int, bool] = field(default_factory=dict)
    # content signatures (title+bullets hash) we've already TTS'd
    content_sigs: Set[str] = field(default_factory=set)
    # session-wide slide numbering
    slide_counter: int = 0
    edited_audio_length_ms: int = 0
    # unique_hashes is used by SlideDetector for visual dedup. Deque with
    # maxlen prevents pathological growth on multi-hour sessions.
    unique_hashes: Deque[imagehash.ImageHash] = field(
        default_factory=lambda: deque(maxlen=MAX_UNIQUE_HASHES)
    )
    edited_audio_length_list: List[int] = field(default_factory=list)
    # Carryover state from the tail of the previous chunk so chunk boundaries
    # don't auto-fire a slide candidate. Populated by SlideDetector.detect().
    last_phash: Optional[imagehash.ImageHash] = None
    last_hist: Optional[Any] = None    # numpy.ndarray (HSV hist)
    last_small: Optional[Any] = None   # numpy.ndarray (downscaled gray)


def plan_chunks(duration_ms: int) -> int:
    return math.ceil(duration_ms / CHUNK_MS) if duration_ms > 0 else 0
