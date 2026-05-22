"""Shared diagnostic event writer.

Every transition point in the live pipeline calls `record(source, event, **fields)`.
A background drain thread serializes events to /tmp/aspire-diag-events.jsonl —
one JSON object per line. The call site NEVER blocks on disk I/O; if the
internal queue fills, events are dropped and the dropped count is recorded.

This module is import-safe even if `start()` is never called — `record()`
becomes a no-op until a writer thread is running.

Behavior-neutral: emits observations only. No state is read by the rest of
the pipeline.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

LOG_PATH = Path("/tmp/aspire-diag-events.jsonl")
ROTATE_BYTES = 50 * 1024 * 1024  # 50 MB

_QUEUE_MAX = 20000
_q: "queue.Queue[bytes]" = queue.Queue(maxsize=_QUEUE_MAX)
_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_dropped: int = 0
_running = False


def is_running() -> bool:
    return _running


def record(source: str, event: str, **fields: Any) -> None:
    """Non-blocking. Safe to call from any thread (incl. asyncio coroutines).
    If the queue is full, the event is dropped and a counter increments."""
    global _dropped
    if not _running:
        return
    obj = {"ts": time.time(), "source": source, "event": event}
    if fields:
        # default=str so weird types (Path, ImageHash) don't explode the writer.
        obj.update(fields)
    try:
        line = (json.dumps(obj, default=str) + "\n").encode("utf-8")
    except Exception:
        return
    try:
        _q.put_nowait(line)
    except queue.Full:
        _dropped += 1


def _writer_loop() -> None:
    fh = None
    bytes_written = 0
    try:
        # Truncate on session start.
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            LOG_PATH.write_bytes(b"")
        except OSError:
            pass
        fh = LOG_PATH.open("ab")
        # Boot event so analyze can find the start
        fh.write((json.dumps({
            "ts": time.time(),
            "source": "system",
            "event": "diag_session_start",
            "queue_max": _QUEUE_MAX,
        }) + "\n").encode("utf-8"))
        fh.flush()

        while not _stop.is_set():
            try:
                line = _q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                fh.write(line)
                bytes_written += len(line)
                if bytes_written > ROTATE_BYTES:
                    fh.flush()
                    try:
                        fh.close()
                    except Exception:
                        pass
                    rotated = LOG_PATH.with_suffix(f".{int(time.time())}.jsonl")
                    try:
                        LOG_PATH.rename(rotated)
                    except OSError:
                        pass
                    fh = LOG_PATH.open("ab")
                    bytes_written = 0
            except Exception:
                # Disk error; skip this line but keep loop alive
                pass

            # Opportunistically batch-flush every ~200 events
            if _q.qsize() == 0:
                try:
                    fh.flush()
                except Exception:
                    pass
    finally:
        if fh is not None:
            try:
                fh.write((json.dumps({
                    "ts": time.time(),
                    "source": "system",
                    "event": "diag_session_end",
                    "dropped_total": _dropped,
                }) + "\n").encode("utf-8"))
                fh.flush()
                fh.close()
            except Exception:
                pass


def start() -> None:
    """Idempotent. Call once when the pipeline starts."""
    global _thread, _running, _stop
    if _running:
        return
    _stop = threading.Event()
    _running = True
    _thread = threading.Thread(target=_writer_loop, daemon=True,
                               name="diag-events")
    _thread.start()


def stop() -> None:
    """Idempotent. Signals the drain thread to exit and waits briefly."""
    global _running
    if not _running:
        return
    _running = False
    _stop.set()
    if _thread is not None:
        try:
            _thread.join(timeout=2.0)
        except Exception:
            pass


def dropped_count() -> int:
    return _dropped
