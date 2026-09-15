# app.py
"""
Aspire RT — real-time slide accessibility backend.

Changes from v0.2:
  * /upload_chunk accepts .ts (MPEG-TS) segments.
  * Per-chunk processing is async (LLM + TTS run in parallel across slides
    found within a chunk).
  * Final audio is written as MPEG-TS (final_<N>.ts) and added to an
    HLS playlist at sessions/<sid>/playlist.m3u8. The playlist only ever
    includes the contiguous prefix starting at chunk 0, so the player
    never skips forward.
  * /listen/<sid> now serves a minimal hls.js-based player.
  * moviepy is no longer used — audio is extracted by piping a direct
    ffmpeg subprocess into pydub.
  * /upload (prerecorded mode) generates a uuid-based session id instead
    of the hardcoded "TestSession".
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import imagehash
import numpy as np
from av import AudioFrame
from aiortc import (MediaStreamTrack, RTCPeerConnection, RTCSessionDescription)
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydub import AudioSegment

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from audio_ops import finalize_chunk_audio
from chunker import CHUNK_MS, TIME_BUFFER_MS, Session, plan_chunks
from llm_tts import analyze_slide_async, build_tts_text, tts_async
from pipeline import (CHANNELS, FRAME_MS, SAMPLES_PER_FRAME, SAMPLE_RATE,
                      get_pipeline)
from slide_detect import SlideDetector
import diag_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("aspire.app")

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Aspire RT", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path("sessions")
BASE.mkdir(exist_ok=True)

SESSIONS: Dict[str, Session] = {}
_session_locks: Dict[str, threading.Lock] = {}

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".ts"}

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def _lock_for(sid: str) -> threading.Lock:
    if sid not in _session_locks:
        _session_locks[sid] = threading.Lock()
    return _session_locks[sid]


# ─────────────────────────────────────────────────────────────────────────────
# Per-session persistent dedup state
# ─────────────────────────────────────────────────────────────────────────────
def _state_path(session_id: str) -> Path:
    return BASE / session_id / "described_hashes.json"


def _restore_session_state(sess) -> None:
    """Repopulate content_sigs + unique_hashes from disk so dedup survives a
    backend restart within the same session_id."""
    p = _state_path(sess.session_id)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        print(f"[state] could not read {p}: {e}")
        return
    sigs = data.get("content_sigs") or []
    hashes = data.get("unique_hashes") or []
    sess.content_sigs.update(s for s in sigs if isinstance(s, str))
    for hex_h in hashes:
        try:
            sess.unique_hashes.append(imagehash.hex_to_hash(hex_h))
        except (ValueError, TypeError):
            continue
    sess.slide_counter = max(sess.slide_counter, int(data.get("slide_counter") or 0))
    print(f"[state] restored {len(sess.content_sigs)} sigs, "
          f"{len(sess.unique_hashes)} hashes, slide_counter={sess.slide_counter}")


def _persist_session_state(sess) -> None:
    """Snapshot dedup state to disk. Called after a slide is described."""
    p = _state_path(sess.session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "content_sigs": sorted(sess.content_sigs),
        "unique_hashes": [str(h) for h in list(sess.unique_hashes)],
        "slide_counter": sess.slide_counter,
    }
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)
    except OSError as e:
        print(f"[state] write failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Landing page
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aspire — Live Slide Accessibility</title>
<style>
  :root{color-scheme:dark}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
       background:#0a0a0a;color:#eaeaea;margin:0;padding:2rem;max-width:780px;
       margin-inline:auto;line-height:1.6}
  h1{font-size:1.9rem;margin:0 0 .3rem}
  h2{font-size:1.1rem;color:#6bf;margin-top:2.2rem;margin-bottom:.6rem}
  p{margin:.5rem 0}
  a{color:#6bf;text-decoration:none}
  a:hover{text-decoration:underline}
  code,kbd{background:#1a1a1a;border:1px solid #2a2a2a;padding:2px 7px;
           border-radius:5px;font-size:.92em;font-family:ui-monospace,Menlo,monospace}
  .url{display:flex;align-items:center;gap:.5rem;background:#121212;
       border:1px solid #2a2a2a;border-radius:10px;padding:1rem 1.2rem;margin:.6rem 0}
  .url a{font-size:1.15rem;word-break:break-all;flex:1}
  .url button{background:#2b7;color:#000;border:0;padding:.5rem 1rem;
              border-radius:6px;font-weight:600;cursor:pointer;font-size:.9rem}
  .url button:active{background:#1a5}
  .status{display:inline-flex;align-items:center;gap:.4rem;padding:.4rem .8rem;
          background:#1a1a1a;border-radius:20px;font-size:.9rem;color:#aaa}
  .dot{width:.55rem;height:.55rem;border-radius:50%;background:#666}
  .dot.live{background:#2b7;box-shadow:0 0 8px #2b7}
  .muted{color:#888;font-size:.9rem}
  ol{padding-left:1.4rem}
  ol li{margin:.4rem 0}
</style>
</head>
<body>
<h1>Aspire</h1>
<p class="muted">Live slide accessibility for blind and low-vision listeners.</p>

<div style="margin-top:1rem"><span class="status"><span id="dot" class="dot"></span><span id="stat">checking…</span></span></div>

<h2>Listener URL</h2>
<p class="muted">Give this URL to listeners. It never changes — the same link works across sessions.</p>
<div class="url">
  <a id="lurl" href="/listen/live">/listen/live</a>
  <button onclick="copy()">Copy</button>
</div>
<p class="muted" id="lan">Loading LAN address…</p>

<h2>Presenter — start capture</h2>
<p>In Terminal on this Mac:</p>
<p><code>cd ~/Desktop &amp;&amp; python3 system_capture.py</code><br>
<code>cd ~/Desktop &amp;&amp; python3 segment_uploader.py</code></p>
<p class="muted">Keep both running. <kbd>Ctrl+C</kbd> in either terminal to stop. Starting a new capture automatically resets the listener stream.</p>

<h2>Other</h2>
<ul>
  <li>Session status: <a href="/status/live">/status/live</a></li>
  <li>Custom session ID: <code>/listen/&lt;id&gt;</code></li>
  <li>API docs: <a href="/docs">/docs</a></li>
</ul>

<script>
const host = window.location.hostname;
const listenUrl = window.location.protocol + "//" + host + ":" + window.location.port + "/listen/live";
document.getElementById("lurl").href = listenUrl;
document.getElementById("lurl").textContent = listenUrl;

if (host === "127.0.0.1" || host === "localhost") {
  document.getElementById("lan").innerHTML = "For a phone on the same Wi-Fi, open <code>http://192.168.12.78:8000/listen/live</code>";
} else {
  document.getElementById("lan").innerHTML = "Other devices on this network use <code>" + listenUrl + "</code>";
}

function copy() {
  navigator.clipboard.writeText(listenUrl);
  const b = event.target; const t = b.textContent; b.textContent = "Copied!";
  setTimeout(() => b.textContent = t, 1200);
}

async function poll() {
  try {
    const r = await fetch("/status/live", {cache: "no-store"});
    if (r.ok) {
      const j = await r.json();
      const n = (j.ready_chunks || []).length;
      document.getElementById("dot").className = "dot live";
      document.getElementById("stat").textContent = "live — " + n + " chunks processed";
    } else {
      document.getElementById("dot").className = "dot";
      document.getElementById("stat").textContent = "idle — waiting for capture";
    }
  } catch (_) {
    document.getElementById("dot").className = "dot";
    document.getElementById("stat").textContent = "idle — waiting for capture";
  }
}
poll(); setInterval(poll, 3000);
</script>
</body>
</html>""")


# ═════════════════════════════════════════════════════════════════════════════
# NEW: WebRTC live path  (sub-2s; replaces the HLS chunks for the new pipeline)
# ═════════════════════════════════════════════════════════════════════════════
_pcs: Set[RTCPeerConnection] = set()


class PeerSession:
    """Per-peer audio playhead into LivePipeline.ring. Each browser tab gets
    its own; they don't compete for frames."""

    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.playhead: int = -1   # set on first recv() to live
        self.last_seek_at: float = 0.0


_peer_sessions: Dict[str, PeerSession] = {}


def _new_peer_session() -> PeerSession:
    pid = uuid.uuid4().hex[:12]
    sess = PeerSession(pid)
    _peer_sessions[pid] = sess
    return sess


class PerPeerAudioTrack(MediaStreamTrack):
    """Pulls 20ms s16-stereo packed frames from the LivePipeline.ring at the
    peer's playhead. Independent per peer — multiple listeners DO NOT share
    a queue; they read snapshots so 10 students can listen to the same URL
    without overlap. The playhead is what enables YouTube-Live-style scrub
    back: /webrtc/seek moves it earlier in the ring; /webrtc/live snaps it
    to the head."""

    kind = "audio"

    def __init__(self, session: PeerSession):
        super().__init__()
        self._session = session
        self._timestamp = 0
        self._sample_rate = SAMPLE_RATE
        self._samples_per_frame = SAMPLES_PER_FRAME
        self._recv_count = 0
        # Wall-clock deadline for the next frame. asyncio.sleep() always
        # overshoots slightly (~3ms on a 20ms sleep), so pacing with a fixed
        # sleep made the playhead drift ~150ms/s behind the ring head until it
        # tripped the 4s underrun snap — an audible 4-SECOND CUT every ~30s.
        # Pacing to an absolute deadline removes the drift entirely. (Same
        # pattern the mixer uses for its 50fps emit loop.)
        self._next_send: Optional[float] = None

    async def recv(self) -> AudioFrame:
        recv_t0 = time.time()
        pipeline = get_pipeline()
        ring = pipeline.ring
        sess = self._session

        # Pace at wall-clock 50 fps against an ABSOLUTE deadline (not a fixed
        # sleep) so no drift accumulates — see _next_send in __init__.
        period = FRAME_MS / 1000.0
        now_w = time.time()
        if self._next_send is None:
            self._next_send = now_w + period
        else:
            self._next_send += period
        delay = self._next_send - now_w
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.5:
            # Event loop stalled badly — resync rather than spin.
            self._next_send = time.time() + period

        from pipeline import (PEER_INITIAL_LAG_FRAMES, PEER_UNDERRUN_FRAMES,
                              PEER_GENTLE_CATCHUP_FRAMES)

        head = ring.head()

        if sess.playhead < 0:
            sess.playhead = max(0, head - PEER_INITIAL_LAG_FRAMES)

        underrun = False
        behind = head - sess.playhead
        if behind > PEER_UNDERRUN_FRAMES:
            log.warning("[peer/%s] underrun %d frames, snap to live",
                        sess.peer_id, behind)
            sess.playhead = max(0, head - PEER_INITIAL_LAG_FRAMES)
            underrun = True
        elif behind > PEER_GENTLE_CATCHUP_FRAMES:
            # Drifted a little (clock jitter). Skip ONE 20ms frame — inaudible
            # — instead of letting it grow into a 4s snap. Emergency snap above
            # should now essentially never fire.
            sess.playhead += 1

        buf = ring.get(sess.playhead)
        if buf is None:
            sess.playhead = max(0, head - PEER_INITIAL_LAG_FRAMES)
            buf = ring.get(sess.playhead) or ring.silence_frame
            if not buf or len(buf) < self._samples_per_frame * CHANNELS * 2:
                buf = ring.silence_frame
            underrun = True
        elif buf == b"":
            buf = ring.silence_frame
            underrun = True
        else:
            sess.playhead += 1

        frame = AudioFrame(format="s16", layout="stereo",
                           samples=self._samples_per_frame)
        frame.planes[0].update(buf)
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_frame

        # Diagnostics: track recv elapsed time + underruns
        elapsed_ms = (time.time() - recv_t0) * 1000.0
        # Subtract the deliberate 20ms pacing sleep so the metric reflects
        # only "real" work time
        work_ms = max(0.0, elapsed_ms - FRAME_MS)
        rtimes = pipeline.peer_recv_times.setdefault(
            sess.peer_id, deque(maxlen=2000))
        rtimes.append(work_ms)
        if underrun:
            uqueue = pipeline.peer_underruns.setdefault(
                sess.peer_id, deque(maxlen=200))
            uqueue.append(time.time())

        self._recv_count += 1
        # Diag: rolling p50/p95 summary every 50 recvs (≈ 1 s)
        if self._recv_count % 50 == 0:
            recent = list(rtimes)[-50:]
            if recent:
                arr = np.array(recent)
                diag_events.record(
                    "peer", "peer_recv",
                    peer_id=sess.peer_id,
                    recv_count=self._recv_count,
                    p50_ms=round(float(np.median(arr)), 3),
                    p95_ms=round(float(np.quantile(arr, 0.95)), 3),
                    behind_frames=head - sess.playhead,
                    underrun=underrun)
        # Per-peer 30s p50/p95 + underrun count log
        if self._recv_count % 1500 == 0:   # 30s @ 50fps
            recent = list(rtimes)
            if recent:
                arr = np.array(recent[-1500:])
                p50 = float(np.median(arr))
                p95 = float(np.quantile(arr, 0.95))
                ucount = sum(
                    1 for t in pipeline.peer_underruns.get(sess.peer_id, [])
                    if time.time() - t < 30.0
                )
                log.info("[peer %s] recv p50=%.2fms p95=%.2fms "
                         "underruns_in_last_30s=%d",
                         sess.peer_id, p50, p95, ucount)
        if self._recv_count % 250 == 0:
            log.info("[track/%s] recv=%d head=%d playhead=%d (behind=%d)",
                     sess.peer_id, self._recv_count, head,
                     sess.playhead, head - sess.playhead)
        return frame


class ProbeToneTrack(MediaStreamTrack):
    """440 Hz sine at -20 dBFS for end-to-end WebRTC validation."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        from pipeline import make_sine_frame_bytes
        self._make = make_sine_frame_bytes
        self._phase = 0.0
        self._timestamp = 0
        self._sample_rate = SAMPLE_RATE
        self._samples_per_frame = SAMPLES_PER_FRAME
        self._recv_count = 0

    async def recv(self) -> AudioFrame:
        # Pace ourselves at wall-clock 50fps
        await asyncio.sleep(FRAME_MS / 1000.0)
        buf, self._phase = self._make(self._phase, freq_hz=440.0, dbfs=-20.0)
        frame = AudioFrame(format="s16", layout="stereo",
                           samples=self._samples_per_frame)
        frame.planes[0].update(buf)
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_frame
        self._recv_count += 1
        if self._recv_count % 50 == 0:
            log.info("[probe-tone] recv=%d sent", self._recv_count)
        return frame


@app.post("/webrtc/start")
async def webrtc_start():
    pipeline = get_pipeline()
    if not pipeline._running:
        try:
            from tts_kokoro import warm_up
            await asyncio.to_thread(warm_up)
        except Exception as e:
            log.warning("[start] kokoro warm-up failed: %s", e)
        await pipeline.start()
    return {"running": True}


@app.post("/webrtc/stop")
async def webrtc_stop():
    pipeline = get_pipeline()
    await pipeline.stop()
    for pc in list(_pcs):
        try:
            await pc.close()
        except Exception:
            pass
        _pcs.discard(pc)
    return {"running": False}


def _sweep_dead_peers() -> int:
    """Drop peer connections that are already dead from the live set.

    The per-connection close handler normally removes them, but connections
    that die during negotiation can slip past it — measured: 137 offers, only
    9 ever reached `connected`, yet 129 objects stayed registered over a
    6-hour run (growth ~0.36/min). Each stale entry also keeps its
    PerPeerAudioTrack and PeerSession alive.

    This is a belt-and-braces sweep on a state the connection itself reports,
    so it cannot evict a healthy peer. Returns how many were reaped.
    """
    dead = [pc for pc in list(_pcs)
            if pc.connectionState in ("closed", "failed")]
    for pc in dead:
        _pcs.discard(pc)
    if dead:
        log.info("[webrtc] swept %d dead peer connection(s), %d live",
                 len(dead), len(_pcs))
    # Drop peer sessions whose connection is gone, so PerPeerAudioTrack state
    # doesn't accumulate either.
    if dead and _peer_sessions:
        live_ids = {getattr(t, "_session", None) and t._session.peer_id
                    for pc in _pcs for t in pc.getSenders() if t.track}
        for pid in [p for p in list(_peer_sessions) if p not in live_ids]:
            _peer_sessions.pop(pid, None)
    return len(dead)


@app.get("/webrtc/status")
def webrtc_status():
    _sweep_dead_peers()
    return get_pipeline().status() | {"peers": len(_pcs)}


async def _build_answer(pc: RTCPeerConnection, body: dict) -> dict:
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def _attach_close_handler(pc: RTCPeerConnection, label: str) -> None:
    @pc.on("connectionstatechange")
    async def _on_state():
        log.info("[webrtc/%s] peer state=%s", label, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            _pcs.discard(pc)
            # Drop the matching peer session if this was a live (per-peer) PC
            if "/" in label:
                pid = label.rsplit("/", 1)[-1]
                _peer_sessions.pop(pid, None)
                log.info("[peer pid=%s] disconnected (state=%s)",
                         pid, pc.connectionState)
                diag_events.record("peer", "peer_disconnect",
                                   peer_id=pid,
                                   reason=pc.connectionState)
            try:
                await pc.close()
            except Exception:
                pass


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    body = await request.json()
    if "sdp" not in body or "type" not in body:
        raise HTTPException(400, "expected {sdp, type}")

    pipeline = get_pipeline()
    if not pipeline._running:
        try:
            from tts_kokoro import warm_up
            await asyncio.to_thread(warm_up)
        except Exception as e:
            log.warning("[offer] kokoro warm-up failed: %s", e)
        await pipeline.start()

    # Reap anything already dead before registering a new one, so repeated
    # reconnects can't accumulate stale entries.
    _sweep_dead_peers()

    pc = RTCPeerConnection()
    _pcs.add(pc)
    sess = _new_peer_session()
    head_now = pipeline.ring.head()
    from pipeline import PEER_INITIAL_LAG_FRAMES as _peer_lag
    initial_playhead = max(0, head_now - _peer_lag)
    behind_live_ms = max(0, head_now - initial_playhead) * FRAME_MS
    log.info("[peer pid=%s] connected, ring_head=%d "
             "peer_initial_playhead=%d behind_live_ms=%d (%d total)",
             sess.peer_id, head_now, initial_playhead, behind_live_ms,
             len(_pcs))
    diag_events.record("peer", "peer_connected",
                       peer_id=sess.peer_id,
                       ring_head=head_now,
                       initial_playhead=initial_playhead,
                       behind_live_ms=behind_live_ms,
                       total_peers=len(_pcs))
    _attach_close_handler(pc, f"live/{sess.peer_id}")
    pc.addTrack(PerPeerAudioTrack(sess))

    answer = await _build_answer(pc, body)
    answer["peer_id"] = sess.peer_id
    answer["ring_history_s"] = pipeline.ring.capacity * FRAME_MS // 1000
    return answer


def _peer_session(peer_id: str) -> PeerSession:
    sess = _peer_sessions.get(peer_id)
    if not sess:
        raise HTTPException(404, "unknown peer_id")
    return sess


@app.post("/webrtc/seek")
async def webrtc_seek(request: Request):
    """Body: {peer_id, behind_live_s}. Moves that peer's playhead to
    (head - behind_live_s × 50 frames). Negative or zero = live edge.
    Clamped to the ring's oldest frame."""
    body = await request.json()
    peer_id = body.get("peer_id")
    behind = float(body.get("behind_live_s", 0))
    sess = _peer_session(peer_id)
    pipeline = get_pipeline()
    head = pipeline.ring.head()
    oldest = pipeline.ring.oldest()
    target = head - int(behind * (1000 // FRAME_MS))
    target = max(oldest, min(head, target))
    sess.playhead = target
    sess.last_seek_at = time.time()
    log.info("[webrtc/seek] peer=%s → behind=%.1fs (playhead=%d head=%d)",
             peer_id, behind, target, head)
    return {
        "peer_id": peer_id,
        "playhead": target,
        "head": head,
        "behind_live_s": (head - target) * FRAME_MS / 1000.0,
    }


@app.post("/webrtc/live")
async def webrtc_live(request: Request):
    """Snap a peer to live (head)."""
    body = await request.json()
    peer_id = body.get("peer_id")
    sess = _peer_session(peer_id)
    pipeline = get_pipeline()
    sess.playhead = max(0, pipeline.ring.head() - 1)
    sess.last_seek_at = time.time()
    log.info("[webrtc/live] peer=%s snapped to head=%d", peer_id, sess.playhead)
    return {"peer_id": peer_id, "playhead": sess.playhead, "head": pipeline.ring.head()}


@app.get("/webrtc/peer/{peer_id}")
def webrtc_peer_status(peer_id: str):
    """Per-peer status — what the listener UI polls to draw the scrub bar."""
    sess = _peer_session(peer_id)
    pipeline = get_pipeline()
    head = pipeline.ring.head()
    return {
        "peer_id": peer_id,
        "playhead": sess.playhead,
        "head": head,
        "behind_live_s": max(0, (head - sess.playhead) * FRAME_MS / 1000.0),
        "ring_history_s": pipeline.ring.capacity * FRAME_MS // 1000,
    }


@app.post("/webrtc/probe-tone")
async def webrtc_probe_tone(request: Request):
    """Same SDP exchange as /webrtc/offer, but the audio track is a 440Hz
    sine. If the listener hears it, WebRTC delivery is fine and any
    silence on /webrtc/offer is downstream of the track."""
    body = await request.json()
    if "sdp" not in body or "type" not in body:
        raise HTTPException(400, "expected {sdp, type}")
    pc = RTCPeerConnection()
    _pcs.add(pc)
    log.info("[webrtc] new probe-tone peer (%d total)", len(_pcs))
    _attach_close_handler(pc, "probe")
    pc.addTrack(ProbeToneTrack())
    return await _build_answer(pc, body)


@app.post("/webrtc/debug-slide")
async def webrtc_debug_slide(request: Request):
    """Force an arbitrary description through the TTS + mixer path. Lets
    the user verify the pause/play/resume mixer behavior without waiting
    for a real slide change."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "expected {text: ...}")
    pipeline = get_pipeline()
    if not pipeline._running:
        try:
            from tts_kokoro import warm_up
            await asyncio.to_thread(warm_up)
        except Exception as e:
            log.warning("[debug-slide] kokoro warm-up failed: %s", e)
        await pipeline.start()
    asyncio.create_task(pipeline.enqueue_tts_text(text))
    return {"queued": True, "text": text}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic-only endpoint. Listener UI (?diag=1) posts here when the user
# taps "Mark stutter". Writes a user_mark event to the shared JSONL so the
# diag_analyze.py script can correlate ±5 s of activity around each mark.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/diag/user_mark")
async def diag_user_mark(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    ts_iso = (body.get("timestamp_iso") or "").strip()
    note = (body.get("browser_note") or body.get("note") or "").strip()[:240]
    diag_events.record("user", "user_mark",
                       timestamp_iso=ts_iso,
                       browser_note=note)
    return {"recorded": True}


# Alias for the existing listen.html "Mark stutter" button which posts to
# /diagnostics/mark with body {note: "..."}. Same effect — emits user_mark.
@app.post("/diagnostics/mark")
async def diagnostics_mark(request: Request):
    return await diag_user_mark(request)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP audio path — HLS / fMP4-AAC. Native on iOS+macOS Safari; hls.js on
# every other browser. ffmpeg writes live.m3u8 + init.mp4 + seg_NNNNN.m4s
# into sessions/live/audio/.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/audio/live.m3u8")
def audio_playlist():
    from pipeline import AUDIO_SEG_DIR
    p = AUDIO_SEG_DIR / "live.m3u8"
    if not p.exists():
        raise HTTPException(404, "Playlist not ready yet")
    return Response(
        content=p.read_bytes(),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/audio/init.mp4")
def audio_init():
    from pipeline import AUDIO_SEG_DIR
    p = AUDIO_SEG_DIR / "init.mp4"
    if not p.exists():
        raise HTTPException(404, "Init segment not ready yet")
    return FileResponse(
        p, media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/audio/{name}")
def audio_segment(name: str):
    from pipeline import AUDIO_SEG_DIR
    if name == "live.m3u8":
        return audio_playlist()
    if name == "init.mp4":
        return audio_init()
    if not (name.startswith("seg_") and name.endswith(".m4s")):
        raise HTTPException(404, "Not found")
    p = AUDIO_SEG_DIR / name
    if not p.exists():
        raise HTTPException(404, "Segment not found")
    return FileResponse(
        p, media_type="video/iso.segment",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/audio/start")
async def audio_start():
    """Start the pipeline if not running. Lightweight wrapper used by the
    HTTP listener so it doesn't have to know about WebRTC's /webrtc/start."""
    pipeline = get_pipeline()
    if not pipeline._running:
        try:
            from tts_kokoro import warm_up
            await asyncio.to_thread(warm_up)
        except Exception as e:
            log.warning("[audio/start] kokoro warm-up failed: %s", e)
        await pipeline.start()
    return {"running": True}


@app.get("/listen", response_class=HTMLResponse)
def listen_webrtc(request: Request):
    p = Path(__file__).parent / "static" / "listen.html"
    if not p.exists():
        raise HTTPException(404, "listen.html missing")
    return HTMLResponse(p.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — for the "Mark stutter" workflow.
#
#   GET  /diagnostics/snapshot  → point-in-time JSON of every queue, pid,
#                                 mixer state, buffer ms, last-60s jitter
#                                 metrics. Capture this when a stutter
#                                 occurs to correlate with the log lines.
#   POST /diagnostics/mark      → body {"note": "..."} appends a
#                                 [USER_MARK] line to /tmp/aspire-backend.log
#                                 with the wallclock time. The listener UI's
#                                 "Mark stutter" button calls this.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/diagnostics/snapshot")
def diagnostics_snapshot():
    pipeline = get_pipeline()
    snap = pipeline.diagnostic_snapshot()
    snap["webrtc_peers_open"] = len(_pcs)
    return snap


@app.post("/diagnostics/mark")
async def diagnostics_mark(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    note = str(body.get("note") or "stutter (no note)")[:300]
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    log.warning("[USER_MARK] %s %s", ts_iso, note)
    return {"marked": True, "ts": ts_iso, "note": note}


@app.on_event("shutdown")
async def _shutdown():
    pipeline = get_pipeline()
    await pipeline.stop()
    for pc in list(_pcs):
        try:
            await pc.close()
        except Exception:
            pass
        _pcs.discard(pc)


# ─────────────────────────────────────────────────────────────────────────────
# Listener page — hls.js + a single <audio> element
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/listen/{session_id}", response_class=HTMLResponse)
def listen_page(session_id: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aspire Live — {session_id}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin:0; min-height:100vh; display:flex; flex-direction:column;
    align-items:center; justify-content:center; padding:1.5rem;
    background:#0a0a0a; color:#eaeaea;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
    text-align:center;
  }}
  h1 {{ font-size:2rem; margin:0 0 .3rem; }}
  #sid {{ color:#777; font-family:ui-monospace,Menlo,monospace;
          font-size:.85rem; margin-bottom:2rem; }}
  audio {{ width:min(560px,92vw); margin:.5rem 0; }}
  #status {{ margin:1rem 0 .5rem; color:#6bf; font-size:1rem;
             min-height:1.4rem; font-weight:500; }}
  .dot {{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
          background:#666; margin-right:.35rem; vertical-align:middle; }}
  .dot.live {{ background:#2b7; box-shadow:0 0 10px #2b7;
               animation:pulse 2s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
  .hint {{ color:#777; font-size:.9rem; margin-top:2rem; max-width:540px;
           line-height:1.55; }}
  button {{
    background:#2b7; color:#000; border:0; padding:1.1rem 2.4rem;
    border-radius:10px; font-size:1.15rem; cursor:pointer; margin-top:1.5rem;
    font-weight:600; min-width:12rem; min-height:3.2rem;
  }}
  button:focus {{ outline:3px solid #6bf; outline-offset:3px; }}
  button:disabled {{ background:#333; color:#777; cursor:default; }}
</style>
</head>
<body>
  <h1>Aspire Live Audio</h1>
  <div id="sid">session: {session_id}</div>

  <audio id="player" controls aria-label="Live audio stream with slide descriptions"></audio>

  <div id="status" role="status" aria-live="polite" aria-atomic="true">
    <span class="dot" id="dot"></span><span id="statusText">Initializing…</span>
  </div>

  <button id="startBtn" aria-label="Start listening to the live audio stream" disabled>Start listening</button>

  <p class="hint">
    Audio plays continuously with slide descriptions mixed in. Expect about a
    10 second delay behind the presenter. If playback pauses, press the play
    button on the audio control.
  </p>

<script src="https://cdn.jsdelivr.net/npm/hls.js@1"></script>
<script>
const SID = {session_id!r};
const audio     = document.getElementById("player");
const statusEl  = document.getElementById("statusText");
const dotEl     = document.getElementById("dot");
const startBtn  = document.getElementById("startBtn");
const playlistUrl = "/hls/" + SID + "/playlist.m3u8";
let hls = null;

function setStatus(msg, live=false) {{
  statusEl.textContent = msg;
  dotEl.className = "dot" + (live ? " live" : "");
  document.title = (live ? "● " : "") + "Aspire Live — " + SID;
}}

async function waitForPlaylist() {{
  setStatus("Waiting for the presenter to start…");
  while (true) {{
    try {{
      const r = await fetch(playlistUrl + "?t=" + Date.now(), {{ cache: "no-store" }});
      if (r.ok) return;
    }} catch (_) {{}}
    await new Promise(res => setTimeout(res, 1000));
  }}
}}

function attach() {{
  if (Hls.isSupported()) {{
    hls = new Hls({{
      // Allow up to ~30s of buffering before the player jumps forward to catch
      // up to the live edge. This smooths out the drift that accumulates when
      // TTS insertions make composed audio longer than the source chunks.
      liveSyncDuration: 8,
      liveMaxLatencyDuration: 30,
      maxBufferLength: 30,
      backBufferLength: 5,
      manifestLoadingRetryDelay: 500,
      manifestLoadingMaxRetry: 12,
      levelLoadingRetryDelay: 500,
      levelLoadingMaxRetry: 12,
      liveDurationInfinity: true,
    }});
    hls.loadSource(playlistUrl);
    hls.attachMedia(audio);
    hls.on(Hls.Events.MANIFEST_PARSED, () => {{
      setStatus("Ready. Press Start to begin.", true);
      startBtn.disabled = false;
    }});
    hls.on(Hls.Events.ERROR, (_e, d) => {{
      if (d.fatal) {{
        setStatus("Reconnecting…");
        setTimeout(() => {{ try {{ hls.startLoad(-1); }} catch(_) {{}} }}, 600);
      }}
    }});
  }} else if (audio.canPlayType("application/vnd.apple.mpegurl")) {{
    // Safari native HLS
    audio.src = playlistUrl;
    setStatus("Ready. Press Start to begin.", true);
    startBtn.disabled = false;
  }} else {{
    setStatus("This browser does not support HLS audio.");
  }}
}}

startBtn.onclick = async () => {{
  startBtn.disabled = true;
  try {{
    await audio.play();
    setStatus("Playing live", true);
  }} catch (_) {{
    setStatus("Tap the play button on the audio control.");
    startBtn.disabled = false;
  }}
}};

audio.addEventListener("playing", () => setStatus("Playing live", true));
audio.addEventListener("pause", () => setStatus("Paused"));
audio.addEventListener("waiting", () => setStatus("Buffering…"));

(async () => {{ await waitForPlaylist(); attach(); }})();
</script>
</body>
</html>""")


# ─────────────────────────────────────────────────────────────────────────────
# HLS endpoints — disk-backed, no SESSIONS lookup required, so they keep
# working even if the server restarts mid-stream.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/hls/{session_id}/playlist.m3u8")
def hls_playlist(session_id: str):
    p = BASE / session_id / "playlist.m3u8"
    if not p.exists():
        raise HTTPException(404, "Playlist not ready yet")
    return Response(
        content=p.read_bytes(),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/hls/{session_id}/{filename}")
def hls_segment(session_id: str, filename: str):
    # Basic path safety — only allow the exact filenames we write
    if not (filename.startswith("final_") and filename.endswith(".ts")):
        raise HTTPException(404, "Not found")
    p = BASE / session_id / filename
    if not p.exists():
        raise HTTPException(404, "Segment not found")
    return FileResponse(p, media_type="video/mp2t")


# ─────────────────────────────────────────────────────────────────────────────
# /upload_chunk — real-time segment upload
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload_chunk")
def upload_chunk(
    session_id: str  = Form(...),
    chunk_idx: int   = Form(...),
    file: UploadFile = File(...),
):
    ext = os.path.splitext(file.filename or "segment.ts")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported extension: {ext}")

    sdir = BASE / session_id

    # chunk_idx=0 means a fresh capture run. Wipe any prior state for this
    # session so a stable listener URL (e.g. /listen/live) works across runs.
    if chunk_idx == 0:
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        if sdir.exists():
            shutil.rmtree(sdir)
        print(f"[upload_chunk] reset session: {session_id}")

    sdir.mkdir(parents=True, exist_ok=True)

    seg_path = sdir / f"segment_{chunk_idx}{ext}"
    with open(seg_path, "wb") as out:
        out.write(file.file.read())

    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(
            session_id=session_id,
            video_path=str(seg_path),
            duration_ms=0,          # unknown in streaming mode
            num_chunks=chunk_idx + 1,
        )
        # Restore persistent dedup state (no-op on chunk_idx=0 because the
        # session dir was just wiped). Survives backend restarts mid-stream.
        _restore_session_state(SESSIONS[session_id])
        print(f"[upload_chunk] new session: {session_id}")

    sess = SESSIONS[session_id]
    if chunk_idx + 1 > sess.num_chunks:
        sess.num_chunks = chunk_idx + 1

    _ensure_segment_processed(sess, chunk_idx, seg_path)

    return {
        "session_id": session_id,
        "chunk_idx": chunk_idx,
        "status": "queued",
        "listen_url": f"/listen/{session_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# /status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status/{session_id}")
def status(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    ready = sorted(i for i, ok in sess.ready.items() if ok)
    return {
        "ready_chunks": ready,
        "num_chunks": sess.num_chunks,
        "listen_url": f"/listen/{session_id}",
        "playlist_url": f"/hls/{session_id}/playlist.m3u8",
    }


# ─────────────────────────────────────────────────────────────────────────────
# /audio  — used by the prerecorded (/upload) flow only.
#           Kept so your old player and research experiments still work.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/audio/{session_id}/{chunk_idx}")
def get_audio_chunk(session_id: str, chunk_idx: int):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    out_path = BASE / session_id / f"final_{chunk_idx}.mp3"
    if not out_path.exists():
        if sess.duration_ms > 0:      # prerecorded mode
            _ensure_chunk_processed(sess, chunk_idx)
        raise HTTPException(404, "Not ready yet")

    if sess.duration_ms > 0:
        _ensure_chunk_processed(sess, chunk_idx + 1)
    return FileResponse(out_path, media_type="audio/mpeg", filename=out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# /upload — prerecorded full-video mode (kept for batch experiments)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    from moviepy.editor import VideoFileClip   # imported lazily — not needed for streaming

    if not file.filename:
        raise HTTPException(400, "No filename provided")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(400, "Unsupported video type")

    sid = f"rec-{uuid.uuid4().hex[:10]}"
    sdir = BASE / sid
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    video_path = sdir / f"source{ext}"
    with open(video_path, "wb") as out:
        out.write(file.file.read())

    clip = VideoFileClip(str(video_path))
    duration_ms = int(clip.duration * 1000)
    try:
        if clip.reader: clip.reader.close()
        if clip.audio:  clip.audio.reader.close_proc()
    except Exception:
        pass

    n_chunks = plan_chunks(duration_ms)
    sess = Session(session_id=sid, video_path=str(video_path),
                   duration_ms=duration_ms, num_chunks=n_chunks)
    SESSIONS[sid] = sess
    _ensure_chunk_processed(sess, 0)
    return {"session_id": sid, "num_chunks": n_chunks, "duration_ms": duration_ms}


# ═════════════════════════════════════════════════════════════════════════════
# PROCESSING — streaming (segment) path
# ═════════════════════════════════════════════════════════════════════════════
def _ensure_segment_processed(sess: Session, chunk_idx: int, seg_path: Path) -> None:
    if sess.ready.get(chunk_idx) or sess.processing.get(chunk_idx):
        return
    sess.processing[chunk_idx] = True
    threading.Thread(
        target=_process_segment_chunk,
        args=(sess, chunk_idx, seg_path),
        daemon=True,
    ).start()


def _extract_audio(video_path: str, target_ms: int = 0) -> AudioSegment:
    """Extract audio from a video file via a direct ffmpeg subprocess (fast).
    If target_ms is given and the extracted audio is shorter, pad with silence
    so the chunk keeps pace with real time (avfoundation occasionally drops
    a few hundred ms of audio under encoder load)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path, "-vn",
         "-f", "wav", "-ac", "2", "-ar", "44100", "-"],
        capture_output=True, check=True,
    )
    audio = AudioSegment.from_file(BytesIO(proc.stdout), format="wav")
    if target_ms and len(audio) < target_ms:
        audio = audio + AudioSegment.silent(duration=target_ms - len(audio))
    return audio


def _segment_duration_ms(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return 5000
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return int((frames / fps) * 1000) if fps and frames else 5000


async def _process_slide(sess: Session, chunk_idx: int, rel_ms: int,
                         pil: Image.Image, sdir: Path,
                         seen_local: set) -> Tuple[int, Path] | None:
    """Analyze + TTS for a single detected slide. Returns (rel_ms, tts_path) or None."""
    try:
        analysis = await analyze_slide_async(pil)
    except Exception as e:
        print(f"[seg {chunk_idx}] analyze failed @ {rel_ms}ms: {e}")
        return None

    sig = analysis["signature"]
    # Session-level dedup is shared, so we guard with a tiny critical section.
    with _lock_for(sess.session_id):
        if sig in seen_local or sig in sess.content_sigs:
            return None
        seen_local.add(sig)
        sess.content_sigs.add(sig)
        sess.slide_counter += 1
        slide_no = sess.slide_counter
    # Persist dedup state so a backend crash mid-session doesn't re-describe
    # already-announced slides on restart.
    try:
        _persist_session_state(sess)
    except Exception as e:
        print(f"[state] persist failed: {e}")

    text = build_tts_text(
        slide_no=slide_no,
        title=analysis["title"],
        bullets=analysis["bullets"],
        summary=analysis["summary"],
    )
    tts_path = sdir / f"tts_{chunk_idx}_{rel_ms}.mp3"
    try:
        await tts_async(text, tts_path)
    except Exception as e:
        print(f"[seg {chunk_idx}] TTS failed @ {rel_ms}ms: {e}")
        return None
    return (rel_ms, tts_path)


def _process_segment_chunk(sess: Session, chunk_idx: int, seg_path: Path) -> None:
    """Process one standalone segment file (.ts) into final_<N>.ts + playlist update."""
    try:
        sdir = BASE / sess.session_id
        video_path = str(seg_path)
        this_ms = _segment_duration_ms(video_path)
        print(f"[seg {chunk_idx}] {seg_path.name} dur={this_ms}ms")

        # 1) Slide detection (sync, CPU-bound, fast)
        det = SlideDetector(sample_ms=400, change_thresh=10,
                            sustain_frames=2, session=sess)
        slide_starts = det.detect(video_path, start_ms=0, duration_ms=this_ms)

        # 2) Pull the relevant frames as PIL images (still sync)
        frames: List[Tuple[int, Image.Image]] = []
        if slide_starts:
            cap = cv2.VideoCapture(video_path)
            for rel_ms, _ in slide_starts:
                cap.set(cv2.CAP_PROP_POS_MSEC, rel_ms + 300)
                ok, frame = cap.read()
                if not ok:
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append((rel_ms, pil))
            cap.release()

        # 3) Parallel LLM + TTS across all slides in this chunk
        events: List[Tuple[int, Path]] = []
        if frames:
            async def run_all():
                seen = set()
                tasks = [
                    _process_slide(sess, chunk_idx, rel_ms, pil, sdir, seen)
                    for rel_ms, pil in frames
                ]
                return await asyncio.gather(*tasks)

            results = asyncio.run(run_all())
            events = [r for r in results if r is not None]
            events.sort(key=lambda x: x[0])

        # 4) Extract original audio and compose final MP3. Pad to this_ms so
        # the chunk always matches the video duration even if capture dropped
        # some audio — keeps the phone stream in sync with real time.
        try:
            orig_chunk = _extract_audio(video_path, target_ms=this_ms)
        except subprocess.CalledProcessError as e:
            print(f"[seg {chunk_idx}] ffmpeg extract failed: {e}")
            orig_chunk = AudioSegment.silent(duration=this_ms)

        composed = finalize_chunk_audio(
            original_chunk=orig_chunk,
            insert_events=events,
            min_duration_ms=TIME_BUFFER_MS,
            do_filler_removal=False,
        )

        # Write final_<N>.ts (MPEG-TS with AAC) via ffmpeg.
        # pydub gives us WAV; ffmpeg re-encodes it cleanly to a TS segment.
        tmp_wav = sdir / f"_tmp_{chunk_idx}.wav"
        composed.export(tmp_wav, format="wav")
        out_ts = sdir / f"final_{chunk_idx}.ts"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_wav),
             "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
             "-f", "mpegts", str(out_ts)],
            check=True,
        )
        try:
            tmp_wav.unlink()
        except OSError:
            pass

        sess.ready[chunk_idx] = True
        sess.edited_audio_length_ms += len(composed)
        sess.edited_audio_length_list.append(len(composed))
        print(f"[seg {chunk_idx}] done → {out_ts.name} ({len(composed)}ms)")

        _write_playlist(sess.session_id)

        # Clean up the uploaded segment — we don't need it anymore
        try:
            seg_path.unlink()
        except OSError:
            pass

    except Exception as e:
        import traceback
        print(f"[seg {chunk_idx}] ERROR: {e}")
        traceback.print_exc()
    finally:
        sess.processing[chunk_idx] = False


# ─────────────────────────────────────────────────────────────────────────────
# HLS playlist maintenance
# ─────────────────────────────────────────────────────────────────────────────
def _ffprobe_duration_s(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out) if out else 5.0
    except Exception:
        return 5.0


# Keep only this many segments in the live playlist. Phone will stay near
# the live edge instead of replaying the whole archive.
LIVE_WINDOW_SEGMENTS = 4


def _write_playlist(session_id: str) -> None:
    """Rebuild the session's HLS playlist as a LIVE sliding window. Tolerates
    missing chunks (e.g. a single failed segment): the playlist still
    advances, with EXT-X-DISCONTINUITY marking the gap so the player can
    flush its decoder cleanly across the break."""
    sdir = BASE / session_id
    with _lock_for(session_id):
        segments: List[Tuple[int, str, float]] = []
        for p in sdir.glob("final_*.ts"):
            stem = p.stem  # e.g. final_42
            try:
                idx = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            segments.append((idx, p.name, _ffprobe_duration_s(p)))
        segments.sort(key=lambda x: x[0])

        if not segments:
            return

        window = segments[-LIVE_WINDOW_SEGMENTS:]
        first_idx = window[0][0]
        target = max(1, int(max(d for _, _, d in window)) + 1)

        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target}",
            f"#EXT-X-MEDIA-SEQUENCE:{first_idx}",
        ]
        prev_idx = None
        for idx, name, dur in window:
            if prev_idx is not None and idx != prev_idx + 1:
                lines.append("#EXT-X-DISCONTINUITY")
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(name)
            prev_idx = idx
        # No #EXT-X-PLAYLIST-TYPE and no #EXT-X-ENDLIST — this is LIVE.
        (sdir / "playlist.m3u8").write_text("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# PROCESSING — prerecorded (legacy) path. Kept almost identical to v0.2,
# but with do_filler_removal defaulting to False (much faster, rarely worse).
# ═════════════════════════════════════════════════════════════════════════════
def _ensure_chunk_processed(sess: Session, chunk_idx: int) -> None:
    if chunk_idx < 0 or chunk_idx >= sess.num_chunks:
        return
    if sess.ready.get(chunk_idx) or sess.processing.get(chunk_idx):
        return
    sess.processing[chunk_idx] = True
    threading.Thread(target=_process_chunk_prerecorded,
                     args=(sess, chunk_idx), daemon=True).start()


def _process_chunk_prerecorded(sess: Session, chunk_idx: int) -> None:
    from llm_tts import analyze_slide, tts  # sync versions are fine here

    try:
        sdir = BASE / sess.session_id
        video_path = sess.video_path
        start_ms = chunk_idx * CHUNK_MS
        remaining = max(0, sess.duration_ms - start_ms)
        this_ms = min(CHUNK_MS, remaining)
        if this_ms <= 0:
            sess.ready[chunk_idx] = True
            return

        det = SlideDetector(sample_ms=400, change_thresh=10,
                            sustain_frames=3, session=sess)
        slide_starts = det.detect(video_path, start_ms, this_ms)

        events: List[Tuple[int, Path]] = []
        seen_local = set()
        if slide_starts:
            cap = cv2.VideoCapture(video_path)
            for rel_ms, _ in slide_starts:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_ms + rel_ms + 300)
                ok, frame = cap.read()
                if not ok:
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    analysis = analyze_slide(pil)
                except Exception as e:
                    print(f"[chunk {chunk_idx}] analyze failed: {e}")
                    continue
                sig = analysis["signature"]
                if sig in seen_local or sig in sess.content_sigs:
                    continue
                seen_local.add(sig)
                sess.content_sigs.add(sig)
                sess.slide_counter += 1
                text = build_tts_text(
                    slide_no=sess.slide_counter,
                    title=analysis["title"], bullets=analysis["bullets"],
                    summary=analysis["summary"],
                )
                tts_path = sdir / f"tts_{chunk_idx}_{rel_ms}.mp3"
                try:
                    tts(text, tts_path)
                    events.append((rel_ms, tts_path))
                except Exception as e:
                    print(f"[chunk {chunk_idx}] TTS failed: {e}")
            cap.release()

        # Extract full audio once per session
        full_audio_path = sdir / "temp_full_audio.wav"
        if not full_audio_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-vn",
                 "-ac", "2", "-ar", "44100", str(full_audio_path)],
                check=True,
            )
        full = AudioSegment.from_file(full_audio_path, format="wav")
        orig_chunk = full[start_ms: start_ms + this_ms]

        final_audio = finalize_chunk_audio(
            original_chunk=orig_chunk,
            insert_events=events,
            min_duration_ms=TIME_BUFFER_MS,
            do_filler_removal=False,     # ← was True, default now False for speed
        )
        out_path = sdir / f"final_{chunk_idx}.mp3"
        final_audio.export(out_path, format="mp3", bitrate="64k")
        sess.ready[chunk_idx] = True
        sess.edited_audio_length_ms += len(final_audio)
        sess.edited_audio_length_list.append(len(final_audio))
        print(f"[chunk {chunk_idx}] done → {out_path.name}")

    except Exception as e:
        print(f"[chunk {chunk_idx}] ERROR: {e}")
    finally:
        sess.processing[chunk_idx] = False
