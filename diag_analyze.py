#!/usr/bin/env python3
"""Reads /tmp/aspire-diag-events.jsonl and produces the deep diagnostic
report described in Phase 2/4 of the diagnostic prompt.

Usage:
    python diag_analyze.py [path/to/aspire-diag-events.jsonl]

Defaults to /tmp/aspire-diag-events.jsonl. Prints to stdout only — does
not modify the pipeline or any source files.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, Iterator, List, Tuple

DEFAULT_LOG = Path("/tmp/aspire-diag-events.jsonl")


def load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def histogram(values: List[float], buckets: List[Tuple[str, float, float]]) -> List[Tuple[str, int]]:
    out = []
    for label, lo, hi in buckets:
        n = sum(1 for v in values if lo <= v < hi)
        out.append((label, n))
    return out


def summarize_event_counts(events: List[Dict[str, Any]]) -> None:
    print("\n── Event counts ──")
    c = Counter((e.get("source", "?"), e.get("event", "?")) for e in events)
    width = max(len(s) + len(ev) + 2 for (s, ev), _ in c.items()) if c else 24
    for (src, ev), n in c.most_common():
        print(f"  {src}.{ev:<{width}}  {n:>8}")


def summarize_audio_capture(events: List[Dict[str, Any]]) -> None:
    print("\n── Audio capture (BlackHole input) ──")
    deltas = [float(e.get("since_last_read_ms", 0.0))
              for e in events
              if e.get("event") == "audio_capture_read"
              and e.get("since_last_read_ms", 0) > 0]
    if not deltas:
        print("  no audio_capture_read events recorded")
        return
    underruns = sum(1 for e in events if e.get("event") == "audio_capture_underrun")
    expected = 20.0  # ms per 20-ms frame
    in_window = sum(1 for d in deltas if 5.0 <= d <= 25.0)
    over30 = sum(1 for d in deltas if d > 30.0)
    print(f"  reads: {len(deltas)}    underruns: {underruns}")
    print(f"  since_last_read_ms  p50={percentile(deltas,0.5):.1f}  "
          f"p95={percentile(deltas,0.95):.1f}  "
          f"p99={percentile(deltas,0.99):.1f}  "
          f"max={max(deltas):.1f}")
    print(f"  reads in expected window (5-25 ms): "
          f"{in_window}/{len(deltas)} = {100*in_window/len(deltas):.1f}%")
    print(f"  reads delayed (>30 ms): {over30}")
    buckets = [
        ("<5",      0.0,   5.0),
        ("5-15",    5.0,  15.0),
        ("15-25",  15.0,  25.0),
        ("25-40",  25.0,  40.0),
        ("40-100", 40.0, 100.0),
        (">100",  100.0, 1e9),
    ]
    print("  histogram of since_last_read_ms:")
    for label, n in histogram(deltas, buckets):
        bar = "█" * min(60, n * 60 // max(1, len(deltas)))
        print(f"    {label:>8}  {n:>7}  {bar}")


def summarize_queue(events: List[Dict[str, Any]]) -> None:
    print("\n── Mixer queues ──")
    gets = [e for e in events if e.get("event") == "speaker_q_get_attempt"]
    if gets:
        miss = sum(1 for e in gets if not e.get("got"))
        print(f"  speaker_q_get_attempt: {len(gets)}    "
              f"missed (got=false): {miss} = {100*miss/len(gets):.2f}%")
    puts = [e for e in events if e.get("event") == "speaker_q_put"]
    if puts:
        sizes = [int(e.get("q_size_after", 0)) for e in puts]
        print(f"  speaker_q_put: {len(puts)}    "
              f"q_size_after p50={percentile(sizes,0.5):.0f}  "
              f"p95={percentile(sizes,0.95):.0f}  max={max(sizes)}")
    tts_puts = sum(1 for e in events if e.get("event") == "tts_q_put")
    tts_gets = sum(1 for e in events if e.get("event") == "tts_q_get")
    print(f"  tts_q_put: {tts_puts}    tts_q_get: {tts_gets}")


def summarize_mixer(events: List[Dict[str, Any]]) -> None:
    print("\n── Mixer state machine ──")
    ticks = [e for e in events if e.get("event") == "mixer_tick"]
    if not ticks:
        print("  no mixer_tick events recorded")
        return
    state_counts: Counter[str] = Counter(t.get("state", "?") for t in ticks)
    src_counts: Counter[str] = Counter(t.get("used_source", "?") for t in ticks)
    print(f"  total ticks: {len(ticks)}")
    print("  state distribution:")
    for s, n in state_counts.most_common():
        print(f"    {s:<18}  {n:>7}  {100*n/len(ticks):5.1f}%")
    print("  used_source distribution:")
    for s, n in src_counts.most_common():
        print(f"    {s:<18}  {n:>7}  {100*n/len(ticks):5.1f}%")
    # Silence pad during LIVE state — the smoking gun for capture starvation.
    live_ticks = [t for t in ticks if t.get("state") == "LIVE"]
    live_silence = sum(1 for t in live_ticks if t.get("used_source") == "silence")
    if live_ticks:
        rate = 100 * live_silence / len(live_ticks)
        flag = " ← SMOKING GUN" if rate > 2.0 else ""
        print(f"  silence pad while LIVE: {live_silence}/{len(live_ticks)} "
              f"= {rate:.2f}%{flag}")
    took = [int(t.get("took_us", 0)) for t in ticks]
    if took:
        print(f"  mixer_tick took_us  "
              f"p50={percentile(took,0.5):.0f}  "
              f"p95={percentile(took,0.95):.0f}  "
              f"p99={percentile(took,0.99):.0f}  "
              f"max={max(took)}")
    changes = [e for e in events if e.get("event") == "mixer_state_change"]
    print(f"  state changes: {len(changes)}")


def summarize_compression(events: List[Dict[str, Any]]) -> None:
    print("\n── Silence compression ──")
    # Newer pipeline emits ON / OFF in mixer_tick.compression_mode. Older
    # JSONLs used NATURAL / GENTLE / AGGRESSIVE — we surface whichever shape
    # is in the file.
    ticks = [e for e in events if e.get("event") == "mixer_tick"
             and e.get("compression_mode")]
    legacy_modes = {"NATURAL", "GENTLE", "AGGRESSIVE"}
    is_legacy = any(t.get("compression_mode") in legacy_modes for t in ticks)

    if ticks:
        mode_counts: Counter[str] = Counter(
            t.get("compression_mode", "?") for t in ticks)
        print(f"  ticks with mode field: {len(ticks)}")
        if is_legacy:
            print("  (legacy 3-regime data)")
            for mode in ("NATURAL", "GENTLE", "AGGRESSIVE"):
                n = mode_counts.get(mode, 0)
                bar = "█" * min(40, n * 40 // max(1, len(ticks)))
                print(f"    {mode:<12}  {n:>7}  "
                      f"{100*n/len(ticks):5.1f}%  {bar}")
        else:
            for mode in ("ON", "OFF"):
                n = mode_counts.get(mode, 0)
                bar = "█" * min(40, n * 40 // max(1, len(ticks)))
                print(f"    {mode:<12}  {n:>7}  "
                      f"{100*n/len(ticks):5.1f}%  {bar}")

    # New: silence_compressed events (one per closed silent run with drops).
    compressed_runs = [e for e in events
                       if e.get("event") == "silence_compressed"]
    if compressed_runs:
        total_dropped = sum(int(e.get("dropped_ms", 0))
                            for e in compressed_runs)
        total_kept = sum(int(e.get("kept_ms", 0)) for e in compressed_runs)
        total_run_ms = sum(int(e.get("run_total_ms", 0))
                           for e in compressed_runs)
        run_lens = [int(e.get("run_total_ms", 0))
                    for e in compressed_runs]
        drop_lens = [int(e.get("dropped_ms", 0))
                     for e in compressed_runs]
        print(f"\n  silence_compressed runs: {len(compressed_runs)}")
        print(f"    total run time:     {total_run_ms/1000:.1f}s")
        print(f"    total kept (≤2s):   {total_kept/1000:.1f}s")
        print(f"    total reclaimed:    {total_dropped/1000:.1f}s")
        print(f"    run length    p50={percentile(run_lens, 0.5):.0f}ms  "
              f"p95={percentile(run_lens, 0.95):.0f}ms  "
              f"max={max(run_lens)}ms")
        print(f"    dropped/run   p50={percentile(drop_lens, 0.5):.0f}ms  "
              f"p95={percentile(drop_lens, 0.95):.0f}ms  "
              f"max={max(drop_lens)}ms")
    else:
        print("\n  silence_compressed runs: 0 "
              "(no silence runs exceeded the 2s keep-floor)")

    # Legacy compression_mode_change events still surfaced if present.
    transitions = [e for e in events
                   if e.get("event") == "compression_mode_change"]
    if transitions:
        t0 = float(events[0].get("ts", 0))
        print(f"\n  legacy compression_mode_change events: {len(transitions)}")
        for tr in transitions[:5]:
            rel = float(tr.get("ts", 0)) - t0
            print(f"    T+{rel:7.1f}s  {tr.get('from_mode')}"
                  f" → {tr.get('to_mode')}"
                  f"  buffer={int(tr.get('buffer_ms', 0))}ms")


def summarize_detect_vision_tts(events: List[Dict[str, Any]]) -> None:
    print("\n── Slide-description pipeline ──")
    cand_commits = [e for e in events if e.get("event") == "candidate_committed"]
    haiku_starts = [e for e in events if e.get("event") == "haiku_request_start"]
    haiku_ends = [e for e in events if e.get("event") == "haiku_request_end"
                  and not e.get("error")]
    haiku_errors = [e for e in events if e.get("event") == "haiku_request_end"
                    and e.get("error")]
    kokoro_starts = [e for e in events if e.get("event") == "kokoro_synth_start"]
    kokoro_ends = [e for e in events if e.get("event") == "kokoro_synth_end"
                   and not e.get("error")]
    kokoro_errors = [e for e in events if e.get("event") == "kokoro_synth_end"
                     and e.get("error")]
    tts_puts = [e for e in events if e.get("event") == "tts_q_put"]
    tts_gets = [e for e in events if e.get("event") == "tts_q_get"]
    print(f"  candidate_committed:   {len(cand_commits)}")
    print(f"  haiku_request_start:   {len(haiku_starts)}")
    print(f"  haiku_request_end ok:  {len(haiku_ends)}   "
          f"errors: {len(haiku_errors)}")
    print(f"  kokoro_synth_start:    {len(kokoro_starts)}")
    print(f"  kokoro_synth_end ok:   {len(kokoro_ends)}   "
          f"errors: {len(kokoro_errors)}")
    print(f"  tts_q_put:             {len(tts_puts)}")
    print(f"  tts_q_get:             {len(tts_gets)}")
    if cand_commits and not haiku_starts:
        print("  VERDICT: breaks at candidate→haiku — haiku_request never fires "
              "for committed candidates.")
    elif haiku_starts and len(haiku_ends) < len(haiku_starts):
        print("  VERDICT: haiku requests hang / error — "
              f"{len(haiku_starts)-len(haiku_ends)-len(haiku_errors)} unfinished, "
              f"{len(haiku_errors)} errored")
    elif haiku_ends and not kokoro_starts:
        print("  VERDICT: breaks between haiku and kokoro_synth_start")
    elif kokoro_starts and not kokoro_ends:
        print("  VERDICT: kokoro_synth never completes — model load or "
              "synthesis error")
    elif kokoro_ends and not tts_puts:
        print("  VERDICT: synth completes but tts_q_put never fires — "
              "queue full or exception after synth")
    elif tts_puts and not tts_gets:
        print("  VERDICT: clips in queue but mixer never picks them up")
    elif tts_gets:
        print(f"  VERDICT: full pipeline completed for {len(tts_gets)} clips")
    else:
        print("  VERDICT: no slide-description activity recorded")


def summarize_encoder(events: List[Dict[str, Any]]) -> None:
    print("\n── Encoder output (HLS) ──")
    writes = [e for e in events if e.get("event") == "encoder_write_attempt"]
    drops = [e for e in events if e.get("event") == "encoder_dropped"]
    resp = [e for e in events if e.get("event") == "encoder_respawn"]
    hls = [e for e in events if e.get("event") == "hls_segment_written"]
    if writes:
        blocked = [int(w.get("blocked_us", 0)) for w in writes]
        print(f"  encoder_write_attempt: {len(writes)}    "
              f"blocked_us  p50={percentile(blocked,0.5):.0f}  "
              f"p95={percentile(blocked,0.95):.0f}  "
              f"p99={percentile(blocked,0.99):.0f}  "
              f"max={max(blocked)}")
    print(f"  encoder_dropped:       {len(drops)}")
    if drops:
        reasons = Counter(d.get("reason", "?") for d in drops)
        for r, n in reasons.most_common():
            print(f"    reason={r!r}: {n}")
    print(f"  encoder_respawn:       {len(resp)}    "
          + ("target=0 OK" if not resp else "← non-zero, encoder died"))
    if hls:
        deltas = [float(s.get("since_last_seg_ms", 0))
                  for s in hls
                  if float(s.get("since_last_seg_ms", 0)) > 0]
        if deltas:
            late = sum(1 for d in deltas if d > 5000)
            print(f"  hls_segment_written: {len(hls)}    "
                  f"since_last_seg_ms  p50={percentile(deltas,0.5):.0f}  "
                  f"p95={percentile(deltas,0.95):.0f}  "
                  f"p99={percentile(deltas,0.99):.0f}  "
                  f"max={max(deltas):.0f}")
            print(f"  late segments (>5000 ms): {late}")


def summarize_peers(events: List[Dict[str, Any]]) -> None:
    print("\n── Peer delivery (WebRTC tracks) ──")
    conns = [e for e in events if e.get("event") == "peer_connected"]
    discs = [e for e in events if e.get("event") == "peer_disconnect"]
    recvs = [e for e in events if e.get("event") == "peer_recv"]
    print(f"  peer_connected:    {len(conns)}    peer_disconnect: {len(discs)}")
    if recvs:
        p50s = [float(r.get("p50_ms", 0)) for r in recvs]
        p95s = [float(r.get("p95_ms", 0)) for r in recvs]
        underruns = sum(1 for r in recvs if r.get("underrun"))
        print(f"  peer_recv samples: {len(recvs)}    "
              f"underrun frames in window: {underruns}")
        print(f"  rolling p50 of recv work_ms: "
              f"p50={percentile(p50s,0.5):.2f}  "
              f"p95={percentile(p50s,0.95):.2f}  "
              f"max={max(p50s):.2f}")
        print(f"  rolling p95 of recv work_ms: "
              f"p50={percentile(p95s,0.5):.2f}  "
              f"p95={percentile(p95s,0.95):.2f}  "
              f"max={max(p95s):.2f}")


def summarize_anchor_gating(events: List[Dict[str, Any]]) -> None:
    print("\n── Buffer-anchored TTS gating ──")
    commits = [e for e in events if e.get("event") == "candidate_committed"]
    reaches = [e for e in events if e.get("event") == "slide_anchor_reached"]
    promos = [e for e in events if e.get("event") == "slide_promotion"]
    giveups = [e for e in events if e.get("event") == "slide_anchor_giveup"]

    print(f"  candidate_committed:   {len(commits)}")
    print(f"  slide_anchor_reached:  {len(reaches)}")
    print(f"  slide_promotion:       {len(promos)}")
    print(f"  slide_anchor_giveup:   {len(giveups)}")

    if reaches:
        waits = [int(e.get("anchor_wait_ms", 0)) for e in reaches]
        print(f"  anchor_wait_ms  p50={percentile(waits,0.5):.0f}  "
              f"p95={percentile(waits,0.95):.0f}  "
              f"max={max(waits)}  "
              f"min={min(waits)}")

    if promos:
        triggers = Counter(p.get("trigger") for p in promos)
        print("  trigger distribution:")
        for t, n in triggers.most_common():
            print(f"    {t:<18} {n:>4} ({100*n/len(promos):.0f}%)")
        anchor_waits = [int(p.get("anchor_wait_ms", 0)) for p in promos]
        pause_waits  = [int(p.get("pause_wait_ms", 0))  for p in promos]
        print(f"  anchor_wait_ms  p50={percentile(anchor_waits,0.5):.0f}  "
              f"max={max(anchor_waits)}")
        print(f"  pause_wait_ms   p50={percentile(pause_waits,0.5):.0f}  "
              f"max={max(pause_waits)}")

    if giveups:
        print("  giveup events (each is a safety-net force-promote):")
        t0 = float(events[0].get("ts", 0))
        for g in giveups:
            print(f"    T+{float(g.get('ts',0))-t0:7.1f}s  slide#{g.get('slide_no')}  "
                  f"anchor={g.get('anchor_pos')}  read={g.get('mixer_read_pos')}  "
                  f"gap={g.get('gap_frames')}  wait={g.get('total_wait_ms')}ms")

    # Per-slide candidate_committed → first emitted-PCM-of-that-slide latency.
    # The mixer's tts_promoted_to_active is when the first frame of that
    # slide's TTS becomes the active output, i.e. the listener starts
    # hearing it. Compare to the slide's commit_time.
    by_slide: Dict[int, Dict[str, float]] = defaultdict(dict)
    for e in events:
        ev = e.get("event")
        sn = e.get("slide_no")
        if sn is None:
            continue
        if ev == "candidate_committed":
            by_slide[sn]["commit"] = float(e.get("ts", 0))
        elif ev == "slide_anchor_reached":
            by_slide[sn].setdefault("anchor_reached", float(e.get("ts", 0)))
        elif ev == "slide_promotion":
            by_slide[sn].setdefault("promoted", float(e.get("ts", 0)))
    if by_slide:
        print("\n  per-slide latency (commit → anchor_reached → promoted):")
        for sn in sorted(by_slide):
            d = by_slide[sn]
            commit = d.get("commit")
            reach  = d.get("anchor_reached")
            prom   = d.get("promoted")
            if commit is None:
                continue
            reach_ms = int((reach - commit) * 1000) if reach else None
            prom_ms = int((prom - commit) * 1000) if prom else None
            reach_s = f"reach=+{reach_ms}ms" if reach_ms is not None else "reach=---"
            prom_s = f"promote=+{prom_ms}ms" if prom_ms is not None else "promote=---"
            print(f"    slide#{sn:<3}  {reach_s:<20}  {prom_s}")


def _active_compression_mode_at(sorted_events: List[Dict[str, Any]],
                                t: float) -> str:
    """Most recent compression_mode_change.to_mode at or before time t,
    falling back to 'NATURAL' if none seen yet."""
    last = "NATURAL"
    for e in sorted_events:
        if float(e.get("ts", 0)) > t:
            break
        if e.get("event") == "compression_mode_change":
            last = e.get("to_mode", last)
    return last


def correlate_user_marks(events: List[Dict[str, Any]]) -> None:
    print("\n── User stutter marks — ±5 s correlation ──")
    marks = [e for e in events if e.get("event") == "user_mark"]
    if not marks:
        print("  no user_mark events recorded")
        return
    print(f"  total marks: {len(marks)}\n")
    sorted_events = sorted(events, key=lambda e: e.get("ts", 0))
    t0_session = sorted_events[0]["ts"] if sorted_events else 0
    for i, m in enumerate(marks, 1):
        t = float(m.get("ts", 0))
        rel = t - t0_session
        window = [e for e in sorted_events
                  if abs(float(e.get("ts", 0)) - t) <= 5.0
                  and e.get("event") != "user_mark"]
        by_src: Dict[str, Counter] = defaultdict(Counter)
        for e in window:
            by_src[e.get("source", "?")][e.get("event", "?")] += 1
        # Active compression mode at the moment the user tapped
        active_mode = _active_compression_mode_at(sorted_events, t)
        print(f"  MARK #{i}  T+{rel:.1f}s  "
              f"compression={active_mode}  "
              f"note={m.get('browser_note','')!r}")
        # Within ±5s, summarize by source
        for src in sorted(by_src):
            counts = by_src[src]
            head = ", ".join(f"{ev}={n}" for ev, n in counts.most_common(6))
            print(f"    [{src}]  {head}")
        # Targeted micro-window: ±100ms around the mark for tick details
        micro = [e for e in window
                 if abs(float(e.get("ts", 0)) - t) <= 0.10
                 and e.get("event") in
                 ("mixer_tick", "encoder_dropped", "encoder_respawn",
                  "audio_capture_underrun", "speaker_q_get_attempt")]
        if micro:
            mt = [e for e in micro if e.get("event") == "mixer_tick"]
            sg = sum(1 for e in mt
                     if e.get("state") == "LIVE"
                     and e.get("used_source") == "silence")
            miss = sum(1 for e in micro
                       if e.get("event") == "speaker_q_get_attempt"
                       and not e.get("got"))
            micro_modes = Counter(t.get("compression_mode") for t in mt
                                  if t.get("compression_mode"))
            mode_str = " ".join(f"{m}={n}" for m, n in micro_modes.most_common()
                                ) or "n/a"
            print(f"    [micro ±100ms]  mixer_ticks={len(mt)}  "
                  f"live_silence_pads={sg}  speaker_q_misses={miss}  "
                  f"modes_in_micro: {mode_str}")
        print()


def main(argv: List[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    if not path.exists():
        print(f"No event log at {path}", file=sys.stderr)
        return 2
    events = load_events(path)
    if not events:
        print(f"Event log {path} is empty", file=sys.stderr)
        return 2

    duration_s = float(events[-1]["ts"]) - float(events[0]["ts"])
    print("═" * 64)
    print("DEEP DIAGNOSTIC REPORT")
    print("═" * 64)
    print(f"  log file:         {path}")
    print(f"  events recorded:  {len(events)}")
    print(f"  duration:         {duration_s:.1f} s "
          f"({duration_s/60:.2f} min)")

    summarize_event_counts(events)
    summarize_detect_vision_tts(events)
    summarize_anchor_gating(events)
    summarize_audio_capture(events)
    summarize_queue(events)
    summarize_mixer(events)
    summarize_compression(events)
    summarize_encoder(events)
    summarize_peers(events)
    correlate_user_marks(events)

    print("\n" + "═" * 64)
    print("END OF REPORT — interpretation is up to the operator.")
    print("═" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
