# audio_ops.py
from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
from pydub import AudioSegment, silence

# Target loudness in dBFS for both speaker and TTS audio
TARGET_DBFS = -20.0


def normalize(audio: AudioSegment, target_dbfs: float = TARGET_DBFS) -> AudioSegment:
    """
    Normalize audio to a target loudness level so TTS and speaker
    audio play at the same perceived volume.
    """
    if len(audio) == 0:
        return audio
    diff = target_dbfs - audio.dBFS
    # Cap gain to avoid over-amplifying very quiet segments (e.g. silence)
    diff = max(-10.0, min(diff, 15.0))
    return audio.apply_gain(diff)


def pad_to_min_duration(audio: AudioSegment, min_ms: int) -> AudioSegment:
    """Pad with digital silence so total length is at least min_ms."""
    if len(audio) >= min_ms:
        return audio
    return audio + AudioSegment.silent(duration=(min_ms - len(audio)))


def insert_tts_no_overlap(
    original: AudioSegment,
    events: List[Tuple[int, Path]],
) -> AudioSegment:
    """
    Insert TTS clips at exact times; pause the original while TTS plays,
    then resume original from the same time index (no overlap, no content loss).
    Both original and TTS are normalized to the same loudness before mixing.

    NOTE: this function accumulates drift (composed audio grows longer than
    source). Kept for the legacy prerecorded flow. Live flow uses
    overlay_tts_with_ducking below, which is drift-free.
    """
    # Normalize the original speaker audio
    original = normalize(original)

    events_sorted = sorted(events, key=lambda x: x[0])
    out = AudioSegment.silent(duration=0)
    cursor_orig = 0

    for t_ms, tts_path in events_sorted:
        # Original up to insertion point
        out += original[cursor_orig:t_ms]

        # Load and normalize TTS to same level as speaker
        fmt = tts_path.suffix.lower().strip(".") or "mp3"
        tts_seg = AudioSegment.from_file(tts_path, format=fmt)
        tts_seg = normalize(tts_seg)

        out += tts_seg
        cursor_orig = t_ms

    out += original[cursor_orig:]
    return out


def overlay_tts_with_ducking(
    original: AudioSegment,
    events: List[Tuple[int, Path]],
    duck_db: float = -15.0,
) -> AudioSegment:
    """
    Overlay TTS clips on top of presenter audio at the given times. The
    presenter track is ducked by `duck_db` during each TTS window so the
    description is clearly intelligible. Net output length == original length:
    TTS that would extend past the chunk end is truncated. This is what keeps
    live streaming in sync — no drift.
    """
    original = normalize(original)
    chunk_len = len(original)
    if chunk_len == 0:
        return original

    events_sorted = sorted(events, key=lambda x: x[0])

    # Build each TTS segment (normalized + truncated to fit within the chunk)
    prepared: List[Tuple[int, AudioSegment]] = []
    for t_ms, tts_path in events_sorted:
        if t_ms >= chunk_len:
            continue
        fmt = tts_path.suffix.lower().strip(".") or "mp3"
        tts = normalize(AudioSegment.from_file(tts_path, format=fmt))
        max_len = chunk_len - t_ms
        if len(tts) > max_len:
            tts = tts[:max_len]
        prepared.append((t_ms, tts))

    if not prepared:
        return original

    # Duck the presenter during each TTS window (non-overlapping by construction
    # because detected slide start times within a chunk are monotonically increasing).
    out = original
    for t_ms, tts in prepared:
        end_ms = t_ms + len(tts)
        before = out[:t_ms]
        during = out[t_ms:end_ms].apply_gain(duck_db)
        after = out[end_ms:]
        out = before + during + after

    # Overlay the TTS on top of the ducked presenter.
    for t_ms, tts in prepared:
        out = out.overlay(tts, position=t_ms)

    return out


def compress_silence_keep_buffers(
    audio: AudioSegment,
    min_duration_ms: int = 5000,
    silence_thresh_db: int = -38,
    min_sil_ms: int = 250,
    keep_buffer_ms: int = 80,
) -> AudioSegment:
    """Collapse silence but keep small buffers for naturalness. Defaults tuned
    aggressively so live playback can reclaim time spent on TTS: any pause of
    250 ms or more at -38 dBFS or quieter is squeezed to an 80 ms buffer on
    each side. min_duration_ms is a floor — compression stops when the output
    reaches that length, so we never shrink below real-time pace."""
    if len(audio) <= min_duration_ms:
        return audio

    regions = silence.detect_nonsilent(
        audio, min_silence_len=min_sil_ms, silence_thresh=silence_thresh_db
    )
    if not regions:
        return audio

    out = AudioSegment.silent(duration=0)
    prev_end = 0
    for start, end in regions:
        if len(out) + len(audio) - start < min_duration_ms:
            out += audio[prev_end:]
            print(f"compress_silence_keep_buffers() reached min_duration_ms {min_duration_ms}ms")
            break
        start = max(0, start - keep_buffer_ms)
        end   = min(len(audio), end + keep_buffer_ms)
        out  += audio[start:end]
        prev_end = end

    print(f"compress_silence_keep_buffers() {len(audio)}ms -> {len(out)}ms")
    return out


def remove_fillers(audio: AudioSegment, min_duration_ms: int = 5000,
                   fillers: Optional[List[str]] = None) -> AudioSegment:
    """Optional ASR-based filler removal (disabled by default)."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return audio

    fillers = fillers or ["um", "uh", "like", "you know", "ah", "er"]

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio.export(tmp.name, format="wav")
        wav_path = tmp.name

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav_path, word_timestamps=True)
    os.unlink(wav_path)

    cuts: List[Tuple[int, int]] = []
    for seg in segments:
        for w in (seg.words or []):
            token = (w.word or "").strip().lower()
            if token in fillers:
                cuts.append((int(w.start * 1000), int(w.end * 1000)))

    if not cuts:
        return audio

    cuts.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in cuts:
        if not merged or s > merged[-1][1] + 30:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    out = AudioSegment.silent(duration=0)
    cursor = 0
    out_length = len(audio)
    for s, e in merged:
        if out_length - (e - s) < min_duration_ms:
            break
        out += audio[cursor:s]
        cursor = e
        out_length -= (e - s)
    out += audio[cursor:]
    return out


def finalize_chunk_audio(
    original_chunk: AudioSegment,
    insert_events: List[Tuple[int, Path]],
    min_duration_ms: int,
    do_filler_removal: bool = False,
) -> AudioSegment:
    """Overlay TTS on top of the presenter audio with the presenter ducked
    while TTS plays. Output length == input length, so there's no drift —
    the live stream stays glued to real time across long sessions."""
    composed = overlay_tts_with_ducking(original_chunk, insert_events)
    if do_filler_removal:
        composed = remove_fillers(composed, min_duration_ms=min_duration_ms)
    return composed