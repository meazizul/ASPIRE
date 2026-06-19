# Aspire

**Real-time accessibility narration for slide presentations.** Aspire is a
macOS application designed for blind and low-vision (BLV) audiences
listening to remote slide presentations. It captures the presenter's Mac
screen and system audio, automatically detects slide changes, generates
AI-spoken descriptions of new slides, mixes them into the speaker's audio
at sensible moments, and streams the result to listeners' phones over
WebRTC (local Wi-Fi) or HLS (public, via a Cloudflare tunnel). This
repository is the reference implementation of the techniques described in
the DIS 2026 paper *"Enhancing Slide Presentation Accessibility for Blind
and Low-Vision Audiences Through Delay-Buffered Editing"* by Azizul Haque & Jonggi Hong
(DOI: [10.1145/3800645.3812965](https://doi.org/10.1145/3800645.3812965)).

---

## What's Working in This Version

- **ScreenCaptureKit-based audio capture** via a small Swift CLI
  (`tools/systemAudioDump`). Replaces BlackHole and eliminates the audible
  roughness from virtual-audio-driver burst delivery.
- **Play-ASAP descriptions.** Each slide description plays at the next
  natural pause in the speaker's audio (a gap of ≥0.30 s), or after an
  8-second max-hold if no pause appears. Descriptions land in a clean gap
  instead of talking over the presenter.
- **Time reclaim — pause + filler removal.** Every spoken description adds
  seconds to the stream; without reclaiming that time the listener falls
  permanently behind. While catching up, Aspire:
  - **trims long pauses** — silence runs are kept for the first 1 second
    (natural rhythm) and the excess is dropped; and
  - **removes filler words** — a background Whisper pass (`faster-whisper`
    "tiny") flags non-lexical fillers ("um", "uh", "er", …) and the mixer
    drops those frames. Fail-open: if Whisper is unavailable the audio is
    untouched and pause-trimming still runs.
- **Robust slide deduplication.** Beyond perceptual-hash dedup, Aspire
  compares the OCR text of each new slide against recent ones (containment
  ≥70%), so a slide's build-up stages (title-only → title + body) and
  re-detections are described **once**, not two or three times.
- **Automatic slide-region detection.** When the slide is only a small part
  of a wide scene (e.g. a conference-hall shot with the speaker and a small
  projected slide), Aspire clusters the OCR text boxes to find the slide
  rectangle and runs detection + OCR on just that region. Falls back to a
  center crop for full-screen sources; manual override via `ASPIRE_SLIDE_ROI`.
- **WebRTC** for local low-latency listening on the same Wi-Fi network.
- **HLS over Cloudflare tunnel** for remote listeners on any network.
- **Listener dashboard** (`/listen`) — live slide log, now-speaking
  indicator, and pipeline/time-reclaim metrics. Accessible start via
  Ctrl/Cmd-P for BLV listeners.
- **Apple Vision OCR + Claude Haiku 4.5** for slide vision; **Kokoro ONNX**
  (`af_heart` voice at 1.5×) for TTS.

---

## How It Works

```
   Mac screen + system audio
       │
       ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Capture                                                │
   │    Audio:  ScreenCaptureKit via SystemAudioDump CLI     │
   │    Video:  ffmpeg avfoundation                          │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Slide Detection                                        │
   │    auto slide-region (OCR text-box clustering)          │
   │    perceptual hash sampled every 0.4 s                  │
   │    5-second stability gate before commit                │
   │    pHash + OCR-text (containment) dedup                 │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Vision + TTS                                           │
   │    Apple Vision OCR  →  text from slide                 │
   │    Claude Haiku 4.5  →  description (≤20 words)         │
   │    Kokoro ONNX TTS   →  speech (af_heart, 1.5×)         │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Pause-and-Resume Mixer                                 │
   │    play-ASAP: descriptions play at next natural pause   │
   │    time reclaim during catch-up:                        │
   │      • trim pauses  (keep 1 s, drop the rest)           │
   │      • drop filler words  (Whisper "tiny", backlog)     │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Streaming                                              │
   │    WebRTC (local low-latency)                           │
   │    HLS over Cloudflare tunnel (remote public)           │
   └─────────────────────────────────────────────────────────┘
```

---

## Requirements

- **macOS 13+ (Ventura)** — required for ScreenCaptureKit
- **Apple Silicon Mac** recommended
- **Python 3.13** (3.9+ works)
- **Xcode Command Line Tools** (provides `swift` and `git`):
  ```
  xcode-select --install
  ```
- **ffmpeg**:
  ```
  brew install ffmpeg
  ```
- **cloudflared** (optional, for the public URL):
  ```
  brew install cloudflare/cloudflare/cloudflared
  ```
- **Anthropic API key** with Claude Haiku 4.5 access
- **~600 MB disk** for code + ML models. Kokoro ONNX + voices (~350 MB) are
  downloaded during install (step 4 below); the `faster-whisper` "tiny"
  filler-removal model (~75 MB) downloads automatically to the Hugging Face
  cache on first run. If that download fails, filler removal simply stays
  off (fail-open) and pause-trimming still works.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/meazizul/ASPIRE.git
cd ASPIRE

# 2. Python venv + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the audio-capture binary (Swift / ScreenCaptureKit)
cd tools/systemAudioDump
swift build -c release
cp .build/release/SystemAudioDump ./SystemAudioDump
chmod +x ./SystemAudioDump
cd ../..

# 4. Download the Kokoro TTS model (~325 MB ONNX + ~28 MB voices)
mkdir -p models/kokoro
cd models/kokoro
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ../..

# 5. Configure your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...
```

---

## First-Run Permissions

macOS will require **Screen Recording** permission for two binaries:

1. `tools/systemAudioDump/SystemAudioDump` — the Swift CLI that captures
   system audio via ScreenCaptureKit.
2. `ffmpeg` — for capturing the screen as video.

Grant at: **System Settings → Privacy & Security → Screen Recording**. Add
both binaries and toggle them on. On first launch, macOS may prompt
automatically; click *Allow* if it does. You may need to restart your
terminal after granting.

---

## Running

**Step 1.** Set your Mac's sound output to your **normal speakers** —
MacBook speakers, AirPods, etc. **Not BlackHole.** ScreenCaptureKit captures
whatever the current output device is playing, so silencing your speakers
silences the listener.

**Step 2.** *Terminal 1* — start the backend:

```bash
cd ASPIRE
source .venv/bin/activate
ASPIRE_AVFOUNDATION_INPUT=3:0 uvicorn app:app --host 0.0.0.0 --port 8000
```

Verify the startup log shows these lines:

```
[capture] audio source: sck_cli
[capture] video source: avfoundation device 3:0
[mixer] silence reclaim: ON (CATCHUP trims silence runs > 1000ms; anchor gate removed, TTS plays ASAP at next pause)
[filler] filler-word removal: ON (whisper tiny, CATCHUP backlog only)
```

(`[filler] … OFF` is fine too — it just means `faster-whisper` isn't
installed or `ASPIRE_DISABLE_FILLER=1`; pause trimming still runs.)

**Step 3.** *Terminal 2* — start the live pipeline:

```bash
curl -X POST http://127.0.0.1:8000/webrtc/start
```

**Step 4 (optional).** *Terminal 3* — public Cloudflare URL:

```bash
cloudflared tunnel --url http://localhost:8000
```

The tunnel will print a `https://<random>.trycloudflare.com` URL. Share
that with off-network listeners.

---

## Testing

1. Open a lecture video on the Mac — in a browser (YouTube) or in VLC.
   Pick something with **clear, distinct slide changes**: TED talks,
   university lecture recordings, conference talks (DIS, ASSETS, CHI).
2. Open the listener page on the phone:
   - **Local** (same Wi-Fi): `http://<mac-ip>:8000/listen`. Get the Mac's
     IP with `ipconfig getifaddr en0`.
   - **Public**: the `trycloudflare.com` URL from Terminal 3 above.
3. Tap **Listening**. Wait for the **Live** indicator (~10-second
   prebuffer).
4. Start playback on the Mac.

You should hear the presenter's voice plus spoken descriptions when slides
change. Avoid music videos and fast-cut content — the pHash detector is
tuned for slide-paced changes, not video editing.

---

## Configuration

All environment variables live in `.env` (see `.env.example` for the
template).

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key for Claude Haiku slide descriptions. |
| `ASPIRE_AVFOUNDATION_INPUT` | `3:0` | `video_idx:audio_idx` for ffmpeg-avfoundation. If an iPhone is registered via Continuity Camera, the screen index may shift — typically to `4:0`. Find the correct index with: `ffmpeg -hide_banner -f avfoundation -list_devices true -i ""` and look for the `Capture screen 0` line. |
| `ASPIRE_AUDIO_CAPTURE` | `sck_cli` | `sck_cli` = SystemAudioDump via ScreenCaptureKit (default, recommended). `blackhole` = legacy ffmpeg-avfoundation path (requires BlackHole 2ch installed). |
| `ASPIRE_DISABLE_COMPRESSION` | unset | When `1`, disables pause trimming (the 1-second silence floor). Listener lag will grow over time as descriptions play. Debug only. |
| `ASPIRE_DISABLE_FILLER` | unset | When `1`, disables Whisper filler-word removal. Pause trimming still runs. Use if the `faster-whisper` model isn't installed or CPU is constrained. |
| `ASPIRE_SLIDE_ROI` | unset | Manual slide region as `x,y,w,h` fractions (0–1, top-left origin), e.g. `0.55,0.1,0.4,0.45`. Overrides automatic detection. |
| `ASPIRE_DISABLE_SLIDE_ROI` | unset | When `1`, disables automatic slide-region detection and uses the center-crop fallback. |
| `KOKORO_VOICE` / `KOKORO_SPEED` | `af_heart` / `1.5` | TTS voice and speech-rate multiplier. |

---

## Known Limitations

- **Anthropic 529 (overloaded) errors:** roughly 10–30 % of slide-description
  requests may fail when the Anthropic API is under load. The system
  catches the error, rolls back the dedup state for that slide, and skips
  the description (no crash). The next slide change is unaffected.
- **macOS only.** ScreenCaptureKit is an Apple-only framework, so the
  capture path cannot run on Linux or Windows.
- **Listener latency:** typically **15–30 seconds** behind live. The system
  trades latency for smoothness, then claws time back via pause trimming and
  filler removal during catch-up.
- **Filler removal is backlog-only and best-effort.** Fillers are removed
  while the listener is catching up after a description (when a backlog
  exists), not from live pass-through audio with no backlog. The model is
  the small "tiny" Whisper (chosen to protect smoothness on CPU), so
  detection is approximate, not exhaustive. This is the real-time adaptation
  of the paper's offline, whole-stream editing.
- **Single source per instance.** One Aspire backend handles one Mac
  presentation source. Multiple listeners can connect to the same backend.

---

## Architecture Notes

**Why ScreenCaptureKit instead of BlackHole?** An earlier prototype used
BlackHole — a virtual audio driver — as the capture source, piped through
ffmpeg's `avfoundation`. That path had audible roughness from burst
delivery: frames arrived in clumps rather than at a steady rate. Capturing
directly through Apple's native ScreenCaptureKit API (via the small Swift
CLI in `tools/systemAudioDump`) produces clean, regular frames and
eliminates the need for virtual audio cables.

**Why play-ASAP at a natural pause?** An earlier design gated each
description on a buffer "anchor" — it waited until the listener's playback
reached the exact frame where the slide was detected. That kept perfect
sync but added latency and could hold a description for a long time. The
current mixer instead plays a ready description at the next natural pause in
the speaker's audio (a silent gap of ≥0.30 s), or after an 8-second
max-hold if no pause appears. Descriptions still land in a clean gap rather
than over speech, but they reach the listener sooner.

**Why time reclaim (pause trimming + filler removal)?** Every TTS
description adds real seconds of audio to the stream. Without reclaiming
that time, the listener falls permanently behind live. While catching up
(after a description has played and a backlog exists), Aspire reclaims time
two ways. First, **pause trimming**: a silence run plays for its first 1
second — to keep rhythm natural — and the excess is dropped. Second,
**filler-word removal**: a background thread runs `faster-whisper` ("tiny",
int8, CPU) over the backlog with word-level timestamps, flags non-lexical
fillers ("um", "uh", "er", …), and the mixer drops exactly those frames —
the same mechanism it uses for over-long silence. Both run only on the
catch-up backlog, never in the 50 fps mixer loop, and both fail open: if
Whisper is missing or slow, the audio is unaffected. "like" / "you know"
are deliberately *not* removed, since cutting real words changes meaning.

**Why OCR-text dedup in addition to perceptual hashing?** pHash catches
identical frames, but a slide that builds up — title appears, then bullets
fade in — produces visually different frames that are really the *same*
slide. Describing each build stage would narrate the same slide two or
three times. Aspire extracts the slide's OCR text and compares it against
recent slides using a containment coefficient; if a new slide's text is
≥70% contained in (or contains) a recent one, it's treated as the same
slide and skipped. OCR runs on the cropped slide region so volatile
chrome (menu bar, clock, app names) can't defeat the comparison.

**Why automatic slide-region detection?** When the source is a full-screen
slide deck, a center crop is enough. But a real talk is often a wide camera
shot — a hall, the speaker, and a small projected slide off to one side.
There, both slide-change detection and OCR need to focus on just the slide
rectangle. Apple Vision already returns per-line text bounding boxes;
`slide_region.py` clusters them and returns the bounding box of the
dominant text cluster as the slide region. If the cluster fills most of the
frame (a full-screen source) it returns nothing and the proven center-crop
fallback is used. The region is re-estimated every few seconds so it
follows the slide if the camera framing shifts.

---

## Project Layout

```
app.py                 FastAPI backend — routes, WebRTC/HLS endpoints, /listen page
pipeline.py            Core engine — capture orchestration, slide-change detection,
                       the pause-and-resume mixer, and time reclaim (pause trimming
                       + filler-frame dropping)
audio_capture_sck.py   ScreenCaptureKit audio capture (SystemAudioDump + ffmpeg resample)
audio_ops.py           Audio helpers — loudness normalization, silence detection
chunker.py             Splits description text into TTS-sized chunks
slide_detect.py        Perceptual-hash / text-first slide-change detection
slide_region.py        Auto slide-region detection (OCR text-box clustering)   [new]
vision_ocr.py          Apple Vision on-device OCR wrapper
vision_haiku.py        Claude Haiku 4.5 slide-description wrapper
tts_kokoro.py          Kokoro ONNX text-to-speech (af_heart, 1.5×)
llm_tts.py             Vision + TTS orchestration helpers
filler_removal.py      Whisper ("tiny") filler-word detection for time reclaim  [new]
diag_events.py         Diagnostic event writer (JSONL)
diag_analyze.py        Post-run diagnostic analyzer
static/listen.html     Listener dashboard (slide log, now-speaking, metrics)
tools/systemAudioDump/ Swift CLI — ScreenCaptureKit system-audio capture
```

---

## Research Context

Implementation of techniques described in:

> Haque, A., & Hong, J. (2026). *Enhancing Slide Presentation Accessibility
> for Blind and Low-Vision Audiences Through Delay-Buffered Editing.*
> Proceedings of the ACM Conference on Designing Interactive Systems (DIS
> 2026). DOI: [10.1145/3800645.3812965](https://doi.org/10.1145/3800645.3812965)

The DIS 2026 paper details the formative user study (n = 12 BLV
participants) that motivated this design, and the underlying delay-
buffered editing technique. Prior work: the ASSETS 2025 poster on Wizard-
of-Oz delay-buffered editing.

---

## Contact

**Azizul Haque**
Stevens Institute of Technology
Interactive Computing Lab (ICLAB)
Advisor: Dr. Jonggi Hong
Email: <ahaque3@stevens.edu>
