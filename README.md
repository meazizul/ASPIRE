# Aspire

**Live Accessibility Narration for Slide Presentations**

Aspire is a real-time system for blind and low-vision (BLV) audiences
listening to remote slide presentations. It captures the presenter's screen
and system audio, automatically detects when slides change, generates
spoken descriptions using AI vision models, and streams the mixed audio to
listeners' phones.

This repository is the reference implementation of the techniques described
in the DIS 2026 paper *"Enhancing Slide Presentation Accessibility for
Blind and Low-Vision Audiences Through Delay-Buffered Editing"* by Hakim &
Hong (2026), DOI: [10.1145/3800645.3812965](https://doi.org/10.1145/3800645.3812965).
The system builds on the ASSETS 2025 poster on Wizard-of-Oz delay-buffered
editing.

---

## How It Works

```
   ┌─────────────────────────────────────────────────────────┐
   │  CAPTURE                                                │
   │   audio  →  SystemAudioDump (Swift / ScreenCaptureKit)  │
   │   video  →  ffmpeg avfoundation                         │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SLIDE DETECTION                                        │
   │   perceptual hash (pHash) sampled every 400 ms          │
   │   5-second stability gate before commit                 │
   └────────────────────────────┬────────────────────────────┘
                                │  on slide change
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  VISION + TTS                                           │
   │   Apple Vision OCR  →  extract slide text               │
   │   Claude Haiku 4.5  →  spoken description (≤20 words)   │
   │   Kokoro ONNX TTS   →  synthesize speech (24 kHz)       │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  PAUSE-AND-RESUME MIXER                                 │
   │   buffer-anchored descriptions                          │
   │     (TTS plays only after listener reaches the slide)   │
   │   smart silence compression (>2 s runs)                 │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │  STREAMING                                              │
   │   WebRTC (low-latency)  +  HLS over Cloudflare tunnel   │
   └─────────────────────────────────────────────────────────┘
```

---

## Requirements

- macOS 13 or newer (ScreenCaptureKit dependency). Apple Silicon recommended.
- Python 3.13 (3.9+ should work).
- Xcode Command Line Tools (provides `swift`, `git`):
  ```
  xcode-select --install
  ```
- ffmpeg:
  ```
  brew install ffmpeg
  ```
- Optional: `cloudflared` for exposing a public listener URL.
  ```
  brew install cloudflared
  ```
- An Anthropic API key (for Claude Haiku slide descriptions).

---

## Installation

```bash
# 1. Clone
git clone https://github.com/InteractiveComputingLab/ASPIRE2.0.git
cd ASPIRE2.0

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

macOS will require **Screen Recording** permission for two things:

1. `tools/systemAudioDump/SystemAudioDump` — the Swift binary that captures
   system audio via ScreenCaptureKit.
2. `ffmpeg` (or Terminal, depending on how you launch) — for the video
   screen capture.

Path: **System Settings → Privacy & Security → Screen Recording**. Add both
binaries and toggle them on. On first launch, macOS may prompt
automatically; click *Allow* if it does.

---

## Running

Make sure your **Mac sound output is your normal speakers** (MacBook
speakers, AirPods, etc.) — **not BlackHole**. ScreenCaptureKit captures the
output device, so silencing your speakers will silence the listener too.

**Terminal 1** — start the backend:

```bash
source .venv/bin/activate
ASPIRE_AVFOUNDATION_INPUT=3:0 uvicorn app:app --host 0.0.0.0 --port 8000
```

**Terminal 2** — start the live pipeline:

```bash
curl -X POST http://127.0.0.1:8000/webrtc/start
```

**Terminal 3 (optional)** — public URL via Cloudflare quick tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

The tunnel will print a `https://<random>.trycloudflare.com` URL — share
that with off-network listeners.

---

## Testing

1. Open a presentation source on the Mac: YouTube lecture, recorded talk in
   VLC, Keynote slideshow with audio commentary, etc.
2. On a listener phone, open:
   - **Local** (same Wi-Fi): `http://<mac-ip>:8000/listen` (find your Mac's
     IP with `ipconfig getifaddr en0`)
   - **Public**: the Cloudflare URL from above.
3. Tap **Listening** on the phone page, wait for the **Live** indicator
   (~10-second prebuffer), then start playback on the Mac.

Recommended content for testing:
- TED talks with slides
- University lecture recordings
- Conference talks (DIS, ASSETS, CHI, etc.)

Avoid music videos and fast-cut visual content — the pHash detector is
tuned for slide-paced changes, not video editing.

---

## Configuration

All environment variables live in `.env` (see `.env.example` for the
template).

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key for Claude Haiku slide-description calls. |
| `ASPIRE_AVFOUNDATION_INPUT` | `3:0` | `video_idx:audio_idx` for ffmpeg-avfoundation. If iPhone is connected via Continuity Camera, may need `4:0`. Find correct index with `ffmpeg -hide_banner -f avfoundation -list_devices true -i ""`. |
| `ASPIRE_AUDIO_CAPTURE` | `sck_cli` | `sck_cli` = SystemAudioDump via ScreenCaptureKit (default, recommended). `blackhole` = legacy ffmpeg-avfoundation path (requires BlackHole 2ch installed). |
| `ASPIRE_DISABLE_COMPRESSION` | unset | When `1`, disables silence compression. Listener lag will grow over time. Debug only. |

---

## Known Limitations

- **Anthropic 529 (overloaded) errors:** roughly 10–30 % of slide-description
  requests may fail during peak load on the Anthropic API. The system
  catches the error, rolls back the dedup state for that slide, and the
  description is skipped (no crash). The next slide change is unaffected.
- **macOS only.** ScreenCaptureKit is an Apple-only framework, so the
  capture path won't run on Linux or Windows.
- **Listener latency:** typically 15–30 seconds behind live, by design.
  Buffer-anchored descriptions trade latency for synchronization.
- **Single source per instance.** One Aspire backend handles one Mac
  presentation source. Multiple listeners can connect to the same backend.

---

## Architecture Notes

**Why ScreenCaptureKit instead of BlackHole?** An earlier prototype used
BlackHole (a virtual audio driver) as the capture source, fed through
ffmpeg's `avfoundation`. That path had audible roughness from burst
delivery — frames arrived in clumps rather than steadily. Capturing
directly via Apple's ScreenCaptureKit API (through a small Swift CLI,
`tools/systemAudioDump`) produced clean, regular frames and eliminated the
roughness in phone listening tests.

**Why buffer-anchored descriptions?** A naïve implementation plays each
slide description as soon as TTS finishes synthesizing. That can land the
description *before* the listener has heard the speaker reach the slide,
or — worse — over audio from a previous video when the presenter switches
sources. Aspire tags each description with the absolute frame position in
the speaker buffer at the moment the slide was committed. The mixer only
promotes the description to active playback once the listener's playback
position has reached that anchor — guaranteeing the description lands
contextually correctly, even across video transitions.

**Why silence compression?** TTS descriptions add real seconds of audio to
the stream. Without anything reclaiming time, the listener falls
permanently behind live. Aspire detects silence runs longer than 2 seconds
and truncates them to 2 seconds, reclaiming time during natural pauses
without trimming speech.

---

## Research Context

This implementation accompanies:

> Hakim, A., & Hong, J. (2026). *Enhancing Slide Presentation Accessibility
> for Blind and Low-Vision Audiences Through Delay-Buffered Editing.*
> Proceedings of the ACM Conference on Designing Interactive Systems (DIS
> 2026). DOI: [10.1145/3800645.3812965](https://doi.org/10.1145/3800645.3812965)

The DIS 2026 paper formalizes the delay-buffered editing approach
prototyped as a Wizard-of-Oz study at ASSETS 2025. This codebase is the
fully-automated realization.

---

## Contact

**Azizul Haque**
Stevens Institute of Technology
Interactive Computing Lab (ICLAB)
Advisor: Dr. Jonggi Hong
Email: <ahaque3@stevens.edu>
