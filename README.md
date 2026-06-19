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
- **Buffer-anchored slide descriptions.** Each description is tagged with
  the buffer frame position where the slide was detected, and only plays
  once the listener's playback reaches that point. Fixes the cross-video
  desync where a new slide's description would land over the previous
  video's audio.
- **Smart silence compression.** Silences longer than 2 seconds have the
  excess removed; silences of 2 seconds or less are preserved for natural
  rhythm. Keeps the listener close to live without trimming speech.
- **WebRTC** for local low-latency listening on the same Wi-Fi network.
- **HLS over Cloudflare tunnel** for remote listeners on any network.
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
   │    perceptual hash sampled every 0.4 s                  │
   │    5-second stability gate before commit                │
   │    dedup against previously-described slides            │
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
   │    buffer-anchored descriptions                         │
   │    smart silence compression (>2 s)                     │
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
- **~500 MB disk** for code + ML models (Kokoro ONNX + voices)

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
cd ASPIRE2.0
source .venv/bin/activate
ASPIRE_AVFOUNDATION_INPUT=3:0 uvicorn app:app --host 0.0.0.0 --port 8000
```

Verify the startup log shows these three lines:

```
[capture] audio source: sck_cli
[capture] video source: avfoundation device 3:0
[mixer] silence compression: ON
```

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
| `ASPIRE_DISABLE_COMPRESSION` | unset | When `1`, disables silence compression. Listener lag will grow over time as TTS plays. Debug only. |

---

## Known Limitations

- **Anthropic 529 (overloaded) errors:** roughly 10–30 % of slide-description
  requests may fail when the Anthropic API is under load. The system
  catches the error, rolls back the dedup state for that slide, and skips
  the description (no crash). The next slide change is unaffected.
- **macOS only.** ScreenCaptureKit is an Apple-only framework, so the
  capture path cannot run on Linux or Windows.
- **Listener latency:** typically **15–30 seconds** behind live. Buffer-
  anchored descriptions trade latency for synchronization.
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

**Why buffer-anchored descriptions?** A naïve implementation plays each
slide description as soon as TTS finishes synthesizing. That can land the
description *before* the listener has heard the speaker reach the slide
moment — or, worse, over audio from a previous video when the presenter
switches sources. Aspire tags each description with the absolute frame
position in the speaker buffer at the moment the slide was committed. The
mixer only promotes the description to active playback once the listener's
playback position has reached that anchor — guaranteeing the description
lands at the right point in the listener's timeline, even across video
transitions.

**Why silence compression with a 2-second floor?** Every TTS description
adds real seconds of audio to the stream. Without something reclaiming
that time, the listener falls permanently behind live. Aspire detects
silence runs longer than 2 seconds and truncates the excess, reclaiming
time during natural pauses. Silences of 2 seconds or less are preserved to
keep speech rhythm natural.

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
