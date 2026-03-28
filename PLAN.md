# Speech-to-Text: Implementation Plan

*Hold right ⌘ to dictate, release to paste. Fully local, EN + RU.*

---

## Stack

| Component | Choice | Why |
|---|---|---|
| **Runtime** | Python 3.13 (installed) | Already available, pip ecosystem |
| **Whisper engine** | `mlx-whisper` | Native Apple Silicon GPU via MLX, ~60x real-time on M-series, trivial pip install |
| **Model** | `large-v3-turbo` (809M params) | Best speed/accuracy tradeoff (~7.75% WER), auto-detects EN/RU |
| **Hotkey** | `pyobjc-framework-Quartz` CGEventTap | Detects right ⌘ press/release natively — no Karabiner needed |
| **Audio** | `sounddevice` + `numpy` | 16kHz mono float32, direct numpy arrays to Whisper (no temp files) |
| **Output** | `pbcopy` + AppleScript `Cmd+V` | Works in any app, simple |

## Architecture

```
Right ⌘ hold → CGEventTap detects keycode 54 down
             → start mic recording (sounddevice InputStream, 16kHz mono)
             → delayed "tink" sound (200ms) — cancelled if Cmd+key combo detected

Right ⌘ release → CGEventTap detects keycode 54 up
               → stop recording
               → mlx-whisper transcribes audio (auto-detects EN/RU)
               → pbcopy → AppleScript Cmd+V → text appears at cursor
               → "pop" sound confirms
```

## Smart Combo Detection

If right ⌘ is held and another key is pressed (e.g., Cmd+C), the recording is silently discarded. Prevents false triggers.

- CGEventTap listens for both `kCGEventFlagsChanged` (modifier keys) and `kCGEventKeyDown` (regular keys)
- If any `kCGEventKeyDown` fires while recording → mark as combo → discard on release
- Recordings shorter than 0.3s are also discarded (accidental taps)

## Project Structure

```
speech-to-text/
├── pyproject.toml          # project config + dependencies
├── PLAN.md                 # this file
├── RESEARCH.md             # research notes
└── stt/
    ├── __init__.py
    └── app.py              # ~150 lines: hotkey, recorder, transcriber, paster
```

## Dependencies

```toml
[project]
dependencies = [
    "mlx-whisper",          # Whisper on Apple MLX
    "sounddevice",          # mic recording
    "numpy",                # audio buffer
    "pyobjc-framework-Quartz",  # CGEventTap for hotkey
]
```

## Key Implementation Details

### Hotkey Detection (CGEventTap)

- Right Command keycode = **54** on macOS
- Listen for `kCGEventFlagsChanged` events
- Check `kCGEventFlagMaskCommand` flag + keycode 54 to distinguish right from left ⌘
- Transition tracking: `right_cmd_was_down` state to detect press/release edges

### Audio Recording

- `sounddevice.InputStream` with callback appending chunks to a list
- 16kHz sample rate, mono, float32 — exactly what Whisper expects
- No file I/O: audio stays as numpy array from mic to Whisper

### Transcription

- `mlx_whisper.transcribe(audio_array, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")`
- `language=None` → auto-detection handles EN/RU seamlessly
- `condition_on_previous_text=False` → each recording is independent, prevents hallucination carry-over
- Runs in a background thread to not block the event loop

### Text Output

- `pbcopy` sets clipboard via subprocess
- `osascript` simulates `Cmd+V` to paste at cursor in any app
- Threading lock prevents race conditions on rapid consecutive dictations

### Sound Feedback

- **Start**: `Tink.aiff` — delayed 200ms (cancelled if combo key detected)
- **Done**: `Pop.aiff` — plays after text is pasted

## Threading Model

```
Main thread       → CFRunLoop + CGEventTap (must be main thread)
Audio thread      → sounddevice InputStream callback (managed by PortAudio)
Transcribe thread → spawned on each release, runs mlx_whisper.transcribe()
```

## One-Time Setup

### Permissions Required

1. **Accessibility**: System Settings → Privacy & Security → Accessibility → add Terminal/iTerm2
   - Needed for: CGEventTap (hotkey detection) + AppleScript keystroke simulation
2. **Microphone**: system prompt on first run → allow Terminal/iTerm2

### Model Download

- `large-v3-turbo` is ~1.5GB, downloaded from HuggingFace on first run
- Cached at `~/.cache/huggingface/`
- App does a warm-up transcription on startup to pre-load into memory

## Usage

```bash
cd speech-to-text
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
stt   # "Loading model... Model loaded. Hold right ⌘ to dictate."
```

## Expected Performance (M1 Pro)

| Metric | Value |
|---|---|
| Recording latency | Real-time (no overhead) |
| Transcription (5s speech) | ~0.5–1.0s |
| Total end-to-end | < 2s from release to text at cursor |
| Model memory | ~2–3GB |
| Languages | English, Russian (auto-detected) |

## Potential Improvements (Later)

- **Faster model**: swap to `distil-large-v3` for English-only mode (~1.5x faster)
- **Streaming**: implement chunked transcription for real-time partial results
- **Launch Agent**: auto-start on login via `launchd` plist
- **Configurable hotkey**: allow changing the trigger key
