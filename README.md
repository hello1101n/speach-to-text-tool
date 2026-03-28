# Speech-to-Text

Hold right Command (⌘) to dictate, release to transcribe and paste. Fully local, runs on Apple Silicon via [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

- **No cloud, no API keys** — everything runs on-device
- **Auto-detects English and Russian**
- **Works in any app** — text is pasted at your cursor
- **Menu bar icon** shows recording/transcribing state
- **Logitech mouse support** — maps a button to right ⌘ for hands-free dictation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~1.5 GB disk for the Whisper model (downloaded on first run)
- ~2–3 GB RAM while running

## Install

```bash
git clone https://github.com/YOUR_USERNAME/speech-to-text.git
cd speech-to-text
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
cd speech-to-text
source .venv/bin/activate
stt
```

On first run, the `large-v3-turbo` model (~1.5 GB) downloads from HuggingFace and caches at `~/.cache/huggingface/`. Subsequent starts are fast.

Once running:
1. **Hold right ⌘** — recording starts (menu bar icon turns red)
2. **Release right ⌘** — audio is transcribed and pasted at your cursor
3. **Cmd+key combos** (e.g., Cmd+C) are detected and recording is discarded — no false triggers
4. **Taps shorter than 0.3s** are ignored

## Permissions

You need to grant two permissions the first time:

1. **Accessibility** — System Settings → Privacy & Security → Accessibility → add your terminal app (Terminal, iTerm2, etc.)
2. **Microphone** — system prompt appears on first run, click Allow

## Performance

| Metric | Value |
|---|---|
| Transcription (5s speech) | ~0.5–1.0s |
| End-to-end latency | < 2s from release to text |
| Model memory | ~2–3 GB |
| Languages | English, Russian (auto-detected) |

## How it works

- **Hotkey**: CGEventTap listens for right ⌘ (keycode 54) press/release
- **Audio**: `sounddevice` records at 16kHz mono float32 — exactly what Whisper expects, no temp files
- **Transcription**: `mlx-whisper` with `large-v3-turbo` model, runs on Apple GPU
- **Output**: clipboard via `pbcopy` + simulated Cmd+V via CGEvent

## License

MIT
