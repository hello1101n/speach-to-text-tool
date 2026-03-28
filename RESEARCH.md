# Local Whisper Speech-to-Text on macOS: Research

*Last updated: 2026-03-20*

---

## 1. Whisper Implementations Compared

### Benchmark Summary (MacBook Pro M4, transcribing same audio)

| Implementation | Time (sec) | GPU Accel | Streaming | Install |
|---|---|---|---|---|
| **FluidAudio CoreML** | 0.19 | ANE + CoreML | Yes | Swift build |
| **Parakeet MLX** | 0.50 | MLX (GPU) | No | pip |
| **MLX Whisper** | 1.02 | MLX (GPU) | No | pip |
| **Insanely Fast Whisper** | 1.13 | MPS (Metal) | No | pip |
| **whisper.cpp + CoreML** | 1.23 | ANE + CoreML + Metal | Yes | cmake build |
| **Lightning Whisper MLX** | 1.82 | MLX (GPU) | No | pip |
| **WhisperKit** | 2.22 | ANE + CoreML | Yes | Swift Package |
| **Whisper-MPS** | 5.37 | MPS (Metal) | No | pip |
| **faster-whisper** | 6.96 | CPU only (no Metal) | No | pip |

Source: [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)

---

### 1A. whisper.cpp

**What it is:** C/C++ port of OpenAI Whisper by Georgi Gerganov (ggml-org). The most mature and widely-used local option.

**Apple Silicon acceleration:**
- Metal GPU support built-in for encoder inference
- CoreML support for Apple Neural Engine (ANE) -- gives ~3x speedup over CPU-only
- Both can be enabled simultaneously

**Installation:**
```bash
# Basic build
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release

# Download a model
sh ./models/download-ggml-model.sh base.en

# Transcribe
./build/bin/whisper-cli -f samples/jfk.wav
```

**CoreML build (recommended on Apple Silicon):**
```bash
pip install ane_transformers openai-whisper coremltools
./models/generate-coreml-model.sh base.en
cmake -B build -DWHISPER_COREML=1
cmake --build build -j --config Release
```

**Streaming / live transcription:**
```bash
# Requires SDL2: brew install sdl2
cmake -B build -DWHISPER_SDL2=ON
cmake --build build -j --config Release
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```
The `stream` tool samples audio every 500ms and runs continuous transcription. Works well for live dictation.

**Speed:** On M2 MacBook Air, large-v3-turbo transcribes 3m20s audio in ~20 seconds. With CoreML, encoder runs on Neural Engine for best throughput.

**Verdict:** Best all-around option. Mature, well-maintained, has streaming built in, CoreML + Metal acceleration. C++ means low overhead. Python bindings available via `pywhispercpp`.

---

### 1B. MLX Whisper

**What it is:** Apple's MLX framework implementation of Whisper, designed specifically for Apple Silicon unified memory architecture.

**Apple Silicon acceleration:** Yes -- MLX runs natively on Apple GPU, no CoreML conversion needed. Leverages unified memory for zero-copy between CPU and GPU.

**Installation:**
```bash
pip install mlx-whisper
```

**Usage:**
```python
import mlx_whisper

result = mlx_whisper.transcribe("audio.mp3", path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
print(result["text"])
```

**Speed:** On M2 Ultra, Whisper v3 Turbo transcribes ~12 minutes of audio in 12.3 seconds (nearly 60x real-time). On M1 Max, ~30-40% faster than whisper.cpp.

**Streaming:** Not built-in. You'd need to implement chunked processing yourself.

**Verdict:** Excellent speed, trivial installation, pure Python. Best option if you want simple pip install + Python API. No streaming out of the box.

---

### 1C. Lightning Whisper MLX

**What it is:** Optimized MLX implementation with batched decoding, quantization, and speculative decoding.

**Installation:**
```bash
pip install lightning-whisper-mlx
```

**Usage:**
```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)
text = whisper.transcribe(audio_path="/path/to/audio.mp3")['text']
```

**Key features:**
- Batched decoding for higher throughput
- 4-bit and 8-bit quantization support
- Distilled model support (fewer decoder layers = faster)
- Default batch_size=12; increase for smaller models, decrease for larger ones

**Speed:** Claims 10x faster than whisper.cpp, 4x faster than standard MLX Whisper. However, mac-whisper-speedtest benchmarks show it at 1.82s vs whisper.cpp's 1.23s on M4, so the claims may be model/config dependent.

**Verdict:** Good for batch processing of longer audio. The batched decoding helps with throughput. No streaming.

---

### 1D. faster-whisper

**What it is:** CTranslate2-based Whisper implementation, popular on Linux/NVIDIA.

**Apple Silicon acceleration:** **No GPU support on macOS.** CTranslate2 does not support Metal/MPS. Falls back to CPU with Apple Accelerate framework. Supports int8 compute on Apple Silicon CPUs.

**Installation:**
```bash
pip install faster-whisper
```

**Speed on macOS:** Slowest option in benchmarks (6.96s on M4). Designed for NVIDIA CUDA GPUs.

**Verdict:** Not recommended for macOS. Use on Linux with NVIDIA GPUs instead.

---

### 1E. WhisperKit (Swift, native)

**What it is:** Native Swift package by Argmax, optimized for Apple Neural Engine. Used in Apple's own SpeechAnalyzer framework (WWDC 2025).

**Apple Silicon acceleration:** Full ANE + CoreML optimization. Achieves near-peak hardware utilization.

**Installation:** Swift Package Manager dependency or standalone CLI.

**Speed:** 0.45s mean per-word latency for streaming. On M4, benchmark shows 2.22s (slower than MLX Whisper in batch, but has real streaming).

**Verdict:** Best if building a native Swift/macOS app. Has real streaming support. Not ideal for Python-based workflows.

---

### 1F. Other Notable Options

- **Parakeet MLX** (0.50s on M4): NVIDIA's Parakeet TDT models on MLX. Extremely fast but less mature.
- **FluidAudio CoreML** (0.19s on M4): Fastest tested. Native Swift + CoreML with Parakeet TDT models. ~110x real-time factor on M4 Pro. Very new.
- **MetalRT**: Claims 714x real-time factor (0.0014 RTF). Commercial/early stage.

---

## 2. Model Selection: Speed vs Accuracy

### Recommended Models

| Model | Params | English WER | Speed | Use Case |
|---|---|---|---|---|
| **large-v3-turbo** | 809M | ~7.75% | 6x faster than large-v3 | **Best overall for local use** |
| **distil-large-v3** | ~756M | ~7.4% (English) | 1.5x faster than turbo | Best English-only speed |
| **small.en** | 244M | ~10% | Very fast | Low-memory devices |
| **base.en** | 74M | ~13% | Near-instant | Real-time streaming |
| **large-v3** | 1.55B | ~7.4% | Baseline | Maximum accuracy |

### Key Tradeoffs

**For dictation (short utterances, English):**
- Use `distil-large-v3` or `large-v3-turbo` -- both give excellent accuracy with fast speed
- `distil-large-v3` is English-only but ~1.5x faster than turbo
- `large-v3-turbo` supports 99+ languages

**For real-time streaming:**
- Use `base.en` or `small.en` -- fast enough for continuous transcription with <1s latency
- Accuracy is lower but acceptable for dictation

**For batch transcription (accuracy priority):**
- Use `large-v3` or `large-v3-turbo`

**Quantization:** 4-bit quantization reduces model size by ~45% and latency by ~19% with minimal accuracy loss. Good for memory-constrained setups.

---

## 3. macOS Global Hotkey: Capturing Right Command Key

### Option A: Karabiner-Elements (RECOMMENDED)

The most reliable way to capture a modifier-only key press on macOS. Karabiner works at the kernel level and can intercept any key before the OS processes it.

**Approach:** Use `to_if_alone` to trigger an action when right Command is tapped (pressed and released without combining with another key).

**Configuration (`~/.config/karabiner/karabiner.json`):**
```json
{
  "type": "basic",
  "from": {
    "key_code": "right_command",
    "modifiers": { "optional": ["any"] }
  },
  "to": [
    { "key_code": "right_command" }
  ],
  "to_if_alone": [
    {
      "key_code": "f18"
    }
  ],
  "parameters": {
    "basic.to_if_alone_timeout_milliseconds": 300
  }
}
```

This maps "tap right Command alone" to F18 (an unused key), while still letting right Command work as a normal modifier when held with other keys. Your app then listens for F18 as a global hotkey.

**Why this is best:**
- Works system-wide, no accessibility permissions needed for the remap itself
- Handles modifier-only detection natively (other tools struggle with this)
- Does not interfere with normal Command key usage
- The `to_if_alone` timeout prevents accidental triggers

---

### Option B: Hammerspoon + LeftRightHotkey Spoon

Hammerspoon can distinguish left/right modifiers using the LeftRightHotkey spoon.

```lua
LeftRightHotkey = hs.loadSpoon("LeftRightHotkey")
LeftRightHotkey:bind({"rCmd"}, "space", function()
    -- trigger recording
end)
LeftRightHotkey:start()
```

**Limitation:** Cannot detect modifier-only presses. Requires right Command + another key (e.g., rCmd+Space). Still a good option if you're okay with a two-key combo.

---

### Option C: Python with Quartz/CGEvent Tap

You can create a CGEvent tap to monitor flagsChanged events and detect right Command key presses by keycode.

```python
import Quartz
from PyObjC import objc

def callback(proxy, event_type, event, refcon):
    if event_type == Quartz.kCGEventFlagsChanged:
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        flags = Quartz.CGEventGetFlags(event)
        # Right Command keycode is 54
        if keycode == 54:
            if flags & Quartz.kCGEventFlagMaskCommand:
                print("Right Command pressed")
            else:
                print("Right Command released")
    return event

tap = Quartz.CGEventTapCreate(
    Quartz.kCGSessionEventTap,
    Quartz.kCGHeadInsertEventTap,
    Quartz.kCGEventTapOptionDefault,
    Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
    callback,
    None
)
```

**Note:** Requires Accessibility permission. Right Command keycode is 54 on macOS. This works but requires careful handling of press/release timing to avoid false triggers.

---

### Option D: Combined Approach (BEST)

Use **Karabiner-Elements** to remap right Command alone to F18, then use **Python** (or Hammerspoon) to listen for F18 as a standard global hotkey. This separates the hard problem (modifier-only detection) from the easy problem (listening for a key).

---

## 4. Audio Recording on macOS

### Recommended: `sounddevice` library

```bash
pip install sounddevice numpy
```

**Recording for Whisper:**
```python
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1         # Mono

# Record for a fixed duration
duration = 5  # seconds
audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()  # Wait until recording is finished
audio = audio.squeeze()  # Remove channel dimension -> shape (samples,)
```

**Callback-based recording (for push-to-talk):**
```python
import sounddevice as sd
import numpy as np
import queue

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# Start recording
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    blocksize=1024
)
stream.start()

# ... collect chunks while hotkey is held ...

stream.stop()

# Concatenate all chunks
chunks = []
while not audio_queue.empty():
    chunks.append(audio_queue.get())
audio = np.concatenate(chunks).squeeze()
```

### Format Requirements for Whisper

- **Sample rate:** 16,000 Hz (16 kHz) -- this is what Whisper expects internally
- **Channels:** 1 (mono)
- **Bit depth:** 16-bit PCM for WAV files, or float32 NumPy array for in-memory
- **Format:** WAV (PCM) is simplest; Whisper can also handle MP3/FLAC via ffmpeg

### Library Comparison

| Library | Pros | Cons |
|---|---|---|
| **sounddevice** | NumPy arrays, simple API, good macOS support | Needs PortAudio |
| **pyaudio** | Mature, widely used | Complex API, tricky macOS install |
| **AVFoundation (via PyObjC)** | Native macOS, no dependencies | Complex ObjC bridge |

**sounddevice** is the clear winner for Python projects. Install PortAudio via `brew install portaudio` if needed.

### macOS Permission

macOS requires microphone permission. When running from Terminal, you'll get a system prompt "Allow Terminal to use the microphone?" on first use.

---

## 5. Clipboard and Text Insertion

### Recommended Approach: Clipboard + Simulated Cmd+V

The most reliable cross-application method:

1. Copy text to clipboard
2. Simulate Cmd+V keystroke

```python
import subprocess
import time

def paste_text(text):
    """Set clipboard and simulate Cmd+V to paste at cursor."""
    # Step 1: Set clipboard content
    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    process.communicate(text.encode('utf-8'))

    # Step 2: Simulate Cmd+V using AppleScript
    subprocess.run([
        'osascript', '-e',
        'tell application "System Events" to keystroke "v" using command down'
    ])
```

**Alternative: Pure Python with Quartz (no subprocess):**
```python
import Quartz
import time

def paste_text(text):
    # Set clipboard
    from AppKit import NSPasteboard, NSPasteboardTypeString
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)

    # Simulate Cmd+V
    # V key = keycode 9
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    cmd_down = Quartz.CGEventCreateKeyboardEvent(source, 9, True)
    Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_down)

    cmd_up = Quartz.CGEventCreateKeyboardEvent(source, 9, False)
    Quartz.CGEventSetFlags(cmd_up, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_up)
```

**Note:** Both approaches require Accessibility permission for the app/terminal posting events.

### Other Options

| Method | Works Cross-App | Complexity | Notes |
|---|---|---|---|
| **pbcopy + osascript Cmd+V** | Yes | Low | Most reliable, recommended |
| **Quartz CGEvent** | Yes | Medium | No subprocess, faster |
| **AppleScript keystroke** | Yes | Low | Character-by-character, slow for long text |
| **Accessibility API (AXUIElement)** | Partial | High | App-specific, not universal |

---

## 6. Recommended Architecture

### Push-to-Talk Dictation Tool

```
Karabiner-Elements                    Python App
  |                                     |
  | right_cmd tap -> F18               |
  |-------------------------------------->|
  |                                     | detect F18 keydown
  |                                     | start sd.InputStream (16kHz)
  |                                     |   ... user speaks ...
  |                                     | detect F18 keyup
  |                                     | stop recording
  |                                     | concatenate audio chunks
  |                                     | run whisper transcription
  |                                     |   (mlx-whisper or whisper.cpp)
  |                                     | pbcopy + Cmd+V paste
  |                                     | text appears at cursor
```

### Recommended Stack

| Component | Choice | Why |
|---|---|---|
| **Whisper engine** | `mlx-whisper` with `large-v3-turbo` | Best speed on Apple Silicon, trivial pip install, good accuracy |
| **Alternative engine** | `whisper.cpp` with CoreML | If you need streaming or lower latency |
| **Hotkey** | Karabiner-Elements (rCmd -> F18) + Python listener | Most reliable modifier-only detection |
| **Audio capture** | `sounddevice` (16kHz, mono, float32) | Simple, reliable, NumPy native |
| **Text insertion** | `pbcopy` + `osascript` Cmd+V | Works everywhere, simple |
| **Model** | `large-v3-turbo` (multilingual) or `distil-large-v3` (English) | Best speed/accuracy balance |

### Expected Performance

- **Recording:** Real-time (no processing overhead)
- **Transcription:** 5 seconds of speech transcribed in ~0.3-1.0 seconds on M-series Macs
- **Total latency:** Under 2 seconds from releasing the key to text appearing

---

## Sources

- [mac-whisper-speedtest benchmark](https://github.com/anvanvan/mac-whisper-speedtest)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [MLX Whisper (Awni Hannun / Apple MLX)](https://x.com/awnihannun/status/1852438366962475038)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperKit](https://github.com/argmaxinc/WhisperKit)
- [Whisper model comparison](https://whisper-api.com/blog/models/)
- [Open source STT benchmarks 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [Whisper Apple Silicon benchmarks](https://www.voicci.com/blog/apple-silicon-whisper-performance.html)
- [All That Whispers (Marcos Huerta)](https://marcoshuerta.com/posts/all-that-whispers/)
- [Whisper large-v3-turbo vs distil-large-v3](https://huggingface.co/openai/whisper-large-v3-turbo/discussions/40)
- [Karabiner-Elements complex modifications](https://karabiner-elements.pqrs.org/docs/json/complex-modifications-manipulator-definition/)
- [Hammerspoon LeftRightHotkey](https://www.hammerspoon.org/Spoons/LeftRightHotkey.html)
- [pynput keyboard handling](https://pynput.readthedocs.io/en/latest/keyboard.html)
- [python-sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.3/)
- [MetalRT speech engine](https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon)
- [Paste as keystrokes macOS](https://gist.github.com/sscotth/310db98e7c4ec74e21819806dc527e97)
- [CGEventCreateKeyboardEvent](https://developer.apple.com/documentation/coregraphics/1456564-cgeventcreatekeyboardevent)
