"""Speech-to-Text: Hold right ⌘ to record, release to transcribe and paste."""

import subprocess
import sys
import threading
import time

import numpy as np
import rumps
import sounddevice as sd
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventGetType,
    CGEventMaskBit,
    CGEventPost,
    CGEventSetFlags,
    CGEventSourceCreate,
    CGEventTapCreate,
    CGEventTapEnable,
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    kCFRunLoopCommonModes,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventSourceStateHIDSystemState,
    kCGEventTapDisabledByTimeout,
    kCGHIDEventTap,
    kCGHeadInsertEventTap,
    kCGSessionEventTap,
    kCGKeyboardEventKeycode,
    kCGEventFlagMaskCommand,
)

RIGHT_CMD_KEYCODE = 54
VIRTUAL_CMD_KEYCODE = 65535  # Logitech virtual right Cmd
MIN_RECORDING_SECONDS = 0.3
SAMPLE_RATE = 16000
MODEL = "mlx-community/whisper-large-v3-turbo"


STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_TRANSCRIBING = "transcribing"

ICON_IDLE = "🎙"
ICON_RECORDING = "🔴"
ICON_TRANSCRIBING = "⏳"


class SpeechToText:
    def __init__(self, on_state_change=None):
        self.recording = False
        self.audio_buffer: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.combo_key_pressed = False
        self.trigger_down = False
        self.paste_lock = threading.Lock()
        self._on_state_change = on_state_change

    def _set_state(self, state):
        if self._on_state_change:
            self._on_state_change(state)

    def warm_up(self):
        from mlx_whisper.load_models import load_model

        print(f"Loading model {MODEL}...")
        load_model(path_or_hf_repo=MODEL)
        print("Model loaded.")

    def setup_event_tap(self):
        engine = self

        def callback(proxy, event_type, event, refcon):
            if event_type == kCGEventTapDisabledByTimeout:
                CGEventTapEnable(tap, True)
                return event

            actual_type = CGEventGetType(event)
            keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            flags = CGEventGetFlags(event)
            cmd_active = bool(flags & kCGEventFlagMaskCommand)

            if actual_type == kCGEventFlagsChanged:
                # Physical right Cmd press/release
                if keycode == RIGHT_CMD_KEYCODE:
                    if cmd_active and not engine.trigger_down:
                        engine.trigger_down = True
                        engine.start_recording()
                    elif not cmd_active and engine.trigger_down:
                        engine.trigger_down = False
                        engine.stop_recording()
                # Logitech virtual Cmd release (keycode=0, all flags cleared)
                elif keycode == 0 and engine.trigger_down and not cmd_active:
                    engine.trigger_down = False
                    engine.stop_recording()

            elif actual_type == kCGEventKeyDown:
                # Logitech virtual Cmd press (keycode=65535 + Cmd flag)
                if keycode == VIRTUAL_CMD_KEYCODE and cmd_active:
                    if not engine.trigger_down:
                        engine.trigger_down = True
                        engine.start_recording()
                # Any other key while recording → combo, discard
                elif engine.recording:
                    engine.combo_key_pressed = True

            return event

        event_mask = (
            CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventKeyDown)
        )

        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            0,
            event_mask,
            callback,
            None,
        )

        if tap is None:
            print(
                "ERROR: Cannot create event tap.\n"
                "Grant Accessibility permission:\n"
                "  System Settings → Privacy & Security → Accessibility\n"
                "  Add and enable your terminal app.",
                file=sys.stderr,
            )
            sys.exit(1)

        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
        CGEventTapEnable(tap, True)
        print("Listening for right ⌘...")

    def start_recording(self):
        self.recording = True
        self.audio_buffer = []
        self.combo_key_pressed = False
        self._set_state(STATE_RECORDING)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()


    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_buffer.append(indata.copy())

    def stop_recording(self):
        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.combo_key_pressed:
            self._set_state(STATE_IDLE)
            return

        if not self.audio_buffer:
            self._set_state(STATE_IDLE)
            return

        audio = np.concatenate(self.audio_buffer, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < MIN_RECORDING_SECONDS:
            self._set_state(STATE_IDLE)
            return

        self._set_state(STATE_TRANSCRIBING)
        threading.Thread(
            target=self._transcribe_and_paste, args=(audio,), daemon=True
        ).start()

    def _transcribe_and_paste(self, audio: np.ndarray):
        import mlx_whisper

        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=MODEL,
                condition_on_previous_text=False,
            )
            lang = result.get("language", "en")
            # If Whisper picks a wrong language, retry forcing English
            if lang not in ("en", "ru"):
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=MODEL,
                    condition_on_previous_text=False,
                    language="en",
                )
            text = result["text"].strip().rstrip(".")
            if text:
                self._paste(text)
        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)
        finally:
            self._set_state(STATE_IDLE)

    def _paste(self, text: str):
        with self.paste_lock:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            time.sleep(0.05)
            # Simulate Cmd+V via CGEvent (no AppleScript, no focus issues)
            source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            # V key = keycode 9
            v_down = CGEventCreateKeyboardEvent(source, 9, True)
            CGEventSetFlags(v_down, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, v_down)
            v_up = CGEventCreateKeyboardEvent(source, 9, False)
            CGEventSetFlags(v_up, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, v_up)


class STTMenuBar(rumps.App):
    def __init__(self):
        super().__init__(ICON_IDLE, quit_button="Quit")
        self.menu = [rumps.MenuItem("Hold right ⌘ to dictate", callback=None)]
        self.engine = SpeechToText(on_state_change=self._update_state)

    def _update_state(self, state):
        icons = {
            STATE_IDLE: ICON_IDLE,
            STATE_RECORDING: ICON_RECORDING,
            STATE_TRANSCRIBING: ICON_TRANSCRIBING,
        }
        self.title = icons.get(state, ICON_IDLE)

    def _setup(self):
        self.engine.warm_up()
        self.engine.setup_event_tap()


def main():
    app = STTMenuBar()
    app._setup()
    app.run()


if __name__ == "__main__":
    main()
