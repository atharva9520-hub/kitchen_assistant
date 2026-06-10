"""
voice_input.py
--------------
Listens for spoken commands from the user during a cooking session.
Runs on macOS using the built-in microphone — no API key, no internet.

Two recognition backends (auto-selected):
  1. Whisper (local, preferred) — most accurate, runs on M4 Neural Engine
  2. Google Speech (fallback)   — requires internet, free tier

Recognised commands:
  "next"   / "done"   / "ready"   → advance to next step
  "skip"               → skip current step
  "repeat" / "again"  → repeat current instruction
  "stop"   / "pause"  → pause session
  "timer"  / "time"   → how much time is left on current timer
  "help"               → list commands

Usage:
    from voice_input import VoiceListener

    listener = VoiceListener(on_command=my_callback)
    listener.start()   # starts background listen loop
    listener.stop()    # stops it
"""

import threading
import queue
import time
import re

# lazy imports — checked at runtime so missing packages give clear errors
_sr  = None
_whisper = None


def _load_sr():
    global _sr
    if _sr is None:
        try:
            import speech_recognition as sr
            _sr = sr
        except ImportError:
            raise ImportError(
                "SpeechRecognition not installed.\n"
                "Run: pip install SpeechRecognition pyaudio\n"
                "     brew install portaudio   (macOS)"
            )
    return _sr


def _load_whisper():
    global _whisper
    if _whisper is None:
        try:
            import whisper
            _whisper = whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed.\n"
                "Run: pip install openai-whisper"
            )
    return _whisper


# ── command patterns ──────────────────────────────────────────────────────────

COMMANDS = {
    "next":   [r"\bnext\b", r"\bdone\b", r"\bready\b", r"\bcontinue\b", r"\bfinished\b", r"\bmove on\b"],
    "skip":   [r"\bskip\b", r"\bskip (this|step)\b"],
    "repeat": [r"\brepeat\b", r"\bagain\b", r"\bsay that again\b", r"\bwhat( did you say)?\b"],
    "stop":   [r"\bstop\b", r"\bpause\b", r"\bwait\b", r"\bhold on\b"],
    "timer":  [r"\btimer\b", r"\bhow (long|much time)\b", r"\btime (left|remaining)\b"],
    "help":   [r"\bhelp\b", r"\bcommands\b", r"\bwhat can (i|you)\b"],
}

def _parse_command(text: str) -> str | None:
    """
    Match transcribed text against known command patterns.
    Returns command name or None if no match.
    """
    text = text.lower().strip()
    for command, patterns in COMMANDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return command
    return None


# ── whisper local recognition ─────────────────────────────────────────────────

_whisper_model = None
_whisper_lock  = threading.Lock()

def _get_whisper_model(size: str = "tiny.en"):
    """Load Whisper model once and cache it."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            whisper = _load_whisper()
            print(f"  [voice_input] Loading Whisper {size} model…")
            _whisper_model = whisper.load_model(size)
            print("  [voice_input] Whisper ready ✓")
    return _whisper_model


def _transcribe_with_whisper(audio_data, sr_module) -> str:
    """
    Transcribe a SpeechRecognition AudioData object using local Whisper.
    Converts to WAV bytes → temp file → Whisper.
    """
    import tempfile, os
    model = _get_whisper_model("tiny.en")  # tiny is fast enough for commands

    # write audio to temp wav
    wav_bytes = audio_data.get_wav_data()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        result = model.transcribe(tmp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


# ── voice listener ────────────────────────────────────────────────────────────

class VoiceListener:
    """
    Background thread that continuously listens for voice commands.
    Calls on_command(command: str) when a command is recognised.

    Args:
        on_command:   callback receiving the command string
        backend:      "whisper" (local) or "google" (online fallback)
        energy_threshold: mic sensitivity (higher = less sensitive)
        pause_duration:   seconds of silence to mark end of phrase
    """

    def __init__(
        self,
        on_command,
        backend: str = "whisper",
        energy_threshold: int = 300,
        pause_duration: float = 0.8,
    ):
        self.on_command       = on_command
        self.backend          = backend
        self.energy_threshold = energy_threshold
        self.pause_duration   = pause_duration

        self._stop_event = threading.Event()
        self._thread     = None
        self._muted      = False  # mute while assistant is speaking

    def start(self):
        """Start the background listen loop."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("  [voice_input] Listening for commands…")
        print(f"  [voice_input] Backend: {self.backend}")

    def stop(self):
        """Stop listening."""
        self._stop_event.set()
        print("  [voice_input] Stopped")

    def mute(self):
        """Temporarily stop processing (e.g. while assistant is speaking)."""
        self._muted = True

    def unmute(self):
        """Resume processing."""
        self._muted = False

    def _listen_loop(self):
        sr = _load_sr()
        recognizer = sr.Recognizer()
        recognizer.energy_threshold  = self.energy_threshold
        recognizer.pause_threshold   = self.pause_duration
        recognizer.dynamic_energy_threshold = True  # adapts to kitchen noise

        # pre-load whisper model in background so first command is fast
        if self.backend == "whisper":
            threading.Thread(
                target=_get_whisper_model, args=("tiny.en",), daemon=True
            ).start()

        with sr.Microphone() as source:
            print("  [voice_input] Adjusting for ambient noise…")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("  [voice_input] Ready — say 'next', 'skip', 'repeat', etc.")

            while not self._stop_event.is_set():
                try:
                    # non-blocking listen with timeout
                    audio = recognizer.listen(
                        source,
                        timeout=1.0,          # wait max 1s for speech to start
                        phrase_time_limit=5,  # max 5s per command
                    )
                except sr.WaitTimeoutError:
                    continue  # no speech — loop again
                except Exception as e:
                    print(f"  [voice_input] Listen error: {e}")
                    time.sleep(0.5)
                    continue

                if self._muted:
                    continue

                # transcribe in a thread so listen loop doesn't block
                threading.Thread(
                    target=self._process_audio,
                    args=(audio, recognizer),
                    daemon=True,
                ).start()

    def _process_audio(self, audio, recognizer):
        """Transcribe audio and fire command callback if recognised."""
        sr = _load_sr()

        try:
            if self.backend == "whisper":
                text = _transcribe_with_whisper(audio, sr)
            else:
                # Google fallback
                text = recognizer.recognize_google(audio)

            if not text:
                return

            print(f"  [voice_input] Heard: '{text}'")
            command = _parse_command(text)

            if command:
                print(f"  [voice_input] Command: {command}")
                try:
                    self.on_command(command)
                except Exception as e:
                    print(f"  [voice_input] Command callback error: {e}")
            else:
                print(f"  [voice_input] No command matched")

        except sr.UnknownValueError:
            pass  # couldn't understand — silence is fine in a kitchen
        except sr.RequestError as e:
            print(f"  [voice_input] Recognition service error: {e}")
        except Exception as e:
            print(f"  [voice_input] Transcription error: {e}")


# ── install check ─────────────────────────────────────────────────────────────

def check_dependencies() -> dict:
    """Check which backends are available. Returns status dict."""
    status = {}

    try:
        import speech_recognition as sr
        import pyaudio
        status["microphone"] = True
    except ImportError as e:
        status["microphone"] = False
        status["microphone_error"] = str(e)

    try:
        import whisper
        status["whisper"] = True
    except ImportError:
        status["whisper"] = False

    status["google_fallback"] = status.get("microphone", False)

    return status


if __name__ == "__main__":
    # quick test
    print("Testing voice input…")
    deps = check_dependencies()
    print(f"Dependencies: {deps}")

    if not deps.get("microphone"):
        print("\nMicrophone not available. Install:")
        print("  brew install portaudio")
        print("  pip install pyaudio SpeechRecognition")
    else:
        def handle(cmd):
            print(f"✓ Got command: {cmd}")

        listener = VoiceListener(on_command=handle)
        listener.start()
        print("Say a command (next / skip / repeat / stop / timer / help)…")
        time.sleep(15)
        listener.stop()
