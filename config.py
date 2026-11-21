"""Central configuration for Jarvis. Override anything via .env or environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- General
ASSISTANT_NAME = _env("ASSISTANT_NAME", "Jarvis")
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------- Triggers
HOTKEY = _env("HOTKEY", "ctrl+space")                     # manual override, always active
QUIT_HOTKEY = _env("QUIT_HOTKEY", "ctrl+alt+q")
BEEP_WAV = BASE_DIR / "beep.wav"                          # optional; falls back to a system beep
FOLLOWUP_SECONDS = _env_float("FOLLOWUP_SECONDS", 10.0)   # stay listening after a reply
END_PHRASES = {"exit", "quit", "stop", "goodbye", "bye", "that's all", "thats all", "thank you", "thanks"}

# ---------------------------------------------------------------- Wake word (openwakeword)
WAKE_WORD_ENABLED = _env_bool("WAKE_WORD_ENABLED", True)
WAKE_WORD_MODEL = _env("WAKE_WORD_MODEL", "hey_jarvis")   # built-in openwakeword model name
WAKE_THRESHOLD = _env_float("WAKE_THRESHOLD", 0.5)        # 0..1, lower = more sensitive
WAKE_INFERENCE = _env("WAKE_INFERENCE", "onnx")           # onnx works on Windows; tflite does not

# ---------------------------------------------------------------- Audio capture / endpointing
SAMPLE_RATE = 16000
SILENCE_SECONDS = _env_float("SILENCE_SECONDS", 1.2)      # silence that ends an utterance
NO_SPEECH_TIMEOUT = _env_float("NO_SPEECH_TIMEOUT", 6.0)  # give up if nobody speaks
MAX_RECORD_SECONDS = _env_float("MAX_RECORD_SECONDS", 20.0)
MIN_SPEECH_SECONDS = 0.35
PREROLL_BLOCKS = 10                                       # audio kept from before speech is detected

VAD_ENGINE = _env("VAD_ENGINE", "energy")                 # energy | silero (silero needs torch/silero-vad installed)
SILERO_THRESHOLD = _env_float("SILERO_THRESHOLD", 0.5)    # speech probability cutoff
SILERO_USE_ONNX = _env_bool("SILERO_USE_ONNX", True)
SILERO_BLOCK = 512                                        # samples per VAD frame at 16 kHz (fixed by model)

# Energy-gate fallback settings (VAD_ENGINE=energy)
BLOCK_MS = 30
MIN_RMS_THRESHOLD = _env_float("MIN_RMS_THRESHOLD", 250.0)   # int16 scale
MAX_RMS_THRESHOLD = _env_float("MAX_RMS_THRESHOLD", 1500.0)
NOISE_MULTIPLIER = _env_float("NOISE_MULTIPLIER", 3.0)

# ---------------------------------------------------------------- STT
STT_ENGINE = _env("STT_ENGINE", "google")                # google | whisper (whisper needs faster-whisper installed)
WHISPER_MODEL = _env("WHISPER_MODEL", "small")            # base | small | medium
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")            # cpu | cuda
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")         # "en" copes best with Tanglish; "" = auto-detect
WHISPER_PROMPT = (
    "Voice commands to a computer assistant, in English or Tanglish. "
    "Chrome open panni YouTube podu. Notepad open pannu. Battery evlo irukku? "
    "Google la search pannu. Volume konjam kammi pannu."
)
GOOGLE_STT_LANGUAGE = _env("GOOGLE_STT_LANGUAGE", "en-IN")

# ---------------------------------------------------------------- LLM
LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()       # groq | gemini | claude
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-flash")
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-haiku-5-5")
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 10)                # strict HTTP timeout in seconds (no long retries)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 400)
MAX_TOOL_ROUNDS = 8
MAX_HISTORY_MESSAGES = 24

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a voice assistant controlling the user's Windows PC.
Your replies are spoken aloud, so:
- Keep them to one or two short sentences. No markdown, lists, emojis or code.
- The user may speak English or Tanglish (Tamil written in English letters, mixed with English).
  Reply in the same style they used. Casual Tanglish is fine, e.g. "VS Code open pannitten."
- When the user wants something done on the PC, call the matching tool instead of describing it.
  You may call several tools in sequence. After the tools run, confirm the result briefly.
- If a tool returns an error, say so plainly and suggest one fix.
- If no tool fits, answer from your own knowledge, briefly.
- Never claim to have done something unless a tool reported success.

Keyboard and window tools act on whichever window is focused:
- After open_app, call wait_seconds(2) before typing or pressing keys so the app is ready.
- If the target window may not be focused, call switch_window first.
- To type into a browser address bar, press ctrl+l first. To open a new tab, press ctrl+t.
- Use type_text with press_enter=true for searches or chat messages that should be submitted.
- Never type passwords, card numbers or other secrets, even if asked."""

# ---------------------------------------------------------------- TTS
TTS_ENGINE = _env("TTS_ENGINE", "edge")                   # edge | offline
TTS_VOICE = _env("TTS_VOICE", "en-IN-PrabhatNeural")      # or en-US-ChristopherNeural
TTS_RATE = _env("TTS_RATE", "+10%")
TTS_OFFLINE_RATE = _env_int("TTS_OFFLINE_RATE", 185)

# ---------------------------------------------------------------- Safety
REQUIRE_CONFIRMATION_FOR_DANGEROUS = _env_bool("REQUIRE_CONFIRMATION", True)
CONFIRM_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "proceed", "aama", "aamam", "seri"}


def validate() -> list[str]:
    """Return a list of human-readable config problems (empty = OK)."""
    problems = []
    if LLM_PROVIDER not in ("groq", "gemini", "claude"):
        problems.append(f"LLM_PROVIDER must be 'groq', 'gemini', or 'claude' (got '{LLM_PROVIDER}').")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        problems.append("GROQ_API_KEY is missing in .env.")
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        problems.append("GEMINI_API_KEY is missing in .env.")
    if LLM_PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        problems.append("ANTHROPIC_API_KEY is missing in .env.")
    if STT_ENGINE not in ("whisper", "google"):
        problems.append("STT_ENGINE must be 'whisper' or 'google'.")
    if VAD_ENGINE not in ("silero", "energy"):
        problems.append("VAD_ENGINE must be 'silero' or 'energy'.")
    return problems
