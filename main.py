"""Jarvis Phase 2: wake word or hotkey -> record (Silero VAD) -> transcribe -> LLM (+tools) -> speak."""
import logging
import sys
import threading

import keyboard

import config
import tools
from core.brain import Brain
from core.stt import SpeechToText
from core.tts import Speaker

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("jarvis")


def beep() -> None:
    """Short 'I'm listening' cue. Uses beep.wav if present, else a system beep."""
    try:
        import winsound
        if config.BEEP_WAV.exists():
            winsound.PlaySound(str(config.BEEP_WAV), winsound.SND_FILENAME)
        else:
            winsound.Beep(880, 110)
    except Exception:
        pass


class Jarvis:
    def __init__(self) -> None:
        self.stt = SpeechToText()
        self.speaker = Speaker()
        self.brain = Brain()
        self.wake = self._init_wake_word()
        self._manual = threading.Event()      # set by the hotkey
        self._quit = threading.Event()
        tools.set_confirm_handler(self._confirm_by_voice)

    @staticmethod
    def _init_wake_word():
        if not config.WAKE_WORD_ENABLED:
            return None
        try:
            from core.wake_word import WakeWordDetector
            return WakeWordDetector()
        except Exception as exc:
            log.warning("Wake word disabled (%s). Using the hotkey only.", exc)
            return None

    # ------------------------------------------------------------ events
    def _on_hotkey(self) -> None:
        if self.speaker.is_speaking:          # barge-in: cut Jarvis off and listen
            self.speaker.stop()
        self._manual.set()

    def _confirm_by_voice(self, question: str) -> bool:
        self.speaker.speak(question)
        beep()
        answer = self.stt.listen(no_speech_timeout=6).lower()
        words = {w.strip(".,!?") for w in answer.split()}
        return bool(words & config.CONFIRM_WORDS)

    def _wait_for_activation(self) -> str:
        """Block until the wake word, the hotkey or quit. Returns 'wake_word' | 'hotkey' | 'quit'."""
        if self.wake is not None:
            try:
                return self.wake.wait_for_trigger(self._manual, self._quit)
            except Exception:
                log.exception("Wake word listener failed; falling back to hotkey only.")
                self.wake = None
        while not self._quit.is_set():
            if self._manual.wait(timeout=0.3):
                self._manual.clear()
                return "hotkey"
        return "quit"

    # ------------------------------------------------------------ conversation
    def _converse(self) -> None:
        beep()
        text = self.stt.listen(config.NO_SPEECH_TIMEOUT)
        while text:
            if text.lower().strip(" .!?") in config.END_PHRASES:
                self.speaker.speak("Okay, I'm here if you need me.")
                self.brain.reset()
                return
            reply = self.brain.respond(text)
            self.speaker.speak(reply)
            if self._manual.is_set():         # hotkey pressed during the reply -> start a fresh turn
                return
            text = self.stt.listen(config.FOLLOWUP_SECONDS)   # follow-up window, no wake word needed

    # ------------------------------------------------------------ run
    def run(self) -> None:
        keyboard.add_hotkey(config.HOTKEY, self._on_hotkey)
        keyboard.add_hotkey(config.QUIT_HOTKEY, self._quit.set)
        trigger = (f"Say 'Hey Jarvis' or press {config.HOTKEY.upper()}" if self.wake
                   else f"Press {config.HOTKEY.upper()}")
        print(f"\n{config.ASSISTANT_NAME} ready. {trigger} to talk, {config.QUIT_HOTKEY.upper()} to quit.\n")
        self.speaker.speak(f"{config.ASSISTANT_NAME} online.")

        while not self._quit.is_set():
            source = self._wait_for_activation()
            if source == "quit":
                break
            log.info("Activated via %s", source)
            try:
                self._converse()
            except Exception:
                log.exception("Conversation turn failed")
        self.speaker.speak("Shutting down. Bye.")


def main() -> int:
    problems = config.validate()
    if problems:
        print("Config problems:\n - " + "\n - ".join(problems))
        print("Copy .env.example to .env and fill it in.")
        return 1
    jarvis = Jarvis()
    print("Loading speech models...")
    jarvis.stt.warmup()
    try:
        jarvis.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
