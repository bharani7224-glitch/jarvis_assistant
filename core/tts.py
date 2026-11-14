"""Text-to-speech: edge-tts (online neural voices) + pygame playback, pyttsx3 offline fallback."""
import asyncio
import logging
import os
import re
import tempfile
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import config  # noqa: E402

log = logging.getLogger("jarvis.tts")


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`#>~]", "", text)
    text = re.sub(r"https?://\S+", "link", text)
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._offline_engine = None
        self._mixer_ready = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self) -> None:
        """Interrupt whatever is currently being spoken."""
        self._stop.set()
        try:
            if self._mixer_ready:
                import pygame
                pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if self._offline_engine is not None:
                self._offline_engine.stop()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """Blocking: returns when speech finishes or is interrupted."""
        text = _clean_for_speech(text)
        if not text:
            return
        log.info("TTS: %s", text)
        self._stop.clear()
        self._speaking.set()
        try:
            if config.TTS_ENGINE == "edge":
                try:
                    self._speak_edge(text)
                    return
                except Exception as exc:
                    log.warning("edge-tts failed (%s); using offline voice.", exc)
            if not self._stop.is_set():
                self._speak_offline(text)
        finally:
            self._speaking.clear()

    # ------------------------------------------------------------ engines
    def _speak_edge(self, text: str) -> None:
        import edge_tts
        import pygame

        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            async def synth():
                await edge_tts.Communicate(text, config.TTS_VOICE, rate=config.TTS_RATE).save(path)

            asyncio.run(synth())
            if self._stop.is_set():
                return
            if not self._mixer_ready:
                pygame.mixer.init()
                self._mixer_ready = True
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self._stop.is_set():
                time.sleep(0.05)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _speak_offline(self, text: str) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", config.TTS_OFFLINE_RATE)
        self._offline_engine = engine
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            self._offline_engine = None
