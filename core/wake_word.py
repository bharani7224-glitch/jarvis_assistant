"""Hands-free wake word ("Hey Jarvis") using openwakeword (offline, CPU, ONNX).

Design: no background thread. `wait_for_trigger()` blocks in the caller, owns the mic only while
waiting, and releases it before returning, so the STT module can open the mic without conflicts.
It also returns early when the manual hotkey event or the quit event is set.
"""
import logging
import queue
import threading

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger("jarvis.wake")

FRAME = 1280  # 80 ms at 16 kHz, the frame size openwakeword expects


class WakeWordDetector:
    def __init__(self) -> None:
        from openwakeword.model import Model

        name = config.WAKE_WORD_MODEL
        try:
            self._model = Model(wakeword_models=[name], inference_framework=config.WAKE_INFERENCE)
        except Exception as exc:
            # First run: the pretrained model files may not be downloaded yet.
            log.info("Downloading wake word models (%s)...", exc)
            from openwakeword import utils

            utils.download_models([name])
            self._model = Model(wakeword_models=[name], inference_framework=config.WAKE_INFERENCE)
        log.info("Wake word ready: '%s' (threshold %.2f)", name, config.WAKE_THRESHOLD)

    def _reset(self) -> None:
        """Clear internal buffers so audio from before this wait can't re-trigger."""
        try:
            self._model.reset()
        except Exception:
            pass

    def wait_for_trigger(self, manual: threading.Event, quit_event: threading.Event) -> str:
        """Block until activation. Returns 'wake_word', 'hotkey' or 'quit'."""
        if quit_event.is_set():
            return "quit"
        if manual.is_set():
            manual.clear()
            return "hotkey"

        self._reset()
        q: queue.Queue = queue.Queue(maxsize=100)

        def callback(indata, frames, t, status):
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME, callback=callback):
            while True:
                if quit_event.is_set():
                    return "quit"
                if manual.is_set():
                    manual.clear()
                    return "hotkey"
                try:
                    frame = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if len(frame) != FRAME:
                    continue
                scores = self._model.predict(frame.astype(np.int16))
                best = max(scores.values()) if scores else 0.0
                if best >= 0.2:
                    log.debug("wake score %.2f", best)
                if best >= config.WAKE_THRESHOLD:
                    log.info("Wake word detected (score %.2f)", best)
                    return "wake_word"
