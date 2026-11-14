"""Speech-to-text: mic capture with Silero VAD endpointing, then Whisper or Google STT.

VAD_ENGINE=silero (default) uses silero-vad's VADIterator for accurate start/end-of-speech.
VAD_ENGINE=energy (or automatic fallback if silero-vad isn't installed) uses the Phase 1 RMS gate.
"""
import logging
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger("jarvis.stt")


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


class SpeechToText:
    def __init__(self) -> None:
        self._model = None              # faster-whisper
        self._torch = None              # silero-vad pieces
        self._vad_iter = None
        self._vad_engine = config.VAD_ENGINE

    # ------------------------------------------------------------ model loading
    def warmup(self) -> None:
        """Load models up front so the first command isn't slow."""
        if self._vad_engine == "silero":
            self._load_silero()
        if config.STT_ENGINE == "whisper":
            self._load_whisper()

    def _load_whisper(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Loading Whisper '%s' (%s/%s)...", config.WHISPER_MODEL,
                     config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
            self._model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                       compute_type=config.WHISPER_COMPUTE_TYPE)
        return self._model

    def _load_silero(self) -> None:
        if self._vad_iter is not None or self._vad_engine != "silero":
            return
        try:
            import torch
            from silero_vad import VADIterator, load_silero_vad

            torch.set_num_threads(1)
            model = load_silero_vad(onnx=config.SILERO_USE_ONNX)
            self._vad_iter = VADIterator(
                model,
                threshold=config.SILERO_THRESHOLD,
                sampling_rate=config.SAMPLE_RATE,
                min_silence_duration_ms=int(config.SILENCE_SECONDS * 1000),
                speech_pad_ms=30,
            )
            self._torch = torch
            log.info("Silero VAD ready (silence cutoff %.1fs).", config.SILENCE_SECONDS)
        except Exception as exc:
            log.warning("Silero VAD unavailable (%s). Falling back to the energy gate.", exc)
            self._vad_engine = "energy"

    # ------------------------------------------------------------ recording
    def record(self, no_speech_timeout: float) -> np.ndarray | None:
        """Record one utterance. Returns int16 mono audio, or None if nobody spoke."""
        if self._vad_engine == "silero":
            self._load_silero()
        if self._vad_engine == "silero":
            audio = self._record_silero(no_speech_timeout)
        else:
            audio = self._record_energy(no_speech_timeout)
        if audio is None or len(audio) / config.SAMPLE_RATE < config.MIN_SPEECH_SECONDS:
            return None
        return audio

    def _record_silero(self, no_speech_timeout: float) -> np.ndarray | None:
        torch = self._torch
        vad = self._vad_iter
        vad.reset_states()
        block = config.SILERO_BLOCK
        q: queue.Queue = queue.Queue()

        def callback(indata, frames, t, status):
            if status:
                log.debug("audio status: %s", status)
            q.put(indata[:, 0].copy())

        frames: list[np.ndarray] = []
        preroll: deque = deque(maxlen=config.PREROLL_BLOCKS)
        started = False

        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=block, callback=callback):
            t_start = time.monotonic()
            while True:
                try:
                    chunk = q.get(timeout=1)
                except queue.Empty:
                    log.warning("No audio from microphone.")
                    return None
                if len(chunk) != block:
                    continue
                now = time.monotonic()
                event = vad(torch.from_numpy(chunk.astype(np.float32) / 32768.0), return_seconds=False)

                if not started:
                    preroll.append(chunk)
                    if event and "start" in event:
                        started = True
                        frames.extend(preroll)
                    elif now - t_start > no_speech_timeout:
                        return None
                else:
                    frames.append(chunk)
                    if event and "end" in event:
                        break
                    if now - t_start > config.MAX_RECORD_SECONDS:
                        break
        return np.concatenate(frames) if frames else None

    def _record_energy(self, no_speech_timeout: float) -> np.ndarray | None:
        sr = config.SAMPLE_RATE
        block = int(sr * config.BLOCK_MS / 1000)
        q: queue.Queue = queue.Queue()

        def callback(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        frames: list[np.ndarray] = []
        preroll: deque = deque(maxlen=config.PREROLL_BLOCKS)
        started = False
        last_voice = 0.0

        with sd.InputStream(samplerate=sr, channels=1, dtype="int16", blocksize=block, callback=callback):
            noise = [_rms(q.get(timeout=2)) for _ in range(10)]
            threshold = float(np.clip(np.median(noise) * config.NOISE_MULTIPLIER,
                                      config.MIN_RMS_THRESHOLD, config.MAX_RMS_THRESHOLD))
            t_start = time.monotonic()
            while True:
                try:
                    chunk = q.get(timeout=1)
                except queue.Empty:
                    return None
                now = time.monotonic()
                level = _rms(chunk)
                if not started:
                    preroll.append(chunk)
                    if level > threshold:
                        started = True
                        frames.extend(preroll)
                        last_voice = now
                    elif now - t_start > no_speech_timeout:
                        return None
                else:
                    frames.append(chunk)
                    if level > threshold:
                        last_voice = now
                    elif now - last_voice >= config.SILENCE_SECONDS:
                        break
                    if now - t_start > config.MAX_RECORD_SECONDS:
                        break
        return np.concatenate(frames) if frames else None

    # ------------------------------------------------------------ transcription
    def transcribe(self, audio: np.ndarray) -> str:
        if config.STT_ENGINE == "google":
            return self._transcribe_google(audio)
        return self._transcribe_whisper(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        model = self._load_whisper()
        pcm = audio.astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            pcm,
            language=config.WHISPER_LANGUAGE or None,
            beam_size=1,
            vad_filter=True,
            initial_prompt=config.WHISPER_PROMPT,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _transcribe_google(self, audio: np.ndarray) -> str:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        data = sr.AudioData(audio.tobytes(), config.SAMPLE_RATE, 2)
        try:
            return recognizer.recognize_google(data, language=config.GOOGLE_STT_LANGUAGE).strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            log.error("Google STT request failed: %s", exc)
            return ""

    # ------------------------------------------------------------ public
    def listen(self, no_speech_timeout: float | None = None) -> str:
        """Record one utterance and return its transcript ('' if nothing was said)."""
        timeout = config.NO_SPEECH_TIMEOUT if no_speech_timeout is None else no_speech_timeout
        audio = self.record(timeout)
        if audio is None:
            return ""
        t0 = time.monotonic()
        text = self.transcribe(audio)
        log.info("STT (%.2fs): %s", time.monotonic() - t0, text)
        return text
