from __future__ import annotations

from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from config import config
from core.persona import persona_manager

# Lazy import — only fails at speak-time so the module can be imported safely
# even if kokoro-onnx is not yet installed.
try:
    from kokoro_onnx import Kokoro
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    logger.warning("kokoro-onnx not installed — TTS will be unavailable.")

# ---------------------------------------------------------------------------
# Voice map: persona name → Kokoro voice id
# ---------------------------------------------------------------------------

VOICE_MAP: dict[str, str] = {
    "JARVIS": "af_heart",   # calm, formal male-adjacent voice
    "FRIDAY": "af_bella",   # warm, upbeat voice
}

# Sample rate expected by Kokoro
SAMPLE_RATE = 24_000

# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------

class TextToSpeech:
    """
    Persona-aware TTS engine backed by Kokoro-ONNX.

    The Kokoro pipeline is built lazily on first use and cached for the
    lifetime of the object to avoid repeated model loading overhead.
    """

    def __init__(
        self,
        model_path: str = config.KOKORO_MODEL_PATH,
        voices_path: str = config.KOKORO_VOICES_PATH,
        speed: float = config.TTS_SPEED,
    ) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self.speed = speed

        self._pipeline: Optional[object] = None  # cached Kokoro instance

    # ------------------------------------------------------------------
    # Pipeline (lazy, cached)
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> "Kokoro":
        if not _KOKORO_AVAILABLE:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run: pip install kokoro-onnx"
            )
        if self._pipeline is None:
            logger.info(f"Loading Kokoro TTS model from {self._model_path}")
            self._pipeline = Kokoro(self._model_path, self._voices_path)
            logger.success("Kokoro TTS pipeline loaded and cached.")
        return self._pipeline

    # ------------------------------------------------------------------
    # Voice resolution
    # ------------------------------------------------------------------

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """
        Priority: explicit argument → current persona → VOICE_MAP default.
        """
        if voice:
            return voice
        return VOICE_MAP.get(persona_manager.name, config.DEFAULT_VOICE)

    # ------------------------------------------------------------------
    # Core speak method
    # ------------------------------------------------------------------

    def speak(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> None:
        """
        Synthesise *text* and play it through the default audio output.

        Args:
            text:   The text to speak.
            voice:  Kokoro voice id. Defaults to the current persona's voice.
            speed:  Playback speed multiplier. Defaults to self.speed.
        """
        if not text or not text.strip():
            return

        resolved_voice = self._resolve_voice(voice)
        resolved_speed = speed if speed is not None else self.speed

        try:
            pipeline = self._get_pipeline()
        except RuntimeError as e:
            logger.error(f"TTS unavailable: {e}")
            return

        try:
            # Kokoro returns (samples: np.ndarray, sample_rate: int)
            samples, sample_rate = pipeline.create(
                text,
                voice=resolved_voice,
                speed=resolved_speed,
                lang="en-us",
            )

            samples = np.array(samples, dtype=np.float32)

            logger.debug(
                f"TTS speak → voice='{resolved_voice}' speed={resolved_speed} "
                f"len={len(text)} chars, {len(samples)/sample_rate:.1f}s"
            )

            sd.play(samples, samplerate=sample_rate)
            sd.wait()  # block until finished to avoid overlap

        except sd.PortAudioError as e:
            logger.error(f"Audio playback error: {e}")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")

    def speak_async(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> None:
        """
        Non-blocking speak — fires audio in a background thread.
        Does NOT guarantee ordering if called rapidly.
        """
        import threading
        threading.Thread(
            target=self.speak,
            args=(text, voice, speed),
            daemon=True,
            name="TTSSpeakThread",
        ).start()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Immediately stop any currently playing audio."""
        try:
            sd.stop()
            logger.debug("TTS audio stopped.")
        except Exception as e:
            logger.warning(f"TTS stop error: {e}")

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.5, min(speed, 2.0))
        logger.info(f"TTS speed set to {self.speed}")

    def set_voice_for_persona(self, persona_name: str, voice_id: str) -> None:
        """Override the default voice for a given persona at runtime."""
        VOICE_MAP[persona_name.upper()] = voice_id
        logger.info(f"Voice for '{persona_name}' set to '{voice_id}'")

    def preload(self) -> None:
        """Explicitly warm up the Kokoro pipeline. Call at startup to avoid first-speak delay."""
        try:
            self._get_pipeline()
        except Exception as e:
            logger.error(f"TTS preload failed: {e}")

    def __repr__(self) -> str:
        loaded = self._pipeline is not None
        return (
            f"<TextToSpeech voice='{self._resolve_voice(None)}' "
            f"speed={self.speed} pipeline_cached={loaded}>"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
tts = TextToSpeech()
