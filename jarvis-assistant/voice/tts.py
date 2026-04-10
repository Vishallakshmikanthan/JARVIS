from __future__ import annotations

import re
from enum import Enum
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
# Tone context
# ---------------------------------------------------------------------------

class ToneContext(str, Enum):
    URGENT  = "urgent"   # fast, snappy
    CASUAL  = "casual"   # relaxed, warm
    SERIOUS = "serious"  # slower, deliberate
    NEUTRAL = "neutral"  # default baseline


# Keyword patterns used to auto-detect tone from spoken text
_TONE_PATTERNS: dict[ToneContext, re.Pattern] = {
    ToneContext.URGENT: re.compile(
        r"\b(urgent|emergency|critical|danger|alert|warning|error|immediately|now|asap|fast|hurry|quick)\b",
        re.IGNORECASE,
    ),
    ToneContext.CASUAL: re.compile(
        r"\b(sure|okay|cool|awesome|fun|joke|lol|nice|hey|chill|no problem|relax|bye|thanks|great)\b",
        re.IGNORECASE,
    ),
    ToneContext.SERIOUS: re.compile(
        r"\b(important|note|remember|please|consider|carefully|warning|system|security|data|privacy|confirm|verify)\b",
        re.IGNORECASE,
    ),
}


def detect_tone(text: str) -> ToneContext:
    """
    Classify the emotional tone of *text* using keyword heuristics.
    When multiple tones match, priority order is: URGENT > SERIOUS > CASUAL.
    Falls back to NEUTRAL.
    """
    return next(
        (tone for tone in (ToneContext.URGENT, ToneContext.SERIOUS, ToneContext.CASUAL)
         if _TONE_PATTERNS[tone].search(text)),
        ToneContext.NEUTRAL,
    )


# speed multipliers per tone (relative to the configured baseline)
_TONE_SPEED: dict[ToneContext, float] = {
    ToneContext.URGENT:  1.35,
    ToneContext.CASUAL:  0.90,
    ToneContext.SERIOUS: 0.78,
    ToneContext.NEUTRAL: 1.00,
}

# ---------------------------------------------------------------------------
# Voice map: persona name → Kokoro voice id
# ---------------------------------------------------------------------------

VOICE_MAP: dict[str, str] = {
    "JARVIS": "af_heart",   # calm, formal male-adjacent voice
    "FRIDAY": "af_bella",   # warm, upbeat voice
}

# Tone-specific voice overrides: (PERSONA_NAME, ToneContext) → Kokoro voice id
# Only entries that differ from the persona default need to be listed.
TONE_VOICE_MAP: dict[tuple[str, ToneContext], str] = {
    ("JARVIS", ToneContext.URGENT):  "af_sky",     # more energetic for urgent
    ("JARVIS", ToneContext.CASUAL):  "af_bella",   # warmer for casual chatter
    ("JARVIS", ToneContext.SERIOUS): "af_heart",   # keep the calm formal voice
    ("FRIDAY", ToneContext.URGENT):  "af_sky",     # snappy for Friday too
    ("FRIDAY", ToneContext.SERIOUS): "af_nicole",  # more grave for serious news
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
    # Tone resolution helpers
    # ------------------------------------------------------------------

    def _resolve_tone(self, text: str, tone: Optional[ToneContext]) -> ToneContext:
        """Return an explicit tone if provided, otherwise auto-detect from text."""
        return tone if tone is not None else detect_tone(text)

    def _resolve_speed_for_tone(self, tone: ToneContext, explicit_speed: Optional[float]) -> float:
        """Apply tone speed multiplier on top of the baseline speed."""
        if explicit_speed is not None:
            return explicit_speed
        return round(self.speed * _TONE_SPEED[tone], 2)

    def _resolve_voice_for_tone(self, tone: ToneContext, explicit_voice: Optional[str]) -> str:
        """Pick voice: explicit → tone override → persona default."""
        if explicit_voice:
            return explicit_voice
        persona = persona_manager.name.upper()
        return TONE_VOICE_MAP.get((persona, tone), self._resolve_voice(None))

    # ------------------------------------------------------------------
    # Core speak method
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        tone: Optional[ToneContext] = None,
    ) -> None:
        """
        Synthesise *text* and play it through the default audio output.

        Tone modulation is applied automatically unless *speed* / *voice* are
        given explicitly:
          - urgent  → faster speech (×1.35) + snappier voice
          - casual  → relaxed (×0.90) + warmer voice
          - serious → slower (×0.78) + more deliberate voice
          - neutral → baseline speed and persona default voice

        Args:
            text:  The text to speak.
            voice: Explicit Kokoro voice id (bypasses tone voice selection).
            speed: Explicit speed multiplier (bypasses tone speed calculation).
            tone:  Force a specific ToneContext; auto-detected when omitted.
        """
        if not text or not text.strip():
            return

        resolved_tone  = self._resolve_tone(text, tone)
        resolved_voice = self._resolve_voice_for_tone(resolved_tone, voice)
        resolved_speed = self._resolve_speed_for_tone(resolved_tone, speed)

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
                f"TTS speak → tone={resolved_tone.value} voice='{resolved_voice}' "
                f"speed={resolved_speed} len={len(text)} chars, "
                f"{len(samples)/sample_rate:.1f}s"
            )

            sd.play(samples, samplerate=sample_rate)
            sd.wait()  # block until finished to avoid overlap

        except sd.PortAudioError as e:
            logger.error(f"Audio playback error: {e}")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")

    def speak_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        tone: Optional[ToneContext] = None,
    ) -> None:
        """
        Non-blocking speak — fires audio in a background thread.
        Does NOT guarantee ordering if called rapidly.
        """
        import threading
        threading.Thread(
            target=self.speak,
            args=(text, voice, speed, tone),
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

    def set_tone_voice(self, persona_name: str, tone: ToneContext, voice_id: str) -> None:
        """Override the Kokoro voice used for a specific persona + tone combination."""
        TONE_VOICE_MAP[(persona_name.upper(), tone)] = voice_id
        logger.info(f"Tone voice for '{persona_name}'/'{tone.value}' set to '{voice_id}'")

    def detect_tone(self, text: str) -> ToneContext:
        """Expose tone detection as a public method for testing / external use."""
        return detect_tone(text)

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
