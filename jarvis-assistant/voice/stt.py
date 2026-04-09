from __future__ import annotations

from typing import Optional

import speech_recognition as sr
from loguru import logger

from core.persona import persona_manager

# ---------------------------------------------------------------------------
# Default tuning constants
# ---------------------------------------------------------------------------

ENERGY_THRESHOLD = 300        # minimum mic energy to detect as speech
PAUSE_THRESHOLD = 0.8         # seconds of silence before phrase ends
TIMEOUT = 5                   # seconds to wait for speech to begin
PHRASE_LIMIT = 10             # max seconds to record a single phrase
LANGUAGE = "en-US"


# ---------------------------------------------------------------------------
# STT engine
# ---------------------------------------------------------------------------

class SpeechToText:
    """
    Captures microphone audio and converts it to text using Google STT.
    
    All timeouts and thresholds are tuneable after instantiation.
    Errors are handled gracefully — the method always returns a str or None.
    """

    def __init__(
        self,
        energy_threshold: int = ENERGY_THRESHOLD,
        pause_threshold: float = PAUSE_THRESHOLD,
        timeout: int = TIMEOUT,
        phrase_limit: int = PHRASE_LIMIT,
        language: str = LANGUAGE,
        dynamic_energy: bool = True,
    ) -> None:
        self.timeout = timeout
        self.phrase_limit = phrase_limit
        self.language = language

        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self._recognizer.pause_threshold = pause_threshold
        self._recognizer.dynamic_energy_threshold = dynamic_energy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self, source: sr.AudioSource, duration: float = 0.5) -> None:
        """Adjust for ambient noise. Called once per listen() invocation."""
        self._recognizer.adjust_for_ambient_noise(source, duration=duration)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, prompt: bool = False) -> Optional[str]:
        """
        Open the microphone, wait for speech, and return the recognised text.

        Returns:
            str  — recognised utterance (stripped, lowercased)
            None — if no speech, timeout, or unrecoverable error occurred
        """
        with sr.Microphone() as source:
            if prompt:
                print(f"{persona_manager.name}: Listening...")

            self._calibrate(source)

            try:
                audio = self._recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_limit,
                )
            except sr.WaitTimeoutError:
                logger.debug("STT timeout — no speech detected within the window.")
                return None

        return self._transcribe(audio)

    def _transcribe(self, audio: sr.AudioData) -> Optional[str]:
        """Send audio to Google STT and return cleaned text."""
        try:
            text: str = self._recognizer.recognize_google(
                audio, language=self.language
            )
            text = text.strip()
            logger.info(f"STT recognised: '{text}'")
            return text

        except sr.UnknownValueError:
            logger.debug("STT could not understand the audio.")
            return None

        except sr.RequestError as e:
            logger.error(f"Google STT API error: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected STT error: {e}")
            return None

    def listen_once(self) -> Optional[str]:
        """
        Convenience wrapper — listens once and returns text or None.
        Identical to listen() but suppresses the prompt.
        """
        return self.listen(prompt=False)

    # ------------------------------------------------------------------
    # Tunables
    # ------------------------------------------------------------------

    def set_language(self, lang: str) -> None:
        self.language = lang
        logger.info(f"STT language set to '{lang}'")

    def set_timeout(self, seconds: int) -> None:
        self.timeout = max(1, seconds)
        logger.info(f"STT timeout set to {self.timeout}s")

    def set_energy_threshold(self, value: int) -> None:
        self._recognizer.energy_threshold = value
        logger.info(f"STT energy threshold set to {value}")

    def __repr__(self) -> str:
        return (
            f"<SpeechToText lang='{self.language}' "
            f"timeout={self.timeout}s phrase_limit={self.phrase_limit}s>"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
stt = SpeechToText()
