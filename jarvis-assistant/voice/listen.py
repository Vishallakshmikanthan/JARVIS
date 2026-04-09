from __future__ import annotations

import struct
import threading
from typing import Callable, Optional

import pvporcupine
import pyaudio
from loguru import logger

from config import config
from core.persona import persona_manager

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Callback signature: receives the name of the detected wake word ("jarvis" | "friday")
WakeCallback = Callable[[str], None]

# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------

class WakeWordListener:
    """
    Listens on the default microphone for wake words using Porcupine.

    Two built-in keywords are used: 'jarvis' and 'friday'.
    When either is detected, *on_wake* is called with the keyword name.

    The audio stream runs in a background daemon thread; the main thread
    remains unblocked. CPU load is kept low because Porcupine processes
    audio at 16 kHz in fixed 512-sample frames and the loop sleeps while
    waiting for PyAudio to fill the buffer.
    """

    # Porcupine requires exactly these values
    _SAMPLE_RATE = 16_000
    _CHANNELS = 1
    _FORMAT = pyaudio.paInt16

    def __init__(
        self,
        on_wake: Optional[WakeCallback] = None,
        sensitivity: float | list[float] = 0.5,
    ) -> None:
        self._on_wake = on_wake or self._default_callback
        self._sensitivity = sensitivity

        self._porcupine: Optional[pvporcupine.Porcupine] = None
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Default callback
    # ------------------------------------------------------------------

    @staticmethod
    def _default_callback(wake_word: str) -> None:
        logger.info(f"Wake word detected: '{wake_word}'")

    # ------------------------------------------------------------------
    # Porcupine initialisation
    # ------------------------------------------------------------------

    def _build_porcupine(self) -> pvporcupine.Porcupine:
        access_key = config.PORCUPINE_API_KEY or config.WAKE_WORD_ACCESS_KEY
        if not access_key:
            raise ValueError(
                "Porcupine access key not set. "
                "Add PORCUPINE_API_KEY to your .env file."
            )

        keywords = ["jarvis", "friday"]
        sensitivities = (
            self._sensitivity
            if isinstance(self._sensitivity, list)
            else [self._sensitivity] * len(keywords)
        )

        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=keywords,
            sensitivities=sensitivities,
        )
        logger.info(
            f"Porcupine initialised — "
            f"keywords={keywords}, sensitivities={sensitivities}"
        )
        return porcupine

    # ------------------------------------------------------------------
    # Audio stream helpers
    # ------------------------------------------------------------------

    def _open_stream(self, audio: pyaudio.PyAudio, frame_length: int):
        return audio.open(
            rate=self._SAMPLE_RATE,
            channels=self._CHANNELS,
            format=self._FORMAT,
            input=True,
            frames_per_buffer=frame_length,
        )

    # ------------------------------------------------------------------
    # Listener loop
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        logger.info("Wake word listener started.")
        _KEYWORD_LABELS = ["jarvis", "friday"]

        try:
            self._porcupine = self._build_porcupine()
            frame_length = self._porcupine.frame_length  # 512 samples

            self._audio = pyaudio.PyAudio()
            self._stream = self._open_stream(self._audio, frame_length)

            while not self._stop_event.is_set():
                raw = self._stream.read(frame_length, exception_on_overflow=False)
                # Unpack int16 PCM samples
                pcm = struct.unpack_from(f"{frame_length}h", raw)

                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    detected = _KEYWORD_LABELS[keyword_index]
                    logger.success(f"Wake word '{detected}' detected!")

                    # Auto-switch persona to the detected keyword
                    try:
                        persona_manager.switch(detected.upper())
                    except ValueError:
                        pass

                    try:
                        self._on_wake(detected)
                    except Exception as cb_err:
                        logger.error(f"Wake callback raised an exception: {cb_err}")

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Listener loop error: {e}")
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None

        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None

        logger.info("Wake word listener cleaned up.")

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Listener is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="WakeWordListener",
            daemon=True,          # dies automatically with the main process
        )
        self._thread.start()
        logger.info("Wake word listener thread started.")

    def stop(self) -> None:
        """Signal the listener to stop and block until the thread exits."""
        if self._thread is None or not self._thread.is_alive():
            return

        logger.info("Stopping wake word listener...")
        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logger.warning("Listener thread did not stop within timeout.")
        else:
            logger.info("Wake word listener stopped cleanly.")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<WakeWordListener [{state}]>"
