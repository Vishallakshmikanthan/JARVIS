"""
main.py — IronMan AI entry point

Usage:
    python main.py            # voice mode  (wake-word -> STT pipeline)
    python main.py --text     # text mode   (type commands in terminal)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import os

# ---------------------------------------------------------------------------
# Ensure the project root is always importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"


def _print(label: str, text: str, color: str = _C.CYAN) -> None:
    """Consistent coloured terminal line:  [LABEL] text"""
    print(f"{color}{_C.BOLD}[{label}]{_C.RESET} {text}")


def _banner(name: str, version: str, persona: str) -> None:
    line = "\u2500" * 52
    print(f"\n{_C.CYAN}{line}")
    print(f"  {_C.BOLD}{name}  v{version}{_C.RESET}{_C.CYAN}  |  Persona: {_C.MAGENTA}{persona}")
    print(f"{_C.CYAN}{line}{_C.RESET}\n")


# ---------------------------------------------------------------------------
# Skill dispatcher
# ---------------------------------------------------------------------------

def _dispatch(route) -> str:
    """
    Dynamically import the module referenced by *route* and call its handler
    with route.parsed_input. Falls back to brain.think() on any error.
    """
    try:
        mod = importlib.import_module(route.module)
        fn  = getattr(mod, route.function)
        result = fn(route.parsed_input)
        return result if isinstance(result, str) else str(result)
    except Exception as exc:
        from loguru import logger
        logger.warning(f"Dispatch failed for {route}: {exc}")
        from core.brain import Brain
        return Brain().think(route.parsed_input)


# ---------------------------------------------------------------------------
# Core initialisation
# ---------------------------------------------------------------------------

def _init_all(text_mode: bool) -> dict:
    """Initialise all subsystems and return them in a context dict."""

    _print("INIT", "Starting subsystem initialisation...", _C.BLUE)

    # Memory DB
    from core.memory import init_db
    init_db()
    _print("INIT", "Memory database ready.", _C.GREEN)

    # Persona
    from core.persona import persona_manager
    _print("INIT", f"Persona: {persona_manager.name}", _C.GREEN)

    # Brain (LLM)
    from core.brain import Brain
    brain = Brain()
    _print("INIT", f"Brain online  model={brain.model}", _C.GREEN)

    # Router
    from core.router import router
    _print("INIT", "Router ready.", _C.GREEN)

    # TTS
    from voice.tts import TextToSpeech
    tts = TextToSpeech()
    _print("INIT", "TTS engine initialised.", _C.GREEN)

    # STT + Wake-word (voice mode only)
    stt = None
    wake_listener = None
    if not text_mode:
        from voice.stt import SpeechToText
        stt = SpeechToText()
        _print("INIT", "STT engine initialised.", _C.GREEN)

        from voice.listen import WakeWordListener
        wake_listener = WakeWordListener()
        _print("INIT", "Wake-word listener ready.", _C.GREEN)

    return {
        "persona_manager": persona_manager,
        "brain":           brain,
        "router":          router,
        "tts":             tts,
        "stt":             stt,
        "wake_listener":   wake_listener,
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _authenticate() -> bool:
    """Run face auth when enabled; return True immediately when disabled."""
    from config import config
    if not config.FACE_AUTH_ENABLED:
        return True

    _print("AUTH", "Face authentication required...", _C.YELLOW)
    from auth.face import verify
    authenticated = verify()
    if authenticated:
        _print("AUTH", "Identity confirmed.", _C.GREEN)
    else:
        _print("AUTH", "Authentication failed.", _C.RED)
    return authenticated


# ---------------------------------------------------------------------------
# Single-turn processing
# ---------------------------------------------------------------------------

def _process(user_text: str, ctx: dict) -> str:
    """Route input -> skill/LLM -> log to memory -> return response string."""
    from core.memory import log_interaction

    log_interaction("user", user_text, persona=ctx["persona_manager"].name)

    route = ctx["router"].route(user_text)
    _print("ROUTE", str(route), _C.DIM)

    response = _dispatch(route)

    log_interaction("assistant", response, persona=ctx["persona_manager"].name)
    return response


# ---------------------------------------------------------------------------
# Voice loop
# ---------------------------------------------------------------------------

def _voice_loop(ctx: dict) -> None:
    """Block forever: wake word -> auth -> STT -> process -> TTS."""
    import time

    _print("MODE", "Voice mode active. Listening for wake word...", _C.CYAN)

    wake_detected = False

    def _on_wake(word: str) -> None:
        nonlocal wake_detected
        _print("WAKE", f"'{word}' detected.", _C.MAGENTA)
        wake_detected = True

    wake_listener = ctx["wake_listener"]
    wake_listener._on_wake = _on_wake
    wake_listener.start()

    try:
        while True:
            if not wake_detected:
                time.sleep(0.05)
                continue

            wake_detected = False

            # Spoken acknowledgement
            wake_line = ctx["persona_manager"].wake_response()
            _print(ctx["persona_manager"].name, wake_line, _C.CYAN)
            ctx["tts"].speak(wake_line)

            # Auth gate
            if not _authenticate():
                deny = ctx["persona_manager"].error_response()
                _print(ctx["persona_manager"].name, deny, _C.RED)
                ctx["tts"].speak(deny)
                continue

            # Capture speech
            _print("STT", "Listening...", _C.YELLOW)
            user_text = ctx["stt"].listen(prompt=False)
            if not user_text:
                _print("STT", "No speech detected. Returning to standby.", _C.DIM)
                continue

            _print("YOU", user_text, _C.GREEN)

            # Route + respond
            response = _process(user_text, ctx)
            _print(ctx["persona_manager"].name, response, _C.CYAN)
            ctx["tts"].speak(response)

    except KeyboardInterrupt:
        _print("SYS", "Interrupt received. Shutting down...", _C.YELLOW)
    finally:
        wake_listener.stop()


# ---------------------------------------------------------------------------
# Text loop
# ---------------------------------------------------------------------------

def _text_loop(ctx: dict) -> None:
    """Interactive text REPL for --text mode."""
    _print("MODE", "Text mode active. Type your command (Ctrl+C or 'exit' to quit).", _C.CYAN)

    while True:
        try:
            user_text = input(f"\n{_C.GREEN}{_C.BOLD}You > {_C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "bye"}:
            farewell = ctx["persona_manager"].farewell()
            _print(ctx["persona_manager"].name, farewell, _C.CYAN)
            ctx["tts"].speak(farewell)
            break

        response = _process(user_text, ctx)
        _print(ctx["persona_manager"].name, response, _C.CYAN)
        ctx["tts"].speak(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IronMan AI — JARVIS / FRIDAY personal assistant"
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Run in text-input mode instead of voice mode."
    )
    args = parser.parse_args()

    from config import config

    _banner(config.APP_NAME, config.VERSION, config.ASSISTANT_NAME)

    ctx = _init_all(text_mode=args.text)

    # Startup greeting
    greeting = ctx["persona_manager"].greeting()
    _print(ctx["persona_manager"].name, greeting, _C.CYAN)
    ctx["tts"].speak(greeting)

    # Dispatch to the appropriate loop
    if args.text:
        _text_loop(ctx)
    else:
        _voice_loop(ctx)

    _print("SYS", "Goodbye.", _C.YELLOW)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{_C.RED}{_C.BOLD}[FATAL]{_C.RESET} {exc}")
        sys.exit(1)

