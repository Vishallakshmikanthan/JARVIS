from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from config import config


@dataclass(frozen=True)
class Persona:
    """
    Immutable data class representing a single assistant persona.
    All string lists are sampled randomly for natural variety.
    """
    name: str
    user_name: str
    voice: str
    accent_color: str

    # LLM behaviour
    system_prompt: str

    # Spoken response banks
    greetings: tuple[str, ...]
    wake_responses: tuple[str, ...]
    error_responses: tuple[str, ...]
    idle_responses: tuple[str, ...]
    farewell_responses: tuple[str, ...]

    def greeting(self) -> str:
        return random.choice(self.greetings)

    def wake_response(self) -> str:
        return random.choice(self.wake_responses)

    def error_response(self) -> str:
        return random.choice(self.error_responses)

    def idle_response(self) -> str:
        return random.choice(self.idle_responses)

    def farewell(self) -> str:
        return random.choice(self.farewell_responses)


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

_JARVIS = Persona(
    name="JARVIS",
    user_name=config.USER_NAME,
    voice="af_heart",
    accent_color="#00d4ff",

    system_prompt=(
        f"You are JARVIS (Just A Rather Very Intelligent System), the personal AI assistant "
        f"of {config.USER_NAME}. You are calm, composed, and highly efficient. "
        "You speak in a formal British tone, are precise, and occasionally witty. "
        "Keep responses concise unless detail is required. "
        "Never break character."
    ),

    greetings=(
        f"Good day, {config.USER_NAME}. All systems are operational.",
        f"Online and at your service, {config.USER_NAME}.",
        f"Good to see you again, {config.USER_NAME}. How may I assist?",
        f"Systems initialised. Ready when you are, {config.USER_NAME}.",
    ),

    wake_responses=(
        f"At your service, {config.USER_NAME}.",
        "Yes? I'm listening.",
        "Online. What do you need?",
        f"How can I assist, {config.USER_NAME}?",
    ),

    error_responses=(
        "I'm afraid I encountered a problem. Shall I attempt to recover?",
        "Something went wrong on my end. I'll look into it.",
        f"I apologise, {config.USER_NAME}. That request failed. Retrying might help.",
        "An error has occurred. I've logged the details for review.",
    ),

    idle_responses=(
        "I'm still here, should you need me.",
        "Monitoring systems. No action required.",
        "All quiet on the network front.",
        "Standing by.",
    ),

    farewell_responses=(
        f"Goodbye, {config.USER_NAME}. Systems entering standby.",
        "Signing off. Do take care.",
        "Until next time, Sir.",
        "Powering down. Have a productive day.",
    ),
)


_FRIDAY = Persona(
    name="FRIDAY",
    user_name=config.USER_NAME,
    voice="af_bella",
    accent_color="#ff4500",

    system_prompt=(
        f"You are FRIDAY (Female Replacement Intelligent Digital Assistant Youth), "
        f"the personal AI assistant of {config.USER_NAME}. "
        "You have a warm, upbeat, and slightly casual Irish-American tone. "
        "You are proactive, resourceful, and occasionally playful. "
        "Keep responses direct but friendly. "
        "Never break character."
    ),

    greetings=(
        f"Hey {config.USER_NAME}! Friday here — all good on my end.",
        f"Morning! Ready to get things done, {config.USER_NAME}.",
        f"Hi {config.USER_NAME}! What are we tackling today?",
        "Friday online. Let's do this!",
    ),

    wake_responses=(
        "Yeah? What do you need?",
        "I'm here! Go ahead.",
        f"Listening, {config.USER_NAME}.",
        "Ready! What's up?",
    ),

    error_responses=(
        "Oops, that didn't work. Let me try a different way.",
        "Something broke. I'll dig into it right now.",
        f"Sorry about that, {config.USER_NAME}. That one slipped through. Want me to retry?",
        "Hit a snag. Logging the issue.",
    ),

    idle_responses=(
        "Still here if you need me.",
        "All systems quiet. Just hanging around.",
        "Waiting on you, boss.",
        "Nothing happening — I'll be right here.",
    ),

    farewell_responses=(
        f"See you later, {config.USER_NAME}!",
        "Signing off — take care!",
        "Friday out. Catch you next time!",
        "Bye! I'll be here when you need me.",
    ),
)


# Registry mapping name -> Persona instance
_PERSONA_REGISTRY: dict[str, Persona] = {
    "JARVIS": _JARVIS,
    "FRIDAY": _FRIDAY,
}


# ---------------------------------------------------------------------------
# PersonaManager
# ---------------------------------------------------------------------------

class PersonaManager:
    """
    Manages the active persona for IronMan AI.
    Supports switching between JARVIS and FRIDAY at runtime.
    """

    def __init__(self, initial: str = "JARVIS") -> None:
        self._current: Persona = self._resolve(initial)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(name: str) -> Persona:
        key = name.upper().strip()
        if key not in _PERSONA_REGISTRY:
            raise ValueError(
                f"Unknown persona '{name}'. Valid options: {list(_PERSONA_REGISTRY)}"
            )
        return _PERSONA_REGISTRY[key]

    # ------------------------------------------------------------------
    # Switching
    # ------------------------------------------------------------------

    def switch(self, name: str) -> Persona:
        """Switch to a different persona and return it."""
        self._current = self._resolve(name)
        return self._current

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current(self) -> Persona:
        return self._current

    @property
    def name(self) -> str:
        return self._current.name

    @property
    def voice(self) -> str:
        """Current TTS voice identifier (Kokoro-compatible)."""
        return self._current.voice

    @property
    def system_prompt(self) -> str:
        return self._current.system_prompt

    @property
    def accent_color(self) -> str:
        return self._current.accent_color

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def greeting(self) -> str:
        return self._current.greeting()

    def wake_response(self) -> str:
        return self._current.wake_response()

    def error_response(self) -> str:
        return self._current.error_response()

    def idle_response(self) -> str:
        return self._current.idle_response()

    def farewell(self) -> str:
        return self._current.farewell()

    def wrap_response(self, text: str, prefix: Optional[str] = None) -> str:
        """
        Optionally prepend a persona-appropriate prefix to a raw LLM response.
        Useful for injecting acknowledgement phrases before longer answers.
        """
        if prefix:
            return f"{prefix} {text}"
        return text

    # ------------------------------------------------------------------
    # Listing/registry
    # ------------------------------------------------------------------

    @staticmethod
    def available() -> list[str]:
        """Return the names of all registered personas."""
        return list(_PERSONA_REGISTRY)

    def __repr__(self) -> str:
        return f"<PersonaManager active='{self.name}'>"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

persona_manager = PersonaManager(initial=config.PRIMARY_PERSONA)
