from __future__ import annotations

import random
from typing import Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Jokes Skill (API + Fallbacks)
# ---------------------------------------------------------------------------

_FALLBACK_JOKES: list[str] = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "Did you hear about the claustrophobic astronaut? He just needed a little space.",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
]

_API_URL = "https://icanhazdadjoke.com/"
_HEADERS = {"Accept": "application/json", "User-Agent": "Jarvis-Assistant/1.0"}

def _fetch_api_joke() -> Optional[str]:
    try:
        response = requests.get(_API_URL, headers=_HEADERS, timeout=5)
        response.raise_for_status()
        return response.json().get("joke")
    except Exception as e:
        logger.warning(f"Joke API unavailable: {e}")
        return None

def tell_joke(user_input: str) -> str:
    """
    Tell a funny joke.
    Examples: "tell me a joke", "say something funny"
    """
    joke = _fetch_api_joke()
    if not joke:
        joke = random.choice(_FALLBACK_JOKES)
        
    return joke
