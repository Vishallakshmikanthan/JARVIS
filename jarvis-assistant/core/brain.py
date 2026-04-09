from __future__ import annotations

from typing import Optional

import ollama
from loguru import logger

from config import config
from core.persona import persona_manager
from core.memory import (
    log_interaction,
    get_conversation_context,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
CONTEXT_WINDOW = 10  # number of recent turns to feed as context


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------

class Brain:
    """
    Central reasoning engine.
    Wraps Ollama chat completions with persona-aware system prompts
    and persistent conversation memory.
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Point the ollama client at the configured host
        self._client = ollama.Client(host=base_url)
        logger.info(f"Brain initialised → model={self.model}, host={base_url}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list[dict]:
        """
        Assemble the full message list:
            1. System prompt (from current persona)
            2. Recent conversation history (from memory DB)
            3. Current user message
        """
        system_msg = {"role": "system", "content": persona_manager.system_prompt}

        history = get_conversation_context(limit=CONTEXT_WINDOW)

        current_msg = {"role": "user", "content": user_input}

        return [system_msg] + history + [current_msg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def think(self, user_input: str) -> str:
        """
        Send *user_input* to the LLM and return the assistant's reply.
        Both the user message and the response are persisted to memory.
        """
        messages = self._build_messages(user_input)

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Ollama request failed ({error_type}): {e}")

            if "refused" in str(e).lower() or "connection" in str(e).lower():
                return (
                    "I'm unable to reach Ollama right now. "
                    "Please make sure it's running with: ollama serve"
                )

            return persona_manager.error_response()

        reply: str = response["message"]["content"].strip()

        # Persist both turns
        log_interaction("user", user_input, persona=persona_manager.name)
        log_interaction("assistant", reply, persona=persona_manager.name)

        logger.debug(
            f"think() → {len(reply)} chars "
            f"(model={self.model}, temp={self.temperature})"
        )
        return reply

    # ------------------------------------------------------------------
    # Runtime tunables
    # ------------------------------------------------------------------

    def set_temperature(self, value: float) -> None:
        self.temperature = max(0.0, min(value, 2.0))
        logger.info(f"Temperature set to {self.temperature}")

    def set_max_tokens(self, value: int) -> None:
        self.max_tokens = max(64, value)
        logger.info(f"Max tokens set to {self.max_tokens}")

    def set_model(self, model: str) -> None:
        self.model = model
        logger.info(f"Model switched to {self.model}")

    def __repr__(self) -> str:
        return (
            f"<Brain model='{self.model}' "
            f"temp={self.temperature} "
            f"max_tokens={self.max_tokens}>"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
brain = Brain()
