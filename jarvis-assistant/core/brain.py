from __future__ import annotations

import functools
from typing import Optional

import ollama
from loguru import logger

from config import config
from core.persona import persona_manager
from core.memory import (
    log_interaction,
    get_conversation_context,
)

# Knowledge Base is imported lazily inside think() to avoid circular-import
# issues during startup — but we expose a module-level convenience function.
@functools.lru_cache(maxsize=256)
def ask_local_knowledge(query: str) -> Optional[str]:
    """Proxy to the KnowledgeBase singleton — cached by query string."""
    from core.knowledge_base import knowledge_base
    return knowledge_base.ask_local_knowledge(query)

# ---------------------------------------------------------------------------
# LLM response cache
# A bounded session cache for deterministic/factual LLM answers.
# Key: (model, system_prompt_hash, user_input)
# Only caches single-turn queries (no multi-turn history dependence).
# ---------------------------------------------------------------------------
_LLM_CACHE: dict[str, str] = {}
_LLM_CACHE_MAX = 128   # evict oldest entry when full

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
        Routing order:
          1. Local Knowledge Base (semantic search over indexed PDFs / notes) — LRU cached.
          2. LLM response cache  (exact-match session cache for repeated queries).
          3. Ollama LLM          (with persona system prompt + conversation history).

        Both the user message and the response are persisted to memory.
        """
        # --- 1. Try local knowledge base (cached) ---
        try:
            if kb_answer := ask_local_knowledge(user_input):
                logger.info("Brain: answered from local knowledge base.")
                log_interaction("user",      user_input, persona=persona_manager.name)
                log_interaction("assistant", kb_answer,  persona=persona_manager.name)
                return kb_answer
        except Exception as e:
            logger.warning(f"Brain: KB lookup failed, falling through to LLM: {e}")

        # --- 2. LLM response cache (exact-match, same persona + model) ---
        _cache_key = f"{self.model}|{persona_manager.name}|{user_input}"
        if cached := _LLM_CACHE.get(_cache_key):
            logger.debug(f"Brain: LLM cache hit for '{user_input[:60]}'")
            return cached

        # --- 3. Fall through to Ollama LLM ---
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

        # Populate LLM cache (evict oldest entry when at capacity)
        if len(_LLM_CACHE) >= _LLM_CACHE_MAX:
            _LLM_CACHE.pop(next(iter(_LLM_CACHE)))
        _LLM_CACHE[_cache_key] = reply

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
