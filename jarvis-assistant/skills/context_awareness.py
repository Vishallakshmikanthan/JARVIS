from __future__ import annotations

"""
skills/context_awareness.py
Tracks conversation intent across multiple turns, providing continuity hints
to other skills and the LLM brain.
"""

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from core.memory import session_ctx

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn with its resolved intent."""
    role: str        # "user" | "assistant"
    text: str
    intent: str      # resolved intent label
    timestamp: float = field(default_factory=time.time)


# A sliding window of the last N turns (default 20)
_MAX_TURNS = 20
_history: deque[Turn] = deque(maxlen=_MAX_TURNS)

# Running counts of how often each intent has appeared this session
_session_intent_counts: dict[str, int] = {}

# The most recently active intent (used as a tie-breaker for ambiguous input)
_last_intent: Optional[str] = None


# ---------------------------------------------------------------------------
# Intent keywords used for lightweight local classification
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, re.Pattern] = {
    "weather":     re.compile(r"\b(weather|temperature|forecast|rain|sunny)\b", re.I),
    "news":        re.compile(r"\b(news|headline|breaking|update)\b", re.I),
    "search":      re.compile(r"\b(search|google|look\s?up|find)\b", re.I),
    "wiki":        re.compile(r"\b(wiki|who\s+is|what\s+is|explain|define)\b", re.I),
    "music":       re.compile(r"\b(play|pause|skip|spotify|track|song)\b", re.I),
    "system":      re.compile(r"\b(cpu|ram|battery|disk|screenshot|volume)\b", re.I),
    "joke":        re.compile(r"\b(joke|funny|laugh|humou?r)\b", re.I),
    "timer":       re.compile(r"\b(timer|alarm|countdown|remind\s+me)\b", re.I),
    "calculate":   re.compile(r"\b(calculat|math|compute|\d+\s*[\+\-\*/]\s*\d+)\b", re.I),
    "automation":  re.compile(r"\b(schedule|automat|every\s+day|daily|cron)\b", re.I),
    "file":        re.compile(r"\b(file|folder|read|write|save|open|directory|document)\b", re.I),
    "module":      re.compile(r"\b(enable|disable|module|skill|turn\s+on|turn\s+off)\b", re.I),
    "persona":     re.compile(r"\b(switch|activate|become|change)\s+(jarvis|friday)\b", re.I),
}


def _classify_intent(text: str) -> str:
    """Lightweight local intent classification (no LLM call)."""
    for intent, pattern in _INTENT_KEYWORDS.items():
        if pattern.search(text):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Public API used by other modules
# ---------------------------------------------------------------------------

def record_turn(role: str, text: str, intent: Optional[str] = None) -> Turn:
    """
    Add a turn to the conversation history.
    If *intent* is not provided it is inferred automatically.
    Returns the created Turn.
    """
    global _last_intent
    resolved = intent or _classify_intent(text)
    turn = Turn(role=role, text=text, intent=resolved)
    _history.append(turn)
    _session_intent_counts[resolved] = _session_intent_counts.get(resolved, 0) + 1
    _last_intent = resolved
    logger.debug(f"Context recorded [{role}] intent={resolved}: {text[:60]}")
    return turn


def get_last_intent() -> Optional[str]:
    """Return the intent from the most recent turn."""
    return _last_intent


def get_dominant_intent(window: int = 5) -> Optional[str]:
    """Return the most frequent intent within the last *window* turns."""
    recent = list(_history)[-window:]
    if not recent:
        return None
    counts: dict[str, int] = {}
    for turn in recent:
        counts[turn.intent] = counts.get(turn.intent, 0) + 1
    return max(counts, key=counts.__getitem__)


def get_context_summary(window: int = 6) -> str:
    """
    Return a compact plain-text summary of the recent turns.
    Suitable for injecting into an LLM prompt.
    """
    recent = list(_history)[-window:]
    if not recent:
        return ""
    lines = ["[Conversation context]"]
    for t in recent:
        ts = time.strftime("%H:%M:%S", time.localtime(t.timestamp))
        lines.append(f"  [{ts}] {t.role.upper()} ({t.intent}): {t.text[:80]}")
    return "\n".join(lines)


def get_history(limit: int = _MAX_TURNS) -> list[Turn]:
    """Return the last *limit* turns as a list."""
    return list(_history)[-limit:]


def clear_context() -> None:
    """Reset the conversation history, all counters, and the router session context."""
    _history.clear()
    _session_intent_counts.clear()
    global _last_intent
    _last_intent = None
    session_ctx.clear()   # keep router context in sync
    logger.info("Conversation context cleared.")


def session_stats() -> dict[str, int]:
    """Return session-wide intent frequency counts."""
    return dict(_session_intent_counts)


# ---------------------------------------------------------------------------
# Public skill function
# ---------------------------------------------------------------------------

def track_context(user_input: str) -> str:
    """
    Natural-language interface for inspecting conversation context.

    Examples::

        "what have we been talking about"
        "show conversation context"
        "what's the current topic"
        "clear context"
        "session stats"
    """
    text = user_input.lower().strip()

    if re.search(r"\b(clear|reset|forget|wipe)\b", text):
        clear_context()
        return "Conversation context has been cleared."

    if re.search(r"\b(stat|count|frequency|how many|session)\b", text):
        stats = session_stats()
        if not stats:
            return "No session activity recorded yet."
        lines = ["Session intent frequency:"]
        for intent, count in sorted(stats.items(), key=lambda x: -x[1]):
            lines.append(f"  {intent}: {count} turn(s)")
        return "\n".join(lines)

    if re.search(r"\b(topic|dominant|main|focus|what\s+are\s+we)\b", text):
        dominant = get_dominant_intent()
        last     = get_last_intent()
        ctx_entities = session_ctx.last_entities
        return (
            f"Most recent intent: {last or 'unknown'}. "
            f"Dominant topic (last 5 turns): {dominant or 'none'}. "
            + (f"Active entities: '{ctx_entities}'." if ctx_entities else "")
        ).strip()

    # Default — show context summary
    summary = get_context_summary()
    if not summary:
        return "No conversation history recorded yet."
    return summary
