from __future__ import annotations

import re
import threading
from typing import Callable, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Active timers registry (supports multiple concurrent timers)
# ---------------------------------------------------------------------------

_active_timers: dict[str, threading.Timer] = {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TIME_PATTERNS = [
    (r"(\d+)\s*hour",   3600),
    (r"(\d+)\s*min",    60),
    (r"(\d+)\s*sec",    1),
    (r"(\d+)\s*h\b",    3600),
    (r"(\d+)\s*m\b",    60),
    (r"(\d+)\s*s\b",    1),
]


def _parse_duration(text: str) -> int:
    """Return total seconds from a natural-language duration string."""
    total = 0
    for pattern, multiplier in _TIME_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            total += int(match.group(1)) * multiplier
    return total


def _format_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h > 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m > 1 else ''}")
    if s:
        parts.append(f"{s} second{'s' if s > 1 else ''}")
    return ", ".join(parts) if parts else "0 seconds"


# ---------------------------------------------------------------------------
# Timer actions
# ---------------------------------------------------------------------------

def _fire(label: str, callback: Optional[Callable] = None) -> None:
    """Called when a timer elapses."""
    logger.success(f"Timer '{label}' elapsed!")
    _active_timers.pop(label, None)

    # Try to speak the notification if TTS is available
    try:
        from voice.tts import tts
        tts.speak(f"Timer complete: {label}")
    except Exception:
        print(f"\n⏰ TIMER DONE: {label}\n")

    if callback:
        try:
            callback(label)
        except Exception as e:
            logger.error(f"Timer callback error: {e}")


# ---------------------------------------------------------------------------
# Public skill function
# ---------------------------------------------------------------------------

def set_timer(user_input: str) -> str:
    """
    Set, list, or cancel a countdown timer.
    Examples:
        "set a timer for 5 minutes"
        "timer for 1 hour 30 minutes"
        "cancel timer"
        "list timers"
        "how many timers"
    """
    text = user_input.lower().strip()

    # --- List timers ---
    if re.search(r"\b(list|show|active|how many)\b", text):
        if not _active_timers:
            return "No active timers."
        names = list(_active_timers.keys())
        return f"{len(names)} active timer(s): {', '.join(names)}."

    # --- Cancel timer ---
    if re.search(r"\b(cancel|stop|remove|delete)\b", text):
        if not _active_timers:
            return "There are no active timers to cancel."
        # Cancel all or a named one
        label_match = re.search(r"cancel\s+(.+)", text)
        if label_match:
            label = label_match.group(1).strip()
            t = _active_timers.pop(label, None)
            if t:
                t.cancel()
                return f"Timer '{label}' cancelled."
            return f"I couldn't find a timer called '{label}'."
        # Cancel all
        for t in _active_timers.values():
            t.cancel()
        count = len(_active_timers)
        _active_timers.clear()
        return f"Cancelled {count} timer(s)."

    # --- Set timer ---
    duration = _parse_duration(text)
    if duration <= 0:
        return (
            "I couldn't parse a duration. "
            "Try: 'Set a timer for 10 minutes' or 'timer for 1 hour 30 seconds'."
        )

    # Extract optional label: "set timer called <name> for ..."
    label_match = re.search(r"(?:called|named|label)\s+(['\"]?)(\w+)\1", text)
    label = label_match.group(2) if label_match else _format_duration(duration)

    # Cancel any existing timer with the same label
    if label in _active_timers:
        _active_timers[label].cancel()

    timer = threading.Timer(duration, _fire, args=(label,))
    timer.daemon = True
    timer.name = f"Timer-{label}"
    timer.start()
    _active_timers[label] = timer

    human_duration = _format_duration(duration)
    logger.info(f"Timer set: '{label}' for {human_duration}")
    return f"Timer set for {human_duration}. I'll let you know when it's done."
