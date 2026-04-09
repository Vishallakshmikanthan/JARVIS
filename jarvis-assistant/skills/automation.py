from __future__ import annotations

"""
skills/automation.py
Schedule recurring or one-off tasks using the `schedule` library.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import schedule
from loguru import logger

# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

@dataclass
class ScheduledJob:
    label: str
    description: str
    tag: str                        # unique schedule tag
    created_at: float = field(default_factory=time.time)

_jobs: dict[str, ScheduledJob] = {}      # label → ScheduledJob
_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Scheduler background thread
# ---------------------------------------------------------------------------

def _start_scheduler() -> None:
    """Start the schedule run-loop in a daemon thread (idempotent)."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return

    _stop_event.clear()

    def _loop():
        logger.info("Automation scheduler started.")
        while not _stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
        logger.info("Automation scheduler stopped.")

    _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="AutomationScheduler")
    _scheduler_thread.start()


def stop_scheduler() -> None:
    """Stop the background scheduler thread cleanly."""
    _stop_event.set()
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Duration / interval parsers
# ---------------------------------------------------------------------------

_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour":   3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day":    86400, "days": 86400,
}

def _parse_interval(text: str) -> Optional[tuple[int, str]]:
    """
    Return ``(quantity, unit_word)`` from text like "every 5 minutes".
    Returns None if no match found.
    """
    match = re.search(
        r"every\s+(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs|day|days)",
        text, re.I
    )
    if match:
        return int(match.group(1)), match.group(2).lower()

    # "every minute / every hour / every day" (no number)
    simple = re.search(
        r"every\s+(minute|hour|day)",
        text, re.I
    )
    if simple:
        return 1, simple.group(1).lower()

    return None


def _parse_time_of_day(text: str) -> Optional[str]:
    """Return HH:MM string from "at 09:30" or "at 9am" etc., or None."""
    # "at 09:30" or "at 9:00"
    match = re.search(r"at\s+(\d{1,2}):(\d{2})\s*(am|pm)?", text, re.I)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        suffix = (match.group(3) or "").lower()
        if suffix == "pm" and h < 12:
            h += 12
        elif suffix == "am" and h == 12:
            h = 0
        return f"{h:02d}:{m:02d}"

    # "at 9am" / "at 3 pm"
    match = re.search(r"at\s+(\d{1,2})\s*(am|pm)", text, re.I)
    if match:
        h = int(match.group(1))
        suffix = match.group(2).lower()
        if suffix == "pm" and h < 12:
            h += 12
        elif suffix == "am" and h == 12:
            h = 0
        return f"{h:02d}:00"

    return None


# ---------------------------------------------------------------------------
# Built-in task actions
# ---------------------------------------------------------------------------

def _make_speak_action(message: str) -> Callable:
    """Create a job action that speaks *message* via TTS."""
    def _action():
        logger.info(f"Automation trigger: {message}")
        try:
            from voice.tts import tts
            tts.speak(message)
        except Exception:
            print(f"\n[AUTOMATION] {message}\n")
    return _action


# ---------------------------------------------------------------------------
# Public skill function
# ---------------------------------------------------------------------------

def schedule_task(user_input: str) -> str:
    """
    Schedule, list, or cancel automated tasks via natural language.

    Examples::

        "schedule a reminder every 30 minutes to drink water"
        "remind me every day at 9am to check emails"
        "remind me at 14:30 to take medicine"
        "list scheduled tasks"
        "cancel task drink water"
        "stop all tasks"
    """
    text = user_input.strip()
    lower = text.lower()

    # Start the scheduler thread on first use
    _start_scheduler()

    # --- List tasks ---
    if re.search(r"\b(list|show|what|active|scheduled)\b", lower):
        if not _jobs:
            return "No tasks are currently scheduled."
        lines = ["Scheduled tasks:"]
        for label, job in _jobs.items():
            lines.append(f"  • {label}: {job.description}")
        return "\n".join(lines)

    # --- Cancel / stop ---
    if re.search(r"\b(cancel|remove|delete|stop)\b", lower):
        # "stop all"
        if re.search(r"\b(all|every|everything)\b", lower):
            for job in _jobs.values():
                schedule.clear(job.tag)
            count = len(_jobs)
            _jobs.clear()
            return f"Cancelled {count} scheduled task(s)."

        # find by label keyword
        cancelled = []
        for label in list(_jobs.keys()):
            if any(word in lower for word in label.lower().split()):
                schedule.clear(_jobs[label].tag)
                del _jobs[label]
                cancelled.append(label)
        if cancelled:
            return f"Cancelled task(s): {', '.join(cancelled)}."
        return "I couldn't find a matching task to cancel."

    # --- Schedule a new task ---
    # Extract the message / action to announce
    msg_match = re.search(
        r"\bto\s+(.+?)(?:\s+every\b|\s+at\b|$)",
        lower
    )
    message = msg_match.group(1).strip().capitalize() if msg_match else "Reminder"
    label = message[:30]

    action = _make_speak_action(message)
    tag = f"task_{len(_jobs)}_{int(time.time())}"

    # --- One-off "at HH:MM" schedule ---
    tod = _parse_time_of_day(lower)
    if tod and "every" not in lower:
        # "daily at HH:MM" one-time approximation using schedule
        schedule.every().day.at(tod).do(action).tag(tag)
        desc = f"Daily at {tod}: {message}"
        _jobs[label] = ScheduledJob(label=label, description=desc, tag=tag)
        logger.info(f"Scheduled daily at {tod}: '{message}'")
        return f"Task scheduled to run daily at {tod}: '{message}'."

    # --- Interval-based ---
    interval = _parse_interval(lower)
    if interval:
        qty, unit = interval
        base_unit = unit.rstrip("s")   # e.g. "minutes" → "minute"

        job_builder = getattr(schedule.every(qty), base_unit, None)
        if job_builder is None:
            return f"I don't know how to schedule every {qty} {unit}."

        if tod:
            job_builder.at(tod).do(action).tag(tag)
            desc = f"Every {qty} {unit} at {tod}: {message}"
        else:
            job_builder.do(action).tag(tag)
            desc = f"Every {qty} {unit}: {message}"

        _jobs[label] = ScheduledJob(label=label, description=desc, tag=tag)
        logger.info(f"Scheduled: {desc}")
        return f"Task scheduled — {desc}."

    return (
        "I couldn't parse a schedule from that. "
        "Try: 'remind me every 30 minutes to drink water' "
        "or 'schedule a reminder at 9am to check emails'."
    )
