from __future__ import annotations

"""
core/autonomous.py
Background intelligence loop for Autonomous Mode.

Activate with:  "go autonomous" / "autonomous mode on"
Deactivate with: "stop autonomous" / "disengage"
Check status:   "autonomous status"

Monitors (on a configurable poll interval):
  - Work session duration  → break reminders every 2 hours
  - CPU / RAM usage        → alerts above configurable thresholds
  - Battery level          → low/critical warnings when unplugged
  - Pending reminders      → fires user-scheduled reminders at the due time

All proactive messages are spoken via the injected `speak_fn` callback and
logged at INFO level.  The loop runs on a daemon thread and won't block
shutdown.

Public surface
--------------
    agent = AutonomousAgent(speak_fn=tts.speak)
    agent.start()          – activate background loop
    agent.stop()           – deactivate
    agent.status()         – human-readable status string
    agent.add_reminder(msg, trigger_at)  – schedule a one-shot reminder
    agent.is_running       – bool property
"""

import datetime
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import psutil
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

CPU_ALERT_THRESHOLD: float = 85.0          # % sustained usage
RAM_ALERT_THRESHOLD: float = 90.0          # %
BATTERY_WARN_THRESHOLD: float = 20.0       # % while unplugged
BATTERY_CRITICAL_THRESHOLD: float = 10.0   # %

WORK_SESSION_ALERT_MINUTES: int = 120      # first break prompt after 2 h
BREAK_REMINDER_INTERVAL_MINUTES: int = 30  # re-prompt every 30 min after that

POLL_INTERVAL_SECONDS: int = 30            # how often the loop ticks
CPU_COOLDOWN_SECONDS: int = 300            # min gap between repeated CPU alerts
BATTERY_COOLDOWN_SECONDS: int = 600        # min gap between repeated battery alerts


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class Reminder:
    """A single one-shot reminder the user has scheduled."""
    message: str
    trigger_at: datetime.datetime
    fired: bool = False


@dataclass
class _State:
    """Mutable runtime state owned by AutonomousAgent."""
    session_start: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_break_reminder: Optional[datetime.datetime] = None
    last_cpu_alert: Optional[datetime.datetime] = None
    last_ram_alert: Optional[datetime.datetime] = None
    last_battery_alert: Optional[datetime.datetime] = None
    reminders: list[Reminder] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class AutonomousAgent:
    """
    Background monitoring and proactive suggestion engine.

    Parameters
    ----------
    speak_fn:
        Any callable that accepts a string and reads it aloud.
        Typically ``tts.speak``.
    persona_name:
        Used for log labels; defaults to ``config.ASSISTANT_NAME``.
    """

    def __init__(
        self,
        speak_fn: Callable[[str], None],
        persona_name: str = config.ASSISTANT_NAME,
    ) -> None:
        self._speak = speak_fn
        self._persona = persona_name
        self._state: _State = _State()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Public control methods
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Activate autonomous mode.  Safe to call when already running."""
        if self.is_running:
            return f"Autonomous mode is already active, {config.USER_NAME}."

        self._stop_event.clear()
        self._state = _State()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="AutonomousAgent",
        )
        self._thread.start()
        logger.info("[AutonomousAgent] started")
        return (
            f"Autonomous mode activated, {config.USER_NAME}. "
            "I will monitor your session and alert you proactively."
        )

    def stop(self) -> str:
        """Deactivate autonomous mode."""
        if not self.is_running:
            return f"Autonomous mode is not currently active, {config.USER_NAME}."
        self._stop_event.set()
        logger.info("[AutonomousAgent] stop requested")
        return f"Autonomous mode deactivated, {config.USER_NAME}."

    def status(self) -> str:
        """Return a human-readable status summary."""
        if not self.is_running:
            return "Autonomous mode is offline."

        elapsed = datetime.datetime.now() - self._state.session_start
        total_minutes = int(elapsed.total_seconds() / 60)
        hours, minutes = divmod(total_minutes, 60)
        pending = sum(not r.fired for r in self._state.reminders)

        duration = (
            f"{hours}h {minutes}m" if hours else f"{minutes}m"
        )
        return (
            f"Autonomous mode is active. "
            f"Session running for {duration}. "
            f"{pending} pending reminder(s)."
        )

    def add_reminder(self, message: str, trigger_at: datetime.datetime) -> None:
        """Schedule a one-shot spoken reminder at *trigger_at*."""
        self._state.reminders.append(Reminder(message=message, trigger_at=trigger_at))
        logger.info(
            f"[AutonomousAgent] reminder added: '{message}' "
            f"at {trigger_at.strftime('%H:%M:%S')}"
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        logger.info("[AutonomousAgent] intelligence loop started")
        while not self._stop_event.is_set():
            try:
                self._check_work_session()
                self._check_system_resources()
                self._check_battery()
                self._check_reminders()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"[AutonomousAgent] loop error: {exc}")
            self._stop_event.wait(timeout=POLL_INTERVAL_SECONDS)
        logger.info("[AutonomousAgent] intelligence loop stopped")

    # ------------------------------------------------------------------
    # Monitoring checks
    # ------------------------------------------------------------------

    def _check_work_session(self) -> None:
        """Suggest a break when the session exceeds the configured threshold."""
        now = datetime.datetime.now()
        elapsed_minutes = (now - self._state.session_start).total_seconds() / 60

        if elapsed_minutes < WORK_SESSION_ALERT_MINUTES:
            return

        last = self._state.last_break_reminder
        if last is not None:
            gap_minutes = (now - last).total_seconds() / 60
            if gap_minutes < BREAK_REMINDER_INTERVAL_MINUTES:
                return

        hours = int(elapsed_minutes // 60)
        mins  = int(elapsed_minutes % 60)
        duration = f"{hours} hour{'s' if hours != 1 else ''}"
        if mins:
            duration += f" and {mins} minute{'s' if mins != 1 else ''}"

        self._announce(
            f"{config.USER_NAME}, you have been working for {duration}. "
            "Shall I set a short break reminder, or lock the screen for you?"
        )
        self._state.last_break_reminder = now

    def _check_system_resources(self) -> None:
        """Alert on sustained high CPU or RAM."""
        now = datetime.datetime.now()

        try:
            cpu = psutil.cpu_percent(interval=1)
        except Exception as exc:
            logger.warning(f"[AutonomousAgent] cpu_percent failed: {exc}")
            return

        if cpu >= CPU_ALERT_THRESHOLD:
            last = self._state.last_cpu_alert
            if last is None or (now - last).total_seconds() >= CPU_COOLDOWN_SECONDS:
                self._announce(
                    f"{config.USER_NAME}, CPU usage has spiked to {cpu:.0f}%. "
                    "Would you like me to identify the heavy processes?"
                )
                self._state.last_cpu_alert = now

        try:
            ram = psutil.virtual_memory().percent
        except Exception as exc:
            logger.warning(f"[AutonomousAgent] virtual_memory failed: {exc}")
            return

        if ram >= RAM_ALERT_THRESHOLD:
            last = self._state.last_ram_alert
            if last is None or (now - last).total_seconds() >= CPU_COOLDOWN_SECONDS:
                self._announce(
                    f"{config.USER_NAME}, RAM usage is at {ram:.0f}%. "
                    "I recommend closing unused applications."
                )
                self._state.last_ram_alert = now

    def _check_battery(self) -> None:
        """Warn when battery is low and the device is unplugged."""
        try:
            bat = psutil.sensors_battery()
        except Exception as exc:
            logger.warning(f"[AutonomousAgent] battery check failed: {exc}")
            return

        if bat is None or bat.power_plugged:
            return

        level = bat.percentage
        if level > BATTERY_WARN_THRESHOLD:
            return

        now = datetime.datetime.now()
        last = self._state.last_battery_alert
        if last is not None and (now - last).total_seconds() < BATTERY_COOLDOWN_SECONDS:
            return

        urgency = "critically low" if level <= BATTERY_CRITICAL_THRESHOLD else "low"
        self._announce(
            f"{config.USER_NAME}, battery is {urgency} at {level:.0f}%. "
            "Please plug in your charger."
        )
        self._state.last_battery_alert = now

    def _check_reminders(self) -> None:
        """Fire any user-scheduled reminders that are now due."""
        now = datetime.datetime.now()
        for reminder in self._state.reminders:
            if not reminder.fired and now >= reminder.trigger_at:
                self._announce(f"{config.USER_NAME}: {reminder.message}")
                reminder.fired = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _announce(self, message: str) -> None:
        """Print and speak a proactive message."""
        logger.info(f"[AutonomousAgent] → {message}")
        try:
            self._speak(message)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[AutonomousAgent] speak failed: {exc}")


# ---------------------------------------------------------------------------
# Regex patterns used by main.py to detect autonomous commands
# ---------------------------------------------------------------------------

import re  # noqa: E402 — import after constants to keep thresholds grouped

_CMD_START = re.compile(
    r"\b(go\s+autonomous|autonomous\s+mode\s*(on|active|start)?|enable\s+autonomous|activate\s+autonomous)\b",
    re.I,
)
_CMD_STOP = re.compile(
    r"\b(stop\s+autonomous|autonomous\s+mode\s*(off|stop)?|disable\s+autonomous|disengage\s+autonomous)\b",
    re.I,
)
_CMD_STATUS = re.compile(
    r"\b(autonomous\s+status|status\s+autonomous|are\s+you\s+autonomous)\b",
    re.I,
)


def detect_autonomous_command(text: str) -> Optional[str]:
    """
    Return ``'start'``, ``'stop'``, ``'status'``, or ``None``.
    Called by main.py before normal routing.
    """
    if _CMD_START.search(text):
        return "start"
    if _CMD_STOP.search(text):
        return "stop"
    return "status" if _CMD_STATUS.search(text) else None
