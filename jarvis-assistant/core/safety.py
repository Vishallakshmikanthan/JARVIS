"""
core/safety.py — JARVIS Safety Module

Two-layer protection for every user command:

Layer 1 — Rule-based (instant, no LLM needed)
    • BLOCKED commands: patterns that are always denied (e.g. rm -rf /, fork bombs).
    • CRITICAL commands: patterns that require explicit user confirmation before execution.

Layer 2 — LLM validation (secondary check for ambiguous inputs)
    • If Layer 1 passes, the LLM rates how dangerous the command is on a 0–10 scale.
    • Commands scoring >= LLM_DANGER_THRESHOLD are treated as CRITICAL (need confirmation).

Public API
----------
check(text)                    → SafetyResult  (call before routing every command)
confirm(text)                  → bool          (call when user replies to a confirmation prompt)
SafetyResult.is_safe           → True if the command can proceed immediately
SafetyResult.needs_confirmation→ True if JARVIS should ask "Are you sure?"
SafetyResult.is_blocked        → True if the command is always denied
SafetyResult.prompt            → The question/message to surface to the user
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import ollama
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Safety verdict
# ---------------------------------------------------------------------------

class SafetyVerdict(str, Enum):
    SAFE         = "safe"          # proceed immediately
    NEEDS_CONFIRM= "needs_confirm" # ask "Are you sure?"
    BLOCKED      = "blocked"       # always deny


@dataclass
class SafetyResult:
    verdict: SafetyVerdict
    reason:  str
    prompt:  str = ""              # message to show the user

    @property
    def is_safe(self) -> bool:
        return self.verdict == SafetyVerdict.SAFE

    @property
    def needs_confirmation(self) -> bool:
        return self.verdict == SafetyVerdict.NEEDS_CONFIRM

    @property
    def is_blocked(self) -> bool:
        return self.verdict == SafetyVerdict.BLOCKED

    def __repr__(self) -> str:
        return f"<SafetyResult verdict={self.verdict.value} reason={self.reason!r}>"


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

# These patterns are ALWAYS blocked — no override possible.
_BLOCKED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("fork bomb",         re.compile(r":\(\)\s*\{", re.I)),
    ("rm -rf /",          re.compile(r"\brm\s+-[rRfF]{1,3}\s+/?(\s|$)", re.I)),
    ("format disk",       re.compile(r"\b(format\s+[a-z]:|\bmkfs\b)", re.I)),
    ("rm system32",       re.compile(r"rm\s+.*system32", re.I)),
    ("drop database",     re.compile(r"\bDROP\s+(DATABASE|TABLE|SCHEMA)\b", re.I)),
    ("overwrite mbr",     re.compile(r"\bdd\s+.*of=/dev/(sda|hda|nvme)", re.I)),
    ("credential dump",   re.compile(r"\b(mimikatz|lsass\s*dump|pass-the-hash)\b", re.I)),
    ("exploit payload",   re.compile(r"\b(msfconsole|msfvenom|metasploit|reverse_shell)\b", re.I)),
    ("sudo rm",           re.compile(r"\bsudo\s+rm\s+-[rRfF]", re.I)),
    ("deltree",           re.compile(r"\bdeltree\s+/[yY]?\s*(c:|d:)?\\", re.I)),
]

# These patterns require confirmation before executing.
_CRITICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("shutdown",          re.compile(r"\b(shutdown|power\s*off|turn\s*off\s*(the\s+)?computer)\b", re.I)),
    ("restart",           re.compile(r"\b(restart|reboot)\b", re.I)),
    ("delete file",       re.compile(r"\b(delete|remove|erase)\s+(the\s+)?(file|folder|directory)\b", re.I)),
    ("wipe",              re.compile(r"\b(wipe|clear|purge)\s+(the\s+)?(disk|drive|database|memory|data)\b", re.I)),
    ("send message",      re.compile(r"\b(send|message)\s+.{1,60}(whatsapp|telegram|email|sms)\b", re.I)),
    ("kill process",      re.compile(r"\b(kill|terminate)\s+(process|service|task)\b", re.I)),
    ("uninstall",         re.compile(r"\buninstall\b", re.I)),
    ("format drive",      re.compile(r"\bformat\s+(drive|partition)\b", re.I)),
    ("overwrite file",    re.compile(r"\b(overwrite|replace)\s+(file|data)\b", re.I)),
    ("disable firewall",  re.compile(r"\b(disable|turn\s*off)\s+(the\s+)?(firewall|antivirus|defender)\b", re.I)),
    ("rm command",        re.compile(r"\brm\s+-[rRfF]\b", re.I)),
    ("drop collection",   re.compile(r"\b(drop|truncate)\s+(collection|table|index)\b", re.I)),
]

# Confirmation phrases the user can say to approve a critical action.
_CONFIRM_PHRASES: re.Pattern = re.compile(
    r"\b(yes|yeah|yep|confirm|sure|proceed|affirmative|do it|go ahead|approve)\b",
    re.I,
)
_DENY_PHRASES: re.Pattern = re.compile(
    r"\b(no|nope|cancel|stop|abort|deny|negative|don't|do not|never mind)\b",
    re.I,
)

# LLM danger score threshold (0–10). Commands rated >= this need confirmation.
LLM_DANGER_THRESHOLD = 7

# ---------------------------------------------------------------------------
# Rule-based checks
# ---------------------------------------------------------------------------

def _rule_check(text: str) -> Optional[SafetyResult]:
    """
    Fast pattern-matching check. Returns a SafetyResult if a rule fires,
    or None if nothing matched (allows LLM validation to proceed).
    """
    # 1. Blocked — always deny
    for label, pattern in _BLOCKED_PATTERNS:
        if pattern.search(text):
            logger.warning(f"[safety] BLOCKED rule='{label}' text='{text[:80]}'")
            return SafetyResult(
                verdict=SafetyVerdict.BLOCKED,
                reason=f"rule:blocked:{label}",
                prompt=(
                    f"I can't do that, Sir. The command matches a blocked pattern "
                    f'("{label}") and will not be executed.'
                ),
            )

    # 2. Critical — require confirmation
    for label, pattern in _CRITICAL_PATTERNS:
        if pattern.search(text):
            logger.warning(f"[safety] CRITICAL rule='{label}' text='{text[:80]}'")
            return SafetyResult(
                verdict=SafetyVerdict.NEEDS_CONFIRM,
                reason=f"rule:critical:{label}",
                prompt=_confirmation_question(label, text),
            )

    return None


def _confirmation_question(action_label: str, original_text: str) -> str:
    """Generate a contextual confirmation question from the action label."""
    questions = {
        "shutdown":        "Are you sure you want to shut down the system?",
        "restart":         "Are you sure you want to restart the system?",
        "delete file":     "Are you sure you want to delete this file? This cannot be undone.",
        "wipe":            "Are you sure you want to wipe this data? This is irreversible.",
        "send message":    "Are you sure you want to send this message?",
        "kill process":    "Are you sure you want to terminate this process?",
        "uninstall":       "Are you sure you want to uninstall this application?",
        "format drive":    "Are you sure you want to format this drive? All data will be lost.",
        "overwrite file":  "Are you sure you want to overwrite this file?",
        "disable firewall":"Are you sure you want to disable the firewall? This reduces system security.",
        "rm command":      "Are you sure you want to run this delete command?",
        "drop collection": "Are you sure you want to drop this database collection? Data will be lost.",
    }
    return questions.get(action_label, f"Are you sure you want to proceed with '{action_label}'?")


# ---------------------------------------------------------------------------
# LLM validation
# ---------------------------------------------------------------------------

_LLM_SAFETY_PROMPT = (
    "You are a strict AI safety classifier. Rate how dangerous the following "
    "user command is on a scale from 0 (completely safe) to 10 (extremely dangerous / "
    "destructive / irreversible). Consider: file deletion, system shutdown/restart, "
    "data loss, unauthorized access, harmful payloads. "
    "Reply with ONLY a single integer between 0 and 10. No explanation.\n\n"
    "Command: {command}"
)


def _llm_danger_score(text: str) -> int:
    """
    Ask Ollama for a danger score (0–10).
    Returns 0 on failure so a broken LLM never blocks execution.
    """
    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict safety classifier. Reply with a single integer 0-10 only.",
                },
                {
                    "role": "user",
                    "content": _LLM_SAFETY_PROMPT.format(command=text[:300]),
                },
            ],
            options={"temperature": 0.0, "num_predict": 5},
        )
        raw = response["message"]["content"].strip()
        # Extract first integer from response
        match = re.search(r"\d+", raw)
        score = int(match.group()) if match else 0
        score = max(0, min(score, 10))   # clamp to [0, 10]
        logger.debug(f"[safety] LLM danger score={score} for '{text[:60]}'")
        return score
    except Exception as e:
        logger.warning(f"[safety] LLM danger check failed: {e} — defaulting to 0")
        return 0


# ---------------------------------------------------------------------------
# Pending confirmation state (simple in-process store)
# ---------------------------------------------------------------------------

class _ConfirmationState:
    """Tracks whether JARVIS is currently awaiting a yes/no confirmation."""

    def __init__(self) -> None:
        self._pending: Optional[str] = None     # original command awaiting confirm
        self._pending_reason: Optional[str] = None

    def set(self, command: str, reason: str) -> None:
        self._pending = command
        self._pending_reason = reason
        logger.debug(f"[safety] confirmation pending for: '{command[:60]}'")

    def clear(self) -> None:
        self._pending = None
        self._pending_reason = None

    @property
    def is_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_command(self) -> Optional[str]:
        return self._pending

    @property
    def pending_reason(self) -> Optional[str]:
        return self._pending_reason


_confirm_state = _ConfirmationState()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(text: str, use_llm: bool = True) -> SafetyResult:
    """
    Run a full two-layer safety check on *text*.

    Layer 1: Rule-based pattern matching (instant).
    Layer 2: LLM danger scoring (only when Layer 1 passes and use_llm=True).

    Returns a SafetyResult describing the verdict and the message to show
    the user. The caller should NOT execute the command if is_blocked or
    needs_confirmation is True.
    """
    # Layer 1 — rules
    rule_result = _rule_check(text)
    if rule_result is not None:
        if rule_result.needs_confirmation:
            _confirm_state.set(text, rule_result.reason)
        return rule_result

    # Layer 2 — LLM danger scoring
    if use_llm:
        score = _llm_danger_score(text)
        if score >= LLM_DANGER_THRESHOLD:
            result = SafetyResult(
                verdict=SafetyVerdict.NEEDS_CONFIRM,
                reason=f"llm:danger_score:{score}",
                prompt=(
                    f"I've assessed this command as potentially high-risk "
                    f"(danger level {score}/10). Are you sure you want to proceed?"
                ),
            )
            _confirm_state.set(text, result.reason)
            logger.warning(
                f"[safety] LLM elevated to CRITICAL score={score} text='{text[:60]}'"
            )
            return result

    return SafetyResult(
        verdict=SafetyVerdict.SAFE,
        reason="passed",
        prompt="",
    )


def handle_confirmation_reply(reply: str) -> tuple[bool, Optional[str]]:
    """
    Process a yes/no reply to a pending confirmation request.

    Returns:
        (approved: bool, original_command: Optional[str])
        - approved=True  → caller should now execute the saved command
        - approved=False → caller should inform the user the action was cancelled
        - original_command is None when no confirmation was pending
    """
    if not _confirm_state.is_pending:
        return False, None

    original = _confirm_state.pending_command
    _confirm_state.clear()

    if _CONFIRM_PHRASES.search(reply):
        logger.info(f"[safety] confirmed → '{original[:60]}'")
        return True, original

    if _DENY_PHRASES.search(reply):
        logger.info(f"[safety] denied → '{original[:60]}'")
        return False, original

    # Ambiguous reply — treat as denial for safety
    logger.info(f"[safety] ambiguous reply '{reply[:40]}' → treating as denial")
    return False, original


def is_awaiting_confirmation() -> bool:
    """True when JARVIS is currently waiting for the user to confirm/deny an action."""
    return _confirm_state.is_pending


def pending_confirmation_command() -> Optional[str]:
    """Return the command currently awaiting confirmation, or None."""
    return _confirm_state.pending_command
