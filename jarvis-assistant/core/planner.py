from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

import ollama
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Executable intent registry
# Mirrors _INTENT_MAP in core/router.py — this is the ground truth of what
# skills exist. Only steps with an intent in this set are considered executable.
# ---------------------------------------------------------------------------

EXECUTABLE_INTENTS: frozenset[str] = frozenset({
    "weather",
    "news",
    "search",
    "wiki",
    "music",
    "system",
    "joke",
    "persona",
    "automation",
    "file",
    "module",
    "context",
    "reasoning",   # handled by LLM brain — always executable
    "general",     # catch-all brain fallthrough — always executable
})

# Intents that map to a real skill module (non-LLM dispatch)
SKILL_INTENTS: frozenset[str] = frozenset({
    "weather", "news", "search", "wiki", "music",
    "system", "joke", "automation", "file", "module", "context",
})

# Maximum number of steps the planner may produce
MAX_STEPS = 8


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """
    A single validated, executable step in a task plan.

    Attributes:
        index:       1-based position in the plan.
        description: Human-readable summary of what this step does.
        intent:      Intent category that maps to a skill or LLM handler.
        query:       Self-contained sub-query passed directly to the skill/executor.
        executable:  True when intent is in EXECUTABLE_INTENTS.
        fallback:    If not executable, suggested closest valid intent.
    """
    index: int
    description: str
    intent: str
    query: str
    executable: bool = True
    fallback: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        status = "OK" if self.executable else f"INVALID→{self.fallback}"
        return f"<PlanStep {self.index} [{self.intent}|{status}] {self.description!r}>"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_FALLBACK_MAP: dict[str, str] = {
    # Common LLM hallucinations → nearest real intent
    "cab":          "automation",
    "ride":         "automation",
    "transport":    "automation",
    "booking":      "automation",
    "maps":         "search",
    "navigate":     "search",
    "directions":   "search",
    "open":         "system",
    "launch":       "system",
    "app":          "system",
    "call":         "system",
    "timer":        "automation",
    "alarm":        "automation",
    "reminder":     "automation",
    "email":        "automation",
    "calendar":     "automation",
    "translate":    "search",
    "calculate":    "reasoning",
    "math":         "reasoning",
    "summarise":    "reasoning",
    "summarize":    "reasoning",
    "explain":      "reasoning",
    "compare":      "reasoning",
    "write":        "reasoning",
    "compose":      "reasoning",
}


def _resolve_intent(raw_intent: str) -> tuple[str, Optional[str]]:
    """
    Normalize and validate a raw intent string from the LLM.

    Returns:
        (resolved_intent, fallback_suggestion)
        - resolved_intent: a valid EXECUTABLE_INTENTS member.
        - fallback_suggestion: the original raw value if substitution occurred, else None.
    """
    normalized = raw_intent.strip().lower()

    if normalized in EXECUTABLE_INTENTS:
        return normalized, None

    # Try fallback map
    if normalized in _FALLBACK_MAP:
        fallback = _FALLBACK_MAP[normalized]
        logger.warning(
            f"[planner:_resolve_intent] unknown intent={normalized!r} "
            f"→ fallback={fallback!r}"
        )
        return fallback, normalized

    # Partial match against known intents (e.g. "weather_check" → "weather")
    for known in EXECUTABLE_INTENTS:
        if known in normalized or normalized in known:
            logger.warning(
                f"[planner:_resolve_intent] partial match {normalized!r} → {known!r}"
            )
            return known, normalized

    # No match — default to reasoning (LLM can handle it)
    logger.warning(
        f"[planner:_resolve_intent] unresolvable intent={normalized!r} → 'reasoning'"
    )
    return "reasoning", normalized


def _validate_steps(raw_steps: list[dict]) -> list[PlanStep]:
    """
    Validate and normalise a list of raw step dicts from the LLM.

    Validation rules:
    - 'index' must be a positive integer.
    - 'description' must be a non-empty string.
    - 'query' must be a non-empty string.
    - 'intent' is resolved via _resolve_intent(); unknown intents fall back gracefully.
    - Steps beyond MAX_STEPS are dropped with a warning.
    - Duplicate indices are re-numbered sequentially.

    Returns:
        List of valid PlanStep objects, re-indexed from 1.
    """
    steps: list[PlanStep] = []

    for i, item in enumerate(raw_steps):
        if len(steps) >= MAX_STEPS:
            logger.warning(
                f"[planner:_validate_steps] MAX_STEPS={MAX_STEPS} reached — "
                f"dropping remaining {len(raw_steps) - i} step(s)"
            )
            break

        # Required fields
        raw_description = str(item.get("description", "")).strip()
        raw_query = str(item.get("query", "")).strip()
        raw_intent = str(item.get("intent", "reasoning")).strip()

        if not raw_description:
            logger.warning(f"[planner:_validate_steps] step {i+1} missing description — skipping")
            continue
        if not raw_query:
            logger.warning(f"[planner:_validate_steps] step {i+1} missing query — using description")
            raw_query = raw_description

        resolved_intent, original_intent = _resolve_intent(raw_intent)
        executable = resolved_intent in EXECUTABLE_INTENTS

        steps.append(PlanStep(
            index=len(steps) + 1,   # re-index sequentially
            description=raw_description,
            intent=resolved_intent,
            query=raw_query,
            executable=executable,
            fallback=original_intent,
        ))

    return steps


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = f"""You are a task planning engine for an AI assistant.

Your role: decompose the user's request into an ordered, minimal list of discrete executable steps.

Output ONLY valid JSON — no prose, no markdown fences.

Each step object must have exactly these keys:
  "index"       : integer, 1-based
  "description" : short human-readable summary of this step (max 12 words)
  "intent"      : one of these exact values: {", ".join(sorted(EXECUTABLE_INTENTS))}
  "query"       : the precise, self-contained sub-query for this step

Rules:
- Use "reasoning" for any LLM synthesis (summarise, explain, calculate, compose).
- Use "search" for web lookups. Use "wiki" for encyclopaedic lookups. 
- Use "weather" only for weather/forecast queries.
- Use "automation" for scheduling, reminders, ride booking, alarms.
- Use "system" for device control, app launching, volume, brightness.
- Each "query" must be fully self-contained — no pronouns referencing other steps.
- Minimum 1 step, maximum {MAX_STEPS} steps.
- Aim for the fewest steps that correctly cover the request.
- Never invent intents outside the allowed list.

Output format (strict):
[
  {{"index": 1, "description": "...", "intent": "...", "query": "..."}},
  {{"index": 2, "description": "...", "intent": "...", "query": "..."}}
]"""


# ---------------------------------------------------------------------------
# Core planner function
# ---------------------------------------------------------------------------

def plan_task(user_input: str, model: str = config.OLLAMA_MODEL) -> list[PlanStep]:
    """
    Convert a natural-language user request into an ordered list of validated PlanSteps.

    Uses the LLM to decompose the request, then validates every step against the
    EXECUTABLE_INTENTS registry. Unknown intents are resolved via fallback mapping
    or normalised to 'reasoning'. Steps are re-indexed sequentially.

    Args:
        user_input: Raw user request (text or transcribed speech).
        model:      Ollama model name. Defaults to config.OLLAMA_MODEL.

    Returns:
        List of PlanStep objects, each guaranteed to have an executable intent.
        Returns a single fallback step on total failure (e.g. LLM unreachable).

    Raises:
        Does not raise — all errors are caught and logged; a fallback step is returned.

    Example:
        >>> steps = plan_task("Book a cab and check the weather")
        >>> for s in steps:
        ...     print(s)
        <PlanStep 1 [weather|OK] 'Check the current weather'>
        <PlanStep 2 [automation|OK] 'Book a cab ride'>
    """
    logger.info(f"[planner:plan_task] input={user_input!r}")

    raw_json = _call_llm(user_input, model)
    raw_steps = _parse_json(raw_json)

    if raw_steps is None:
        logger.warning("[planner:plan_task] LLM JSON parse failed — using single fallback step")
        return _fallback_step(user_input)

    if not raw_steps:
        logger.warning("[planner:plan_task] LLM returned empty step list — using single fallback step")
        return _fallback_step(user_input)

    steps = _validate_steps(raw_steps)

    if not steps:
        logger.warning("[planner:plan_task] all steps failed validation — using single fallback step")
        return _fallback_step(user_input)

    logger.info(
        f"[planner:plan_task] produced {len(steps)} validated step(s): "
        f"{[s.intent for s in steps]}"
    )
    return steps


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def steps_to_json(steps: list[PlanStep]) -> str:
    """
    Serialise a list of PlanStep objects to a formatted JSON string.

    Args:
        steps: List of PlanStep objects.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps([s.to_dict() for s in steps], indent=2)


def steps_from_json(raw: str) -> list[PlanStep]:
    """
    Deserialise a JSON string back into a validated list of PlanStep objects.

    Args:
        raw: JSON string previously produced by steps_to_json or an external source.

    Returns:
        List of validated PlanStep objects. Returns single fallback step on parse errors.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")
        return _validate_steps(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(f"[planner:steps_from_json] failed: {exc}")
        return _fallback_step(raw[:80])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _call_llm(user_input: str, model: str) -> str:
    """Send the planning prompt to Ollama and return the raw response string."""
    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_input},
            ],
            options={"temperature": 0.2, "num_predict": 768},
        )
        reply: str = response["message"]["content"].strip()
        logger.debug(f"[planner:_call_llm] raw_reply={reply[:200]!r}")
        return reply
    except Exception as exc:
        logger.error(f"[planner:_call_llm] LLM call failed: {exc}")
        return ""


def _parse_json(raw: str) -> Optional[list[dict]]:
    """
    Extract and parse a JSON array from the LLM's raw output.

    Returns parsed list of dicts, or None on failure.
    """
    if not raw:
        return None

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw).strip()

    # Locate the first JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        logger.warning(f"[planner:_parse_json] no JSON array found in output: {raw[:120]!r}")
        return None

    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            logger.warning("[planner:_parse_json] parsed JSON is not a list")
            return None
        return data
    except json.JSONDecodeError as exc:
        logger.error(f"[planner:_parse_json] JSON decode error: {exc} | raw={raw[:120]!r}")
        return None


def _fallback_step(user_input: str) -> list[PlanStep]:
    """Return a single 'reasoning' step to ensure the pipeline always has something to run."""
    return [PlanStep(
        index=1,
        description="Process the full request using reasoning",
        intent="reasoning",
        query=user_input,
        executable=True,
        fallback=None,
    )]
