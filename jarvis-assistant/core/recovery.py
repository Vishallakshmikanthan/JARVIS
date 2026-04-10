from __future__ import annotations

"""
core/recovery.py
Fault-tolerance layer for failed skill executions.

Pipeline for a failed skill call:
    1. Alternate skill  — swap to a semantically equivalent skill
                          (e.g. Spotify → YouTube search, weather API → wiki)
    2. Decompose query  — strip the failed skill's trigger words and retry with
                          a simpler general-knowledge lookup via wiki/search
    3. LLM explanation  — ask the LLM to explain the failure and propose what
                          the user can do instead
    4. Hard failure     — return a polite error message (never raises)

Public API
----------
    recover(intent, query, original_error, prior_context) -> RecoveryResult
    classify_error(exc) -> ErrorCategory
"""

import importlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import ollama
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    """Broad category of what went wrong."""
    IMPORT_ERROR     = "import_error"      # module/function not found
    AUTH_ERROR       = "auth_error"        # missing API key / token expired
    NETWORK_ERROR    = "network_error"     # requests/timeout/DNS failure
    NOT_FOUND        = "not_found"         # 404, empty result, item not found
    RATE_LIMIT       = "rate_limit"        # 429 / quota exceeded
    PERMISSION_ERROR = "permission_error"  # OS permission denied
    PARSE_ERROR      = "parse_error"       # unexpected response format
    UNKNOWN          = "unknown"


# Regex → ErrorCategory mapping (evaluated in order; first match wins)
_ERROR_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"No module named|ImportError|ModuleNotFoundError|has no attribute", re.I), ErrorCategory.IMPORT_ERROR),
    (re.compile(r"api[_\s]?key|token|auth|credential|unauthorized|403|401",           re.I), ErrorCategory.AUTH_ERROR),
    (re.compile(r"timeout|connection|network|DNS|refused|unreachable|socket",          re.I), ErrorCategory.NETWORK_ERROR),
    (re.compile(r"404|not found|no results?|empty|nothing found",                      re.I), ErrorCategory.NOT_FOUND),
    (re.compile(r"429|rate.?limit|quota|too many requests",                            re.I), ErrorCategory.RATE_LIMIT),
    (re.compile(r"permission denied|access denied|PermissionError",                    re.I), ErrorCategory.PERMISSION_ERROR),
    (re.compile(r"json|parse|decode|unexpected|format",                                re.I), ErrorCategory.PARSE_ERROR),
]


def classify_error(exc: Exception) -> ErrorCategory:
    """Map an exception to the nearest ErrorCategory."""
    msg = f"{type(exc).__name__}: {exc}"
    return next(
        (category for pattern, category in _ERROR_PATTERNS if pattern.search(msg)),
        ErrorCategory.UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Alternate skill table
# ---------------------------------------------------------------------------

# Maps a failed intent to one or more (module, function, label) tuples tried
# in order.  Each entry is the "plan B" for that skill.
_ALTERNATES: dict[str, list[tuple[str, str, str]]] = {
    # Music: Spotify failed → suggest a YouTube search instead
    "music": [
        ("skills.search",  "web_search",   "YouTube"),
        ("skills.wiki",    "wiki_search",  "Artist info"),
    ],
    # Weather: API failed → fall back to a web search for the forecast
    "weather": [
        ("skills.search",  "web_search",   "web weather search"),
        ("skills.wiki",    "wiki_search",  "climate info"),
    ],
    # News: API key / rate limit → try a web search
    "news": [
        ("skills.search",  "web_search",   "news search"),
    ],
    # Wiki: article not found → try a broader web search
    "wiki": [
        ("skills.search",  "web_search",   "general search"),
    ],
    # Search: primary engine failed → try wiki
    "search": [
        ("skills.wiki",    "wiki_search",  "Wikipedia"),
    ],
    # System commands: psutil call failed → report via LLM
    "system": [],    # no alternate skill; goes straight to LLM
    # Automations: scheduling failed → remind via LLM text
    "automation": [
        ("skills.timer",   "set_timer",    "timer"),
    ],
    # File operations: permission / not-found → LLM guidance
    "file": [],
    # Jokes: if the joke module fails that's pretty bad — use LLM
    "joke": [],
}

# Human-friendly suggestion injected into the LLM explanation prompt
_SUGGESTION_TEXT: dict[str, str] = {
    "music":      "You could try asking me to search for the song on YouTube instead.",
    "weather":    "You could try asking me to search the web for today's forecast.",
    "news":       "You could try asking me to perform a live web search for the latest news.",
    "wiki":       "You could try rephrasing your question or asking me to search the web.",
    "search":     "You could try asking me to look it up on Wikipedia instead.",
    "system":     "You could check Task Manager or System Settings manually.",
    "automation": "You could ask me to set a timer as a reminder instead.",
    "file":       "Please check file permissions or provide the full path and try again.",
    "joke":       "I am temporarily unable to fetch a joke — I can attempt one from memory.",
}


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    """
    Outcome of a recovery attempt.

    Attributes:
        output          Final string to return to the user. Always non-empty.
        recovered       True if an alternate skill or LLM produced a useful result.
        strategy        Human-readable description of what recovery path was used.
        intent          Original failing intent.
        error_category  Classified error type.
        alternate_used  Module path of the alternate skill, if one succeeded.
    """
    output: str
    recovered: bool
    strategy: str
    intent: str
    error_category: ErrorCategory
    alternate_used: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM explanation system prompt
# ---------------------------------------------------------------------------

_RECOVERY_SYSTEM_PROMPT = (
    f"You are {config.ASSISTANT_NAME}, a highly capable AI assistant. "
    "A requested skill has just encountered an error. Your job is to:\n"
    "1. Briefly acknowledge what went wrong (one sentence, no technical jargon).\n"
    "2. Suggest a practical alternative the user can try right now.\n"
    "3. If possible, answer the question from your own knowledge.\n"
    "Keep the total response under 3 sentences. Be helpful, not apologetic."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_alternate(module_path: str, fn_name: str, query: str) -> str:
    """Dynamically call an alternate skill. Raises on failure."""
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, fn_name)
    result = fn(query)
    return result if isinstance(result, str) else str(result)


def _build_alternate_query(intent: str, original_query: str, label: str) -> str:
    """
    Transform the original query for use with an alternate skill.
    E.g. for music → YouTube: prepend "youtube " so the search is targeted.
    """
    transforms: dict[str, str] = {
        "YouTube":          f"youtube {original_query}",
        "web weather search": f"weather forecast {original_query}",
        "news search":      f"latest news {original_query}",
        "general search":   original_query,
    }
    return transforms.get(label, original_query)


def _llm_explain(
    intent: str,
    query: str,
    error_category: ErrorCategory,
    original_error: str,
    prior_context: str,
) -> str:
    """Ask the LLM to acknowledge the failure and suggest alternatives."""
    suggestion = _SUGGESTION_TEXT.get(intent, "You could try rephrasing your request.")
    messages = [
        {"role": "system", "content": _RECOVERY_SYSTEM_PROMPT},
    ]
    if prior_context:
        messages.append({"role": "assistant", "content": prior_context})

    user_content = (
        f"The user asked: \"{query}\"\n"
        f"The skill for intent '{intent}' failed with a {error_category.value} error.\n"
        f"Error detail: {original_error[:200]}\n"
        f"Suggestion hint: {suggestion}\n"
        "Please respond helpfully."
    )
    messages.append({"role": "user", "content": user_content})

    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.4, "num_predict": 150},
        )
        output = response["message"]["content"].strip()
        return output or _hard_failure_message(intent, error_category)
    except Exception as exc:
        logger.error(f"[recovery:_llm_explain] LLM call failed: {exc}")
        return _hard_failure_message(intent, error_category)


def _hard_failure_message(intent: str, category: ErrorCategory) -> str:
    """Last-resort user-facing message when everything fails."""
    suggestion = _SUGGESTION_TEXT.get(intent, "Please try rephrasing your request.")
    return (
        f"I was unable to complete your {intent} request "
        f"({category.value.replace('_', ' ')}). {suggestion}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recover(
    intent: str,
    query: str,
    original_error: Exception,
    prior_context: str = "",
) -> RecoveryResult:
    """
    Attempt to recover from a failed skill execution.

    Recovery pipeline (stops at first success):
        1. Alternate skill(s) listed in _ALTERNATES[intent]
        2. LLM explanation + suggestion
        3. Hard failure message

    Parameters
    ----------
    intent:
        The intent key of the failing skill (e.g. ``"music"``).
    query:
        The parsed query that was passed to the failing skill.
    original_error:
        The exception raised by the failing skill.
    prior_context:
        Accumulated output from previous steps in the execution plan
        (used to give the LLM useful context).

    Returns
    -------
    RecoveryResult — never raises.

    Example
    -------
    ::

        from core.recovery import recover

        try:
            output = play_music("Bohemian Rhapsody")
        except Exception as exc:
            result = recover("music", "Bohemian Rhapsody", exc)
            print(result.output)
            # "I couldn't connect to Spotify. You could search YouTube for
            #  Bohemian Rhapsody instead."
    """
    error_category = classify_error(original_error)
    original_error_str = str(original_error)

    logger.warning(
        f"[recovery:recover] intent={intent!r} "
        f"category={error_category.value} "
        f"error={original_error_str[:120]!r}"
    )

    # ------------------------------------------------------------------ #
    # Stage 1 — Try alternate skills                                       #
    # ------------------------------------------------------------------ #
    alternates = _ALTERNATES.get(intent, [])
    for module_path, fn_name, label in alternates:
        alt_query = _build_alternate_query(intent, query, label)
        try:
            output = _call_alternate(module_path, fn_name, alt_query)
            logger.info(
                f"[recovery:recover] alternate succeeded "
                f"intent={intent!r} alternate={module_path}.{fn_name} label={label!r}"
            )
            return RecoveryResult(
                output=f"[{label} result]\n{output}",
                recovered=True,
                strategy=f"alternate skill: {module_path}.{fn_name} ({label})",
                intent=intent,
                error_category=error_category,
                alternate_used=module_path,
            )
        except Exception as alt_exc:
            logger.debug(
                f"[recovery:recover] alternate {module_path}.{fn_name} failed: {alt_exc}"
            )

    # ------------------------------------------------------------------ #
    # Stage 2 — LLM explanation + suggestion                              #
    # ------------------------------------------------------------------ #
    logger.info(f"[recovery:recover] escalating to LLM explanation intent={intent!r}")
    llm_output = _llm_explain(
        intent=intent,
        query=query,
        error_category=error_category,
        original_error=original_error_str,
        prior_context=prior_context,
    )

    # Determine if LLM gave a substantive answer (more than just apology text)
    llm_recovered = len(llm_output) > 40 and "unable" not in llm_output.lower()

    return RecoveryResult(
        output=llm_output,
        recovered=llm_recovered,
        strategy="llm_explanation",
        intent=intent,
        error_category=error_category,
    )
