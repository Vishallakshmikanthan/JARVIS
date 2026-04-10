from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import importlib
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import ollama
from loguru import logger

from config import config
from core.planner import PlanStep, SKILL_INTENTS
from core.recovery import recover as _recovery_recover

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 2          # attempts before falling back to LLM
RETRY_DELAY_SECONDS: float = 0.4     # wait between retry attempts

# Intents whose outputs are safe to cache within the session.
# Time-sensitive skills (weather, news) are intentionally excluded.
_CACHEABLE_INTENTS: frozenset[str] = frozenset({
    "joke", "wiki", "search",
})

# Thread pool reused across all async plan executions.
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="jarvis-skill"
)


# ---------------------------------------------------------------------------
# Intent → (module, function) dispatch table
# Single source of truth — mirrors core/router._INTENT_MAP exactly.
# ---------------------------------------------------------------------------

_SKILL_DISPATCH: dict[str, tuple[str, str]] = {
    "weather":    ("skills.weather",           "get_weather"),
    "news":       ("skills.news",              "get_news"),
    "search":     ("skills.search",            "web_search"),
    "wiki":       ("skills.wiki",              "wiki_search"),
    "music":      ("skills.spotify",           "play_music"),
    "system":     ("skills.system",            "system_command"),
    "joke":       ("skills.jokes",             "tell_joke"),
    "persona":    ("core.persona",             "switch_persona"),
    "automation": ("skills.automation",        "schedule_task"),
    "file":       ("skills.file_system",       "file_command"),
    "module":     ("skills.assistant_control", "manage_module"),
    "context":    ("skills.context_awareness", "track_context"),
}


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """
    Outcome of executing a single PlanStep.

    Attributes:
        step:        The PlanStep that was executed.
        output:      The string result returned by the skill or LLM fallback.
        success:     True if the step completed without error.
        attempts:    Total number of execution attempts made (1 = first try succeeded).
        used_fallback: True if the LLM brain was used instead of a skill.
        error:       Last error message if success=False.
    """
    step: PlanStep
    output: str
    success: bool = True
    attempts: int = 1
    used_fallback: bool = False
    error: Optional[str] = None

    def __repr__(self) -> str:
        tag = "LLM" if self.used_fallback else "skill"
        status = "OK" if self.success else "FAIL"
        return (
            f"<StepResult [{status}|{tag}] step={self.step.index} "
            f"attempts={self.attempts} output={self.output[:50]!r}>"
        )


@dataclass
class ExecutionSummary:
    """
    Aggregated outcome of running execute_plan().

    Attributes:
        steps:           All PlanStep objects that were attempted.
        results:         Ordered StepResult for each step.
        combined_response: Final combined string suitable for TTS / GUI display.
        total_steps:     Total number of steps.
        successful:      Steps that completed without error.
        failed:          Steps that ultimately failed after retries.
        fallback_used:   Steps that fell back to LLM reasoning.
    """
    steps: list[PlanStep]
    results: list[StepResult]
    combined_response: str
    total_steps: int = 0
    successful: int = 0
    failed: int = 0
    fallback_used: int = 0

    def __post_init__(self) -> None:
        self.total_steps = len(self.results)
        self.successful = sum(r.success for r in self.results)
        self.failed = sum(not r.success for r in self.results)
        self.fallback_used = sum(r.used_fallback for r in self.results)

    def __repr__(self) -> str:
        return (
            f"<ExecutionSummary total={self.total_steps} "
            f"ok={self.successful} fail={self.failed} "
            f"fallback={self.fallback_used}>"
        )


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def execute_plan(
    steps: list[PlanStep],
    max_retries: int = DEFAULT_MAX_RETRIES,
    context: Optional[str] = None,
) -> str:
    """
    Execute an ordered list of PlanStep objects and return a combined response.

    Single-step plans run synchronously (zero overhead).
    Multi-step plans fan all steps out to the thread pool and run them in
    parallel — the LLM combiner already handles final synthesis, so cross-step
    context dependencies are resolved there. Wall-clock time for multi-step
    plans is therefore bounded by the *slowest* single step rather than the
    sum of all steps.

    For each step the execution order is:
        1. Skill dispatch  — dynamically import the skill module and call its handler.
        2. Retry           — on failure, retry up to *max_retries* times with back-off.
        3. LLM fallback    — if all skill attempts fail, ask the LLM brain directly.
        4. Hard failure    — if LLM also fails, emit a safe user-facing error message.

    Args:
        steps:       Ordered list of PlanStep objects from core.planner.plan_task().
        max_retries: Maximum skill-dispatch retries before triggering LLM fallback.
                     Set to 0 to skip retries and go straight to LLM fallback.
        context:     Optional external context string prepended to LLM fallback prompts.

    Returns:
        A single combined response string. Always non-empty.
    """
    if not steps:
        logger.warning("[executor:execute_plan] received empty step list")
        return "I have no steps to execute. Please try rephrasing your request."

    _t0 = time.perf_counter()
    logger.info(f"[executor:execute_plan] executing {len(steps)} step(s)")

    if len(steps) == 1:
        # Fast path — skip all async overhead for the common single-step case.
        result = _execute_step(
            step=steps[0],
            max_retries=max_retries,
            prior_context=context or "",
        )
        results: list[StepResult] = [result]
    else:
        # Parallel path — fan all steps out concurrently.
        results = _execute_parallel(steps, max_retries, context or "")

    combined = _combine_results(results)
    elapsed = time.perf_counter() - _t0

    summary = ExecutionSummary(steps=steps, results=results, combined_response=combined)
    logger.info(
        f"[executor:execute_plan] complete {summary} "
        f"elapsed={elapsed:.3f}s parallel={'yes' if len(steps) > 1 else 'no'}"
    )

    return combined


def _execute_parallel(
    steps: list[PlanStep],
    max_retries: int,
    base_context: str,
) -> list[StepResult]:
    """
    Run all *steps* concurrently using the module-level ThreadPoolExecutor.
    Returns results ordered by step.index so _combine_results stays deterministic.
    """
    futures: dict[concurrent.futures.Future, int] = {}

    for step in steps:
        future = _THREAD_POOL.submit(
            _execute_step,
            step=step,
            max_retries=max_retries,
            prior_context=base_context,   # each step gets the same baseline context
        )
        futures[future] = step.index

    # Collect and sort by original step order
    indexed_results: list[tuple[int, StepResult]] = []
    for future in concurrent.futures.as_completed(futures):
        idx = futures[future]
        try:
            indexed_results.append((idx, future.result()))
        except Exception as exc:
            # Manufacture a failure result so the plan never silently loses a step
            failed_step = steps[idx - 1]
            indexed_results.append((idx, _hard_failure(failed_step, 1, str(exc))))
            logger.error(f"[executor:_execute_parallel] step={idx} raised: {exc}")

    indexed_results.sort(key=lambda t: t[0])
    results = [r for _, r in indexed_results]
    for r in results:
        logger.info(f"[executor:_execute_parallel] {r}")
    return results


# ---------------------------------------------------------------------------
# Step execution with retry + fallback
# ---------------------------------------------------------------------------

def _execute_step(
    step: PlanStep,
    max_retries: int,
    prior_context: str,
) -> StepResult:
    """
    Attempt to execute one PlanStep.

    Strategy:
        • If intent is in _SKILL_DISPATCH → try skill, retry on failure.
        • If intent is reasoning/general or all skill retries are exhausted → LLM fallback.

    Args:
        step:          The PlanStep to execute.
        max_retries:   Max skill retry attempts before LLM fallback.
        prior_context: Accumulated output from previous steps.

    Returns:
        StepResult with final output, success flag, and attempt count.
    """
    logger.debug(f"[executor:_execute_step] starting {step}")

    # LLM-only intents skip skill dispatch entirely
    if step.intent in ("reasoning", "general"):
        return _llm_fallback(step, prior_context, attempts=1, skill_error=None)

    dispatch = _SKILL_DISPATCH.get(step.intent)
    if not dispatch:
        logger.warning(
            f"[executor:_execute_step] no dispatch entry for intent={step.intent!r} "
            "— routing to LLM fallback"
        )
        return _llm_fallback(step, prior_context, attempts=1, skill_error=None)

    module_path, function_name = dispatch
    last_error: Optional[str] = None

    # Route through the LRU cache for safe-to-cache intents.
    invoke_skill = (
        _cached_skill_call
        if step.intent in _CACHEABLE_INTENTS
        else _call_skill
    )

    for attempt in range(1, max_retries + 2):      # +2: 1 initial + max_retries
        try:
            output = invoke_skill(module_path, function_name, step.query)
            logger.info(
                f"[executor:_execute_step] skill OK intent={step.intent!r} "
                f"attempt={attempt}"
            )
            return StepResult(
                step=step,
                output=output,
                success=True,
                attempts=attempt,
                used_fallback=False,
            )
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                f"[executor:_execute_step] skill attempt={attempt} "
                f"intent={step.intent!r} error={last_error!r}"
            )
            if attempt <= max_retries:
                time.sleep(RETRY_DELAY_SECONDS * attempt)   # progressive back-off

    # All skill attempts exhausted — invoke recovery pipeline
    logger.warning(
        f"[executor:_execute_step] all {max_retries + 1} attempt(s) failed for "
        f"intent={step.intent!r} — invoking recovery pipeline"
    )

    # Build a real Exception from the last error string so recovery can classify it
    skill_exc = RuntimeError(last_error or "unknown skill error")
    recovery = _recovery_recover(
        intent=step.intent,
        query=step.query,
        original_error=skill_exc,
        prior_context=prior_context,
    )

    return StepResult(
        step=step,
        output=recovery.output,
        success=recovery.recovered,
        attempts=max_retries + 2,
        used_fallback=True,
        error=None if recovery.recovered else last_error,
    )


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _cached_skill_call(module_path: str, function_name: str, query: str) -> str:
    """
    LRU-cached wrapper around the raw skill dispatcher.
    Only called for intents listed in *_CACHEABLE_INTENTS*.
    """
    return _call_skill(module_path, function_name, query)


def _call_skill(module_path: str, function_name: str, query: str) -> str:
    """
    Dynamically import *module_path* and invoke *function_name*(query).

    Args:
        module_path:   Importable dotted path (e.g. "skills.weather").
        function_name: Name of the callable within the module.
        query:         The sub-query string to pass as the sole argument.

    Returns:
        String result from the skill.

    Raises:
        Any exception raised by the skill — callers handle retry/fallback logic.
    """
    module = importlib.import_module(module_path)
    fn = getattr(module, function_name)
    result = fn(query)
    return result if isinstance(result, str) else str(result)


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM_PROMPT = (
    "You are JARVIS, a highly capable AI assistant. "
    "Complete the following task step accurately and concisely. "
    "If prior context is provided, use it to inform your answer. "
    "Output only the result — no preamble, no meta-commentary."
)


def _llm_fallback(
    step: PlanStep,
    prior_context: str,
    attempts: int,
    skill_error: Optional[str],
) -> StepResult:
    """
    Fall back to the Ollama LLM brain for a step that a skill could not handle.

    Args:
        step:         The step to reason about.
        prior_context: Accumulated results from prior steps.
        attempts:     Total attempts made so far (for logging).
        skill_error:  Last skill error string, if any (included in LLM prompt for context).

    Returns:
        StepResult with used_fallback=True. success=False only if LLM also returns empty.
    """
    logger.info(f"[executor:_llm_fallback] step={step.index} intent={step.intent!r}")

    prompt_parts = [f"Task: {step.description}", f"Query: {step.query}"]
    if prior_context.strip():
        prompt_parts.insert(0, f"Prior context:\n{prior_context.strip()}\n")
    if skill_error:
        prompt_parts.append(f"(Skill error encountered: {skill_error})")

    prompt = "\n".join(prompt_parts)

    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.5, "num_predict": 512},
        )
        if output := response["message"]["content"].strip():

            return StepResult(
                step=step,
                output=output,
                success=True,
                attempts=attempts,
                used_fallback=True,
            )

        logger.error(f"[executor:_llm_fallback] LLM returned empty for step={step.index}")
        return _hard_failure(step, attempts, "LLM returned empty response")

    except Exception as exc:
        logger.error(f"[executor:_llm_fallback] LLM call failed: {exc}")
        return _hard_failure(step, attempts, str(exc))


def _hard_failure(step: PlanStep, attempts: int, error: str) -> StepResult:
    """Return a user-safe failure result when both skill and LLM have failed."""
    message = (
        f"I was unable to complete step {step.index} "
        f"({step.description}) after {attempts} attempt(s)."
    )
    return StepResult(
        step=step,
        output=message,
        success=False,
        attempts=attempts,
        used_fallback=True,
        error=error,
    )


# ---------------------------------------------------------------------------
# Result combiner
# ---------------------------------------------------------------------------

_COMBINE_SYSTEM_PROMPT = (
    "You are a response composer for an AI assistant. "
    "Given multiple step results, write a single, coherent, natural assistant response. "
    "Be concise. Integrate all results smoothly. No filler or meta-commentary."
)


def _combine_results(results: list[StepResult]) -> str:
    """
    Merge all step outputs into one coherent response string.

    For a single step, the output is returned directly (no LLM needed).
    For multiple steps, the LLM composes a unified response; if that fails,
    outputs are joined with newlines as a safe fallback.

    Args:
        results: Ordered list of StepResult objects.

    Returns:
        A single combined response string.
    """
    if not results:
        return "No results were produced."

    if len(results) == 1:
        return results[0].output

    # Build a numbered summary for the composer
    summary_lines = [
        f"{r.step.index}. {r.step.description}: {r.output}"
        for r in results
    ]
    summary = "\n".join(summary_lines)

    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _COMBINE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Step results:\n{summary}"},
            ],
            options={"temperature": 0.6, "num_predict": 512},
        )
        if composed := response["message"]["content"].strip():
            logger.info("[executor:_combine_results] LLM composition succeeded")
            return composed
    except Exception as exc:
        logger.warning(f"[executor:_combine_results] LLM composition failed: {exc}")

    # Safe fallback: join outputs separated by newlines
    logger.info("[executor:_combine_results] using plain join fallback")
    return "\n\n".join(r.output for r in results if r.output)
