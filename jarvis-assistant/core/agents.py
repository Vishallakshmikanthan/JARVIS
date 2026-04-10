from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import ollama
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single discrete task produced by the Planner."""
    index: int
    description: str
    intent: str           # maps to router intent categories (or "reasoning")
    query: str            # cleaned sub-query to pass to executor

    def __repr__(self) -> str:
        return f"<Step {self.index}: [{self.intent}] {self.description!r}>"


@dataclass
class ExecutionResult:
    """Output for a single executed step."""
    step: Step
    output: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentResponse:
    """Final packaged response from the multi-agent pipeline."""
    original_input: str
    steps: list[Step]
    results: list[ExecutionResult]
    final_response: str
    critic_passed: bool
    critic_notes: str = ""

    def __repr__(self) -> str:
        return (
            f"<AgentResponse steps={len(self.steps)} "
            f"critic_passed={self.critic_passed} "
            f"response={self.final_response[:60]!r}>"
        )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Shared foundation for all agents.
    Each agent gets its own Ollama client, isolated system prompt,
    and temperature tuning — no shared mutable state.
    """

    _AGENT_NAME: str = "BaseAgent"

    def __init__(
        self,
        system_prompt: str,
        model: str = config.OLLAMA_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = ollama.Client(host=config.OLLAMA_BASE_URL)
        logger.debug(
            f"[agents:{self._AGENT_NAME}] initialised model={self.model} "
            f"temp={self.temperature}"
        )

    def _call(self, user_message: str, extra_context: str = "") -> str:
        """
        Send a single-turn message to the LLM and return the raw reply.

        Args:
            user_message: The prompt to send.
            extra_context: Optional additional context prepended as a system note.

        Returns:
            Raw string response from the model.
        """
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

        if extra_context:
            messages.append({
                "role": "system",
                "content": f"[Additional context]\n{extra_context}",
            })

        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            reply: str = response["message"]["content"].strip()
            logger.debug(f"[agents:{self._AGENT_NAME}] reply_length={len(reply)}")
            return reply
        except Exception as exc:
            logger.error(f"[agents:{self._AGENT_NAME}] LLM call failed: {exc}")
            return ""


# ---------------------------------------------------------------------------
# Planner Agent  (JARVIS persona)
# Delegates to core.planner.plan_task for validated, structured step generation.
# ---------------------------------------------------------------------------

class PlannerAgent(BaseAgent):
    """
    Breaks the user request into an ordered list of Steps using core.planner.
    Persona: JARVIS — precise, formal, minimal.
    """

    _AGENT_NAME = "Planner"

    def __init__(self, model: str = config.OLLAMA_MODEL) -> None:
        # System prompt not used directly — plan_task manages its own prompt.
        # BaseAgent is still initialised for potential future direct _call use.
        super().__init__(
            system_prompt="",
            model=model,
            temperature=0.2,
            max_tokens=512,
        )

    def plan(self, user_input: str) -> list[Step]:
        """
        Decompose *user_input* into an ordered, validated list of Steps.
        Delegates to core.planner.plan_task which handles LLM call, JSON parsing,
        and intent validation against EXECUTABLE_INTENTS.

        Args:
            user_input: The raw user request.

        Returns:
            List of Step objects (converted from PlanStep). Always non-empty.
        """
        from core.planner import plan_task, PlanStep
        logger.info(f"[agents:Planner] planning input={user_input!r}")

        plan_steps = plan_task(user_input, model=self.model)

        # Convert PlanStep → Step (agents pipeline uses Step dataclass)
        steps = [
            Step(
                index=ps.index,
                description=ps.description,
                intent=ps.intent,
                query=ps.query,
            )
            for ps in plan_steps
        ]

        logger.info(f"[agents:Planner] produced {len(steps)} validated step(s)")
        for step in steps:
            logger.debug(f"[agents:Planner] {step}")
        return steps


# ---------------------------------------------------------------------------
# Executor Agent  (FRIDAY persona)
# ---------------------------------------------------------------------------

_EXECUTOR_SYSTEM_PROMPT = """You are FRIDAY, the Executor Agent of a multi-agent AI system.

Your responsibility: execute a single task step and return its result.

Rules:
- You will receive a step description and a query.
- For "reasoning" steps: think through the query carefully and provide a complete, accurate answer.
- For all other intents: confirm what action you would take and produce the best possible result from your knowledge.
- Be direct and concise — no preamble, no meta-commentary about being an AI.
- Output only the result of the step. Nothing else.
- If you cannot complete the step, explain specifically what is missing."""


class ExecutorAgent(BaseAgent):
    """
    Executes individual Steps.
    Falls back to LLM reasoning for steps whose intent has no live skill handler.
    Persona: FRIDAY — warm, efficient, action-oriented.
    """

    _AGENT_NAME = "Executor"

    def __init__(self, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(
            system_prompt=_EXECUTOR_SYSTEM_PROMPT,
            model=model,
            temperature=0.5,
            max_tokens=768,
        )
        # Optional: register live skill dispatchers keyed by intent name
        # Populated externally via register_skill_handler()
        self._skill_handlers: dict[str, callable] = {}

    def register_skill_handler(self, intent: str, handler: callable) -> None:
        """
        Attach a live skill function for a given intent.

        Args:
            intent: Intent name (e.g. "weather").
            handler: Callable that accepts a query string and returns a result string.
        """
        self._skill_handlers[intent] = handler
        logger.info(f"[agents:Executor] registered handler for intent={intent!r}")

    def execute(self, step: Step) -> ExecutionResult:
        """
        Execute a single Step and return an ExecutionResult.

        Args:
            step: The Step to execute.

        Returns:
            ExecutionResult with output and success flag.
        """
        logger.info(f"[agents:Executor] executing {step}")

        # Prefer live skill handler if registered
        handler = self._skill_handlers.get(step.intent)
        if handler:
            try:
                output = handler(step.query)
                logger.info(f"[agents:Executor] skill handler success intent={step.intent!r}")
                return ExecutionResult(step=step, output=str(output), success=True)
            except Exception as exc:
                logger.error(f"[agents:Executor] skill handler error intent={step.intent!r}: {exc}")
                # fall through to LLM reasoning

        # LLM reasoning fallback
        prompt = (
            f"Step {step.index}: {step.description}\n"
            f"Query: {step.query}"
        )
        output = self._call(prompt)

        if output:
            return ExecutionResult(step=step, output=output, success=True)
        else:
            return ExecutionResult(
                step=step,
                output="I was unable to complete this step.",
                success=False,
                error="LLM returned empty response",
            )

    def execute_all(self, steps: list[Step]) -> list[ExecutionResult]:
        """
        Execute all steps sequentially, passing prior results as context.

        Args:
            steps: Ordered list of Steps from the Planner.

        Returns:
            List of ExecutionResult in step order.
        """
        results: list[ExecutionResult] = []
        accumulated_context = ""

        for step in steps:
            if accumulated_context:
                # Inject prior outputs so later steps can reference them
                original_call = self._call

                def _call_with_context(msg: str, _ctx=accumulated_context) -> str:
                    return original_call(msg, extra_context=_ctx)

                self._call = _call_with_context
                result = self.execute(step)
                self._call = original_call
            else:
                result = self.execute(step)

            results.append(result)
            accumulated_context += f"Step {step.index} result: {result.output}\n"

        logger.info(f"[agents:Executor] completed {len(results)} step(s)")
        return results


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """You are the Critic Agent of a multi-agent AI system.

Your responsibility: evaluate the quality of a composed response and decide whether it is acceptable.

Evaluation criteria:
1. Accuracy — does the response correctly address the original request?
2. Completeness — are all parts of the request handled?
3. Clarity — is the response clear and well-structured?
4. Tone — is it appropriate for an AI assistant?

Output ONLY valid JSON with these exact keys:
{
  "passed": true | false,
  "score": <integer 1-10>,
  "notes": "<brief explanation of verdict>",
  "revised_response": "<improved version if passed=false, otherwise empty string>"
}

Be strict. Score below 6 → passed=false. Always provide revised_response when passed=false."""


class CriticAgent(BaseAgent):
    """
    Validates the quality of the composed response.
    Returns the original or a revised version based on its evaluation.
    """

    _AGENT_NAME = "Critic"

    def __init__(self, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(
            system_prompt=_CRITIC_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=512,
        )

    def critique(
        self,
        original_input: str,
        composed_response: str,
        step_results: list[ExecutionResult],
    ) -> tuple[bool, str, str]:
        """
        Evaluate the composed response against the original user request.

        Args:
            original_input: The raw user query.
            composed_response: The synthesised response to evaluate.
            step_results: All execution results for additional context.

        Returns:
            Tuple of (passed: bool, final_response: str, notes: str).
        """
        logger.info("[agents:Critic] evaluating response quality")

        context_lines = "\n".join(
            f"Step {r.step.index} [{r.step.intent}]: {r.output}" for r in step_results
        )

        prompt = (
            f"Original user request: {original_input}\n\n"
            f"Step execution results:\n{context_lines}\n\n"
            f"Composed response to evaluate:\n{composed_response}"
        )

        raw = self._call(prompt)
        passed, final_response, notes = self._parse_verdict(raw, composed_response)

        logger.info(
            f"[agents:Critic] verdict passed={passed} notes={notes!r}"
        )
        return passed, final_response, notes

    def _parse_verdict(
        self, raw: str, fallback_response: str
    ) -> tuple[bool, str, str]:
        """Parse the Critic's JSON verdict, with safe fallback on parse failure."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            logger.warning(f"[agents:Critic] no JSON found — accepting response as-is")
            return True, fallback_response, "Critic parse failed; response accepted"

        try:
            data: dict = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.error(f"[agents:Critic] JSON decode error: {exc}")
            return True, fallback_response, "Critic parse error; response accepted"

        passed: bool = bool(data.get("passed", True))
        notes: str = str(data.get("notes", ""))
        revised: str = str(data.get("revised_response", "")).strip()

        final_response = revised if (not passed and revised) else fallback_response
        return passed, final_response, notes


# ---------------------------------------------------------------------------
# Composer  (internal synthesis helper — not an Ollama agent)
# ---------------------------------------------------------------------------

_COMPOSER_SYSTEM_PROMPT = """You are a response composer for an AI assistant.

Given a user's original request and a set of step results, compose a single, coherent, natural assistant response.

Rules:
- Write in first person as the AI assistant.
- Be concise — no filler, no meta-commentary.
- Integrate all relevant step results naturally.
- Match the tone: professional but friendly."""


class ComposerAgent(BaseAgent):
    """
    Synthesises individual step results into a single coherent response.
    Lightweight internal agent — not part of the public Plan/Execute/Critique loop directly.
    """

    _AGENT_NAME = "Composer"

    def __init__(self, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(
            system_prompt=_COMPOSER_SYSTEM_PROMPT,
            model=model,
            temperature=0.6,
            max_tokens=512,
        )

    def compose(self, original_input: str, results: list[ExecutionResult]) -> str:
        """
        Combine step results into a final natural-language response.

        Args:
            original_input: The original user request for context.
            results: All ExecutionResult objects from the Executor.

        Returns:
            A single composed response string.
        """
        logger.info("[agents:Composer] composing final response")

        if len(results) == 1:
            # No composition needed for single-step responses
            return results[0].output

        step_summaries = "\n".join(
            f"Step {r.step.index} ({r.step.description}): {r.output}"
            for r in results
        )

        prompt = (
            f"User request: {original_input}\n\n"
            f"Step results:\n{step_summaries}\n\n"
            "Compose a single assistant response."
        )

        composed = self._call(prompt)
        return composed or results[-1].output


# ---------------------------------------------------------------------------
# Multi-Agent Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Coordinates the full Planner → Executor → Composer → Critic pipeline.

    Usage:
        orchestrator = AgentOrchestrator()
        response = orchestrator.run("What's the weather in Paris and summarise the latest news?")
        print(response.final_response)
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        planner: Optional[PlannerAgent] = None,
        executor: Optional[ExecutorAgent] = None,
        composer: Optional[ComposerAgent] = None,
        critic: Optional[CriticAgent] = None,
    ) -> None:
        self.planner = planner or PlannerAgent(model=model)
        self.executor = executor or ExecutorAgent(model=model)
        self.composer = composer or ComposerAgent(model=model)
        self.critic = critic or CriticAgent(model=model)
        logger.info(f"[agents:Orchestrator] initialised model={model}")

    def register_skill(self, intent: str, handler: callable) -> None:
        """
        Register a live skill handler with the Executor for a given intent.

        Args:
            intent: Intent name (must match router intent categories).
            handler: Callable(query: str) -> str.
        """
        self.executor.register_skill_handler(intent, handler)

    def run(self, user_input: str) -> AgentResponse:
        """
        Execute the full multi-agent pipeline for a user request.

        Flow: user_input → Planner → Steps → Executor → Results → Composer → Critic → AgentResponse

        Args:
            user_input: Raw text or transcribed voice input from the user.

        Returns:
            AgentResponse containing all steps, results, and the final validated response.
        """
        logger.info(f"[agents:Orchestrator] run input={user_input!r}")

        # 1. Plan
        steps = self.planner.plan(user_input)

        # 2. Execute
        results = self.executor.execute_all(steps)

        # 3. Compose
        composed = self.composer.compose(user_input, results)

        # 4. Critique
        passed, final_response, notes = self.critic.critique(
            original_input=user_input,
            composed_response=composed,
            step_results=results,
        )

        logger.info(
            f"[agents:Orchestrator] pipeline complete "
            f"steps={len(steps)} critic_passed={passed}"
        )

        return AgentResponse(
            original_input=user_input,
            steps=steps,
            results=results,
            final_response=final_response,
            critic_passed=passed,
            critic_notes=notes,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

orchestrator = AgentOrchestrator()
