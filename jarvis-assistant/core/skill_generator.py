"""
core/skill_generator.py — Dynamic Skill Generator

Listens for commands like "Create a new skill for X" and:
  1. Uses the LLM to generate a well-formed Python skill file.
  2. Writes the file to the skills/ directory.
  3. Validates the generated code (syntax + required function signature).
  4. Auto-registers the new intent in the live Router instance.
"""
from __future__ import annotations

import ast
import importlib
import re
import textwrap
from pathlib import Path

import ollama
from loguru import logger

from config import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Trigger pattern: "create a new skill for X" / "generate skill for X"
_TRIGGER_PATTERN = re.compile(
    r"\b(create|generate|make|build|add)\s+(a\s+)?(new\s+)?skill\s+(for|called|named|that|to)\s+(.+)",
    re.IGNORECASE,
)

# Template injected into the LLM prompt so it understands the convention
_SKILL_TEMPLATE_EXAMPLE = textwrap.dedent("""\
    from __future__ import annotations
    from loguru import logger

    def {function_name}(user_input: str) -> str:
        \"\"\"
        One-line summary of what this skill does.
        \"\"\"
        # TODO: implement logic here
        return "Skill result"
""")

# Full LLM prompt for skill generation
_GEN_PROMPT = textwrap.dedent("""\
    You are an expert Python developer building a skill module for an AI assistant called JARVIS.

    The user wants a new skill for: "{purpose}"

    Rules:
    - Return ONLY valid Python code. No markdown fences, no prose.
    - The file must import `from loguru import logger` at the top.
    - It must define exactly ONE public entry-point function named `{function_name}`.
    - That function must accept a single argument `user_input: str` and return `str`.
    - Use the `requests` library for any HTTP calls (imported inside the function is fine).
    - Handle ALL exceptions with try/except and return a human-readable error string.
    - Do NOT use any credentials or API keys; use free/public APIs when possible.
    - Keep the function under 60 lines.

    Example skeleton:
    {template}

    Now write the full skill file for: "{purpose}"
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_skill_creation_command(text: str) -> str | None:
    """
    Return the skill *purpose* string if *text* is a skill-creation command,
    otherwise return None.
    """
    m = _TRIGGER_PATTERN.search(text)
    return m.group(5).strip().rstrip(".?!") if m else None


def _purpose_to_names(purpose: str) -> tuple[str, str]:
    """
    Derive a safe snake_case skill name and an entry-point function name
    from a free-form purpose string.

    Returns (skill_name, function_name).
    e.g. "telling riddles" → ("riddles", "tell_riddles")
    """
    # Strip common filler verbs at the start
    cleaned = re.sub(
        r"^(getting|telling|showing|fetching|checking|finding|playing|"
        r"searching|calculating|managing|tracking|sending|reading|writing)\s+",
        "",
        purpose,
        flags=re.I,
    ).strip()

    # Build a snake_case slug from the first 4 meaningful words
    words = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned).lower().split()[:4]
    slug = "_".join(words) if words else "custom"
    return slug, slug


def _call_llm(purpose: str, function_name: str) -> str:
    """Call Ollama to generate the skill source code."""
    prompt = _GEN_PROMPT.format(
        purpose=purpose,
        function_name=function_name,
        template=_SKILL_TEMPLATE_EXAMPLE.format(function_name=function_name),
    )
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict Python code generator. "
                    "Return ONLY raw Python code. No backticks, no markdown, no comments outside the code."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2, "num_predict": 1024},
    )
    raw: str = response["message"]["content"].strip()

    # Strip any accidental markdown fences the LLM may add
    raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.M)
    raw = re.sub(r"^```\s*$", "", raw, flags=re.M)
    return raw.strip()


def _validate_skill(source: str, function_name: str) -> tuple[bool, str]:
    """
    Validate the generated skill source:
      1. Must parse as valid Python.
      2. Must contain the expected entry-point function.
      3. Entry-point must accept at least one argument.

    Returns (is_valid, reason).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Find the entry-point function
    func_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if not func_defs:
        return False, f"Entry-point function '{function_name}' not found in generated code."

    fn = func_defs[0]
    num_args = len(fn.args.args)
    if num_args < 1:
        return False, f"Function '{function_name}' must accept at least one argument (user_input)."

    return True, "ok"


def _write_skill_file(skill_name: str, source: str) -> Path:
    """Write the generated source code to skills/<skill_name>.py."""
    path = SKILLS_DIR / f"{skill_name}.py"
    path.write_text(source, encoding="utf-8")
    logger.success(f"Skill file written → {path}")
    return path


def _register_in_router(skill_name: str, function_name: str, purpose: str) -> None:
    """
    Hot-patch the live Router singleton to recognise the new skill:
      - Adds an entry to _INTENT_MAP
      - Adds a keyword rule to _KEYWORD_RULES
      - Adds the intent to the INTENTS tuple variant in router
    """
    from core import router as router_module

    intent_key = skill_name

    # Guard: don't register duplicates
    if intent_key in router_module._INTENT_MAP:
        logger.info(f"Intent '{intent_key}' already registered — skipping re-registration.")
        return

    # 1. Extend _INTENT_MAP
    router_module._INTENT_MAP[intent_key] = (f"skills.{skill_name}", function_name)

    # 2. Build a simple keyword pattern from the purpose words
    if kw_words := re.sub(r"[^a-zA-Z0-9 ]", "", purpose).lower().split()[:6]:
        pattern_str = r"\b(" + "|".join(re.escape(w) for w in kw_words) + r")\b"
        router_module._KEYWORD_RULES.append(
            (intent_key, re.compile(pattern_str, re.I))
        )

    # 3. Patch the INTENTS constant (tuple → new tuple)
    if intent_key not in router_module.INTENTS:
        router_module.INTENTS = router_module.INTENTS + (intent_key,)

    # 4. Update the LLM classifier prompt to know about the new intent
    router_module._CLASSIFY_PROMPT = (
        "You are an intent classifier. Given a user message, reply with EXACTLY ONE word "
        f"from this list: {', '.join(router_module.INTENTS)}.\n"
        "Do NOT explain. Do NOT add punctuation. Just the single word.\n\n"
        "User message: {input}"
    )

    logger.success(f"Skill '{skill_name}' auto-registered in router as intent '{intent_key}'.")


def _try_import_skill(skill_name: str, function_name: str) -> str:
    """
    Attempt to import the generated skill module and call the entry-point
    with a test string to get a sanity-check response.
    Returns a brief status string for the user.
    """
    try:
        module = importlib.import_module(f"skills.{skill_name}")
        fn = getattr(module, function_name)
        test_result = fn(f"test {skill_name}")
        logger.info(f"Skill '{skill_name}' smoke-test result: {test_result[:80]}")
        return "passed"
    except Exception as e:
        logger.warning(f"Skill smoke-test failed for '{skill_name}': {e}")
        return f"smoke-test warning: {e}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def create_skill(purpose: str) -> str:
    """
    Full pipeline:
      1. Derive safe names.
      2. Ask LLM to write the skill.
      3. Validate.
      4. Write to disk.
      5. Auto-register in the live router.
      6. Smoke-test import.

    Returns a human-readable status string.
    """
    skill_name, function_name = _purpose_to_names(purpose)
    logger.info(f"Skill generator: purpose='{purpose}' → skill='{skill_name}' fn='{function_name}'")

    # --- 1. Generate via LLM ---
    try:
        source = _call_llm(purpose, function_name)
    except Exception as e:
        logger.error(f"LLM skill generation failed: {e}")
        return f"I couldn't generate the skill — LLM error: {e}"

    # --- 2. Validate ---
    valid, reason = _validate_skill(source, function_name)
    if not valid:
        logger.error(f"Generated skill failed validation: {reason}")
        return (
            f"I generated a skill for '{purpose}', but it failed validation: {reason}. "
            "Please try again or provide more detail."
        )

    # --- 3. Write to disk ---
    skill_path = _write_skill_file(skill_name, source)

    # --- 4. Auto-register in router ---
    _register_in_router(skill_name, function_name, purpose)

    # --- 5. Smoke-test ---
    test_status = _try_import_skill(skill_name, function_name)

    return (
        f"Skill '{skill_name}' created successfully and registered as a new intent.\n"
        f"Entry-point: skills.{skill_name}.{function_name}(user_input)\n"
        f"File: {skill_path.name}\n"
        f"Validation: {test_status}"
    )
