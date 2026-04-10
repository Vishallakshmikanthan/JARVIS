from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import ollama
from loguru import logger

from config import config
from core.memory import session_ctx
from core.persona import persona_manager

# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------

INTENTS = (
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
    "general",
)

# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------

@dataclass
class Route:
    """Describes where a user utterance should be dispatched."""
    intent: str               # one of INTENTS
    module: str               # e.g. "skills.weather"
    function: str             # e.g. "get_weather"
    parsed_input: str         # cleaned / extracted query for the handler
    confidence: str = "keyword"  # "keyword" | "llm"

    def __repr__(self) -> str:
        return f"<Route {self.intent} → {self.module}.{self.function}('{self.parsed_input[:40]}') [{self.confidence}]>"


# ---------------------------------------------------------------------------
# Intent → handler mapping
# ---------------------------------------------------------------------------

_INTENT_MAP: dict[str, tuple[str, str]] = {
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
    "general":    ("core.brain",               "think"),
}

# ---------------------------------------------------------------------------
# Keyword patterns  (compiled once)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    ("weather",    re.compile(r"\b(weather|temperature|forecast|humid|rain|sunny|cold|hot)\b", re.I)),
    ("news",       re.compile(r"\b(news|headline|latest\s+update|current\s+event|breaking)\b", re.I)),
    ("search",     re.compile(r"\b(search|google|look\s?up|find\s+online|browse)\b", re.I)),
    ("wiki",       re.compile(r"\b(wiki|wikipedia|who\s+is|what\s+is|tell\s+me\s+about|define)\b", re.I)),
    ("music",      re.compile(r"\b(play|song|music|spotify|track|album|pause|resume|skip|next\s+song)\b", re.I)),
    ("system",     re.compile(r"\b(cpu|ram|memory|battery|disk|shutdown|restart|system\s+info|ip\s+address|volume|brightness)\b", re.I)),
    ("joke",       re.compile(r"\b(joke|funny|laugh|humour|humor|make\s+me\s+laugh)\b", re.I)),
    ("persona",    re.compile(r"\b(switch\s+to|activate|become|change\s+persona|switch\s+persona)\s*(jarvis|friday)\b", re.I)),
    ("automation", re.compile(r"\b(schedule|automat|remind\s+me|every\s+\d+|cron|recurring|daily)\b", re.I)),
    ("file",       re.compile(r"\b(read\s+file|open\s+file|write\s+to|save\s+to|delete\s+file|list\s+files|create\s+file|append\s+to|show\s+workspace)\b", re.I)),
    ("module",     re.compile(r"\b(enable\s+\w+|disable\s+\w+|turn\s+on|turn\s+off|list\s+modules|module\s+status)\b", re.I)),
    ("context",    re.compile(r"\b(conversation\s+context|what\s+have\s+we|current\s+topic|session\s+stats|clear\s+context)\b", re.I)),
]

# ---------------------------------------------------------------------------
# Keyword-based routing
# ---------------------------------------------------------------------------

def _keyword_classify(text: str) -> Optional[str]:
    """Return the first matching intent via regex, or None."""
    for intent, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            return intent
    return None


# ---------------------------------------------------------------------------
# Follow-up / context-continuation detection
# ---------------------------------------------------------------------------

# Phrases that strongly signal a follow-up to the previous turn
_FOLLOWUP_PATTERN = re.compile(
    r"^(what|how)\s+about\b"       # "what about tomorrow?", "how about Paris?"
    r"|^and\s+\w"                   # "and in London?"
    r"|^also\b"                     # "also next week"
    r"|^(what|how)\s+if\b"         # "what if it rains?"
    r"|^(it|that|this|there|those|they)\b",  # pronoun-led short queries
    re.I,
)

# Max token count for a query that has no strong intent keyword to still be
# treated as a follow-up (e.g. "tomorrow?", "in Berlin?")
_FOLLOWUP_MAX_TOKENS = 6


def _is_followup(text: str) -> bool:
    """Return True when *text* looks like a context-dependent follow-up."""
    if _FOLLOWUP_PATTERN.match(text):
        return True
    # Short input with zero strong-intent keywords → probably a follow-up
    return len(text.split()) <= _FOLLOWUP_MAX_TOKENS and _keyword_classify(text) is None


# ---------------------------------------------------------------------------
# LLM-based intent classification
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "You are an intent classifier. Given a user message, reply with EXACTLY ONE word "
    "from this list: weather, news, search, wiki, music, system, joke, persona, automation, file, module, context, general.\n"
    "Do NOT explain. Do NOT add punctuation. Just the single word.\n\n"
    "User message: {input}"
)


def _llm_classify(text: str) -> Optional[str]:
    """
    Ask Ollama to classify the intent.
    Returns a valid intent string or None on failure.
    """
    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict intent classifier. Reply with one word only."},
                {"role": "user", "content": _CLASSIFY_PROMPT.format(input=text)},
            ],
            options={"temperature": 0.0, "num_predict": 10},
        )
        raw = response["message"]["content"].strip().lower().rstrip(".")
        if raw in INTENTS:
            logger.debug(f"LLM classified '{text[:50]}' → {raw}")
            return raw
        logger.warning(f"LLM returned invalid intent: '{raw}'")
        return None
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Input extraction helpers
# ---------------------------------------------------------------------------

def _extract_persona_target(text: str) -> str:
    """Pull out 'JARVIS' or 'FRIDAY' from a persona-switch command."""
    match = re.search(r"(jarvis|friday)", text, re.I)
    return match.group(1).upper() if match else ""


def _strip_trigger_words(text: str, intent: str) -> str:
    """Remove common trigger phrases so the handler gets a cleaner query."""
    removals = {
        "weather":    r"(what('?s| is)?\s+the\s+)?weather\s*(in|at|for)?\s*",
        "news":       r"(latest|get|show|tell\s+me)\s*(the\s+)?(news|headlines?)\s*(about|on|for)?\s*",
        "search":     r"(search|google|look\s?up|find\s+online|browse)\s*(for)?\s*",
        "wiki":       r"(wiki(pedia)?|who\s+is|what\s+is|tell\s+me\s+about|define)\s*",
        "music":      r"(play|put\s+on)\s*(the\s+)?(song|track|music)?\s*",
        "joke":       r"(tell\s+me\s+a\s+)?joke\s*",
        "automation": r"(schedule\s+(a\s+)?(task|reminder|job)|remind\s+me)\s*",
        "file":       r"(read|open|write\s+to|save\s+to|delete|create|append\s+to)\s+(file\s+)?",
        "module":     r"(enable|disable|turn\s+on|turn\s+off)\s+(the\s+)?",
        "context":    r"(show|display|get)\s+(conversation\s+)?(context|history|topic)\s*",
    }
    pattern = removals.get(intent)
    if pattern:
        cleaned = re.sub(pattern, "", text, flags=re.I).strip()
        return cleaned if cleaned else text
    return text


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

class Router:
    """
    Routes user input to the correct skill module.
    Strategy: try LLM classification first, fall back to keyword matching,
    and default to 'general' (LLM chat).
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    def route(self, user_input: str) -> Route:
        text = user_input.strip()
        intent: Optional[str] = None
        confidence = "keyword"
        query_text = text  # may be enriched below

        # --- 0. Context-aware follow-up resolution ---
        if session_ctx.has_context() and _is_followup(text):
            intent = session_ctx.last_intent
            confidence = "context"
            # Enrich the query: prepend known entities from the previous turn
            # e.g. last_entities="London", text="what about tomorrow?" → "London tomorrow"
            stripped_followup = re.sub(
                r"^(what|how)\s+about\s*|^and\s+|^also\s+|^(it|that|this|there|those|they)\s*",
                "",
                text,
                flags=re.I,
            ).strip()
            query_text = f"{session_ctx.last_entities} {stripped_followup}".strip()
            logger.debug(
                f"Context follow-up | inherited='{intent}' "
                f"enriched='{query_text[:60]}'"
            )

        # --- 1. Try LLM classification ---
        if intent is None and self.use_llm:
            intent = _llm_classify(text)
            if intent:
                confidence = "llm"

        # --- 2. Fallback to keyword matching ---
        if intent is None:
            intent = _keyword_classify(text)
            confidence = "keyword"

        # --- 3. Default to general ---
        if intent is None:
            intent = "general"
            confidence = "keyword"

        # --- Build the parsed input ---
        if intent == "persona":
            parsed = _extract_persona_target(query_text)
        elif intent == "general":
            parsed = query_text
        else:
            parsed = _strip_trigger_words(query_text, intent)

        module, function = _INTENT_MAP[intent]

        route = Route(
            intent=intent,
            module=module,
            function=function,
            parsed_input=parsed,
            confidence=confidence,
        )
        logger.info(f"Router → {route}")

        # --- Update session context for the next turn ---
        session_ctx.update(intent, parsed, text)

        return route

    def route_keyword_only(self, user_input: str) -> Route:
        """Force keyword-only routing (skips LLM classification)."""
        text = user_input.strip()
        query_text = text

        # Honour context for follow-up queries even in keyword-only mode
        if session_ctx.has_context() and _is_followup(text):
            intent = session_ctx.last_intent
            confidence = "context"
            stripped_followup = re.sub(
                r"^(what|how)\s+about\s*|^and\s+|^also\s+|^(it|that|this|there|those|they)\s*",
                "",
                text,
                flags=re.I,
            ).strip()
            query_text = f"{session_ctx.last_entities} {stripped_followup}".strip()
        else:
            intent = _keyword_classify(text) or "general"
            confidence = "keyword"

        if intent == "persona":
            parsed = _extract_persona_target(query_text)
        elif intent == "general":
            parsed = query_text
        else:
            parsed = _strip_trigger_words(query_text, intent)

        module, function = _INTENT_MAP[intent]
        route = Route(intent=intent, module=module, function=function,
                      parsed_input=parsed, confidence=confidence)
        session_ctx.update(intent, parsed, text)
        return route


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
router = Router(use_llm=True)
