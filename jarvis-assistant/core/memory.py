from __future__ import annotations

import re
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — semantic_search will be unavailable.")

from config import config

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
DB_PATH = config.DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    message     TEXT    NOT NULL,
    persona     TEXT    NOT NULL DEFAULT 'JARVIS',
    timestamp   TEXT    NOT NULL,
    importance  REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    category      TEXT    NOT NULL DEFAULT 'general',
    content       TEXT    NOT NULL,
    source        TEXT,
    created_at    TEXT    NOT NULL,
    importance    REAL    NOT NULL DEFAULT 0.5,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);
"""

# Migration: add new columns to existing databases that predate this schema
_MIGRATIONS = [
    "ALTER TABLE conversations ADD COLUMN importance REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE facts ADD COLUMN importance    REAL    NOT NULL DEFAULT 0.5",
    "ALTER TABLE facts ADD COLUMN access_count  INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE facts ADD COLUMN last_accessed TEXT",
]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Return a new connection with WAL mode and row-factory enabled."""
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create tables and apply any pending column migrations."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        for stmt in _MIGRATIONS:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists — safe to skip
    logger.info(f"Memory database initialised at {DB_PATH}")


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

# Weighted keyword signals.  Keys are regex patterns; values are score boosts.
_IMPORTANCE_SIGNALS: list[tuple[re.Pattern, float]] = [
    # Explicit memory requests
    (re.compile(r"\b(remember|don'?t forget|note that|keep in mind|save this)\b", re.I), 0.40),
    # Personal facts
    (re.compile(r"\b(my name is|i am|i'm|i live|i work|i like|i love|i hate|i prefer|i always|i never)\b", re.I), 0.30),
    # Preferences / settings
    (re.compile(r"\b(prefer|favourite|favorite|always|never|every time|set .* to|change .* to)\b", re.I), 0.20),
    # Named entities heuristic: Title Case words (excluding sentence starts)
    (re.compile(r"(?<![.!?]\s)\b[A-Z][a-z]{2,}\b"), 0.05),
    # Numbers / dates that might be important
    (re.compile(r"\b\d{4}\b|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), 0.05),
    # Questions — user is actively seeking knowledge (moderate importance)
    (re.compile(r"\?"), 0.10),
]

# Patterns for extracting auto-facts from interactions
_FACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"my name is ([\w\s]+)", re.I),          "identity"),
    (re.compile(r"i(?:'m| am) ([\w\s]+)", re.I),          "identity"),
    (re.compile(r"i live (?:in |at )?([\w\s,]+)", re.I),  "location"),
    (re.compile(r"i work (?:at |for )?([\w\s]+)", re.I),  "occupation"),
    (re.compile(r"i (?:like|love|enjoy) ([\w\s]+)", re.I),"preference"),
    (re.compile(r"i (?:hate|dislike|don'?t like) ([\w\s]+)", re.I), "preference"),
    (re.compile(r"i prefer ([\w\s]+)", re.I),             "preference"),
    (re.compile(r"remember (?:that )?(.+)", re.I),         "explicit"),
    (re.compile(r"note (?:that )?(.+)", re.I),            "explicit"),
]

# Thresholds
_IMPORTANCE_AUTO_SAVE_THRESHOLD: float = 0.45   # save fact automatically
_IMPORTANCE_RETAIN_THRESHOLD: float = 0.15      # keep in DB; below this → purgeable
_MAX_STALE_DAYS: int = 30                        # default age for forget pass


def score_importance(text: str) -> float:
    """
    Return an importance score in [0.0, 1.0] for *text*.

    Combines:
    - Keyword / pattern signals          (weighted additive)
    - Message length bonus               (longer = slightly more important)
    - Frequency proxy via repeated words (very approximate)
    """
    score = 0.0
    for pattern, boost in _IMPORTANCE_SIGNALS:
        if matches := pattern.findall(text):
            score += boost * min(len(matches), 3)  # cap repeated matches

    # Length bonus: 0 → 0.05 at ~200 chars
    length_bonus = min(len(text) / 4000, 0.05)
    score += length_bonus

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def log_interaction(
    role: str,
    message: str,
    persona: str = "JARVIS",
    importance: Optional[float] = None,
) -> None:
    """Store a single conversation turn with an auto-computed importance score."""
    ts = datetime.now().isoformat()
    imp = importance if importance is not None else score_importance(message)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (role, message, persona, timestamp, importance) "
            "VALUES (?, ?, ?, ?, ?)",
            (role, message, persona, ts, imp),
        )
    logger.debug(f"Logged [{role}] message ({len(message)} chars) importance={imp:.3f}")


def get_important_conversations(
    min_importance: float = 0.4,
    limit: int = 20,
    persona: Optional[str] = None,
) -> list[dict]:
    """Return the highest-importance conversations, newest first."""
    with _connect() as conn:
        if persona:
            rows = conn.execute(
                "SELECT role, message, persona, timestamp, importance FROM conversations "
                "WHERE importance >= ? AND persona = ? ORDER BY importance DESC, id DESC LIMIT ?",
                (min_importance, persona, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, message, persona, timestamp, importance FROM conversations "
                "WHERE importance >= ? ORDER BY importance DESC, id DESC LIMIT ?",
                (min_importance, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def get_recent_conversations(limit: int = 20, persona: Optional[str] = None) -> list[dict]:
    """
    Retrieve the most recent conversation turns.
    Optionally filter by persona.
    """
    with _connect() as conn:
        if persona:
            rows = conn.execute(
                "SELECT role, message, persona, timestamp FROM conversations "
                "WHERE persona = ? ORDER BY id DESC LIMIT ?",
                (persona, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, message, persona, timestamp FROM conversations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    # Return in chronological order
    return [dict(r) for r in reversed(rows)]


def get_conversation_context(limit: int = 10, persona: Optional[str] = None) -> list[dict]:
    """
    Return recent turns formatted as Ollama-style message dicts
    [{"role": "user", "content": "..."}, ...].
    """
    rows = get_recent_conversations(limit, persona)
    return [{"role": r["role"], "content": r["message"]} for r in rows]


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def store_preference(key: str, value) -> None:
    """Upsert a preference (value is JSON-serialised)."""
    ts = datetime.now().isoformat()
    serialised = json.dumps(value)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, serialised, ts),
        )
    logger.debug(f"Preference '{key}' stored")


def retrieve_preference(key: str, default=None):
    """Fetch a single preference by key, deserialised from JSON."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def get_all_preferences() -> dict:
    """Return all preferences as a {key: value} dict."""
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM preferences").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def add_fact(
    content: str,
    category: str = "general",
    source: Optional[str] = None,
    importance: float = 0.5,
) -> None:
    """Store a new knowledge fact; silently skips near-duplicates."""
    # Dedup: skip if an almost identical fact already exists
    pattern = f"%{content[:40]}%"
    with _connect() as conn:
        if existing := conn.execute(
            "SELECT id FROM facts WHERE content LIKE ?", (pattern,)
        ).fetchone():
            logger.debug(f"Duplicate fact skipped: '{content[:60]}'")
            return
        ts = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO facts (category, content, source, created_at, importance) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, content, source, ts, importance),
        )
    logger.debug(f"Fact added [{category}] importance={importance:.2f}: {content[:60]}")


def search_facts(query: str, category: Optional[str] = None, limit: int = 10) -> list[dict]:
    """
    Simple keyword search across facts using SQL LIKE.
    Increments `access_count` and updates `last_accessed` for each returned fact.
    For semantic search, use semantic_search() instead.
    """
    pattern = f"%{query}%"
    with _connect() as conn:
        if category:
            rows = conn.execute(
                "SELECT id, category, content, source, created_at, importance, access_count "
                "FROM facts WHERE content LIKE ? AND category = ? "
                "ORDER BY importance DESC, id DESC LIMIT ?",
                (pattern, category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category, content, source, created_at, importance, access_count "
                "FROM facts WHERE content LIKE ? "
                "ORDER BY importance DESC, id DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
        result = [dict(r) for r in rows]
        # Update access tracking
        now = datetime.now().isoformat()
        for row in result:
            conn.execute(
                "UPDATE facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, row["id"]),
            )
    return result


def get_important_facts(min_importance: float = 0.6, limit: int = 20) -> list[dict]:
    """Return facts scored at or above *min_importance*, ordered by importance then access frequency."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, category, content, source, created_at, importance, access_count "
            "FROM facts WHERE importance >= ? "
            "ORDER BY importance DESC, access_count DESC LIMIT ?",
            (min_importance, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# learn_from_interaction
# ---------------------------------------------------------------------------

def learn_from_interaction(
    user_input: str,
    response: str,
    persona: str = "JARVIS",
) -> dict:
    """
    Analyse a completed interaction, score its importance, auto-save facts,
    and return a summary of what was learned.

    Returns
    -------
    dict with keys:
        importance   (float)   combined importance score
        facts_saved  (int)     number of new facts persisted
        facts        (list)    list of extracted fact strings
    """
    combined_text = f"{user_input} {response}"

    # --- 1. Score importance ---
    user_score = score_importance(user_input)
    # Apply a modest recency boost: interactions just processed are fresh
    combined_score = round(min(user_score * 1.1, 1.0), 4)

    # --- 2. Extract facts from user input ---
    extracted: list[tuple[str, str]] = []   # (content, category)
    for pattern, category in _FACT_PATTERNS:
        for match in pattern.finditer(user_input):
            raw = match.group(1).strip().rstrip(".,!?")
            if len(raw) >= 3:  # ignore noise
                extracted.append((raw, category))

    # --- 3. Persist high-importance facts ---
    facts_saved = 0
    saved_contents: list[str] = []
    if combined_score >= _IMPORTANCE_AUTO_SAVE_THRESHOLD or extracted:
        fact_importance = round(min(combined_score + 0.1, 1.0), 4)
        for content, category in extracted:
            add_fact(
                content=content,
                category=category,
                source="interaction",
                importance=fact_importance,
            )
            facts_saved += 1
            saved_contents.append(content)

    logger.info(
        f"[learn] importance={combined_score:.3f} "
        f"facts_saved={facts_saved} "
        f"facts={saved_contents}"
    )
    return {
        "importance": combined_score,
        "facts_saved": facts_saved,
        "facts": saved_contents,
    }


# ---------------------------------------------------------------------------
# Forget stale / low-importance data
# ---------------------------------------------------------------------------

def forget_stale_conversations(
    max_age_days: int = _MAX_STALE_DAYS,
    min_importance: float = _IMPORTANCE_RETAIN_THRESHOLD,
) -> int:
    """
    Delete conversation rows that are:
      - older than *max_age_days*, AND
      - below *min_importance*

    Returns the number of rows deleted.
    """
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations "
            "WHERE timestamp < ? AND importance < ?",
            (cutoff, min_importance),
        )
        deleted = cur.rowcount
    logger.info(f"[forget] removed {deleted} stale conversation row(s)")
    return deleted


def forget_stale_facts(
    max_age_days: int = _MAX_STALE_DAYS * 2,
    min_importance: float = _IMPORTANCE_RETAIN_THRESHOLD,
    min_access_count: int = 0,
) -> int:
    """
    Delete facts that are old, low-importance, and never accessed.

    A fact survives the purge if *any* of these hold:
      - importance >= *min_importance*
      - access_count > *min_access_count*
      - age <= *max_age_days*
    """
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM facts "
            "WHERE created_at < ? AND importance < ? AND access_count <= ?",
            (cutoff, min_importance, min_access_count),
        )
        deleted = cur.rowcount
    logger.info(f"[forget] removed {deleted} stale fact row(s)")
    return deleted


# ---------------------------------------------------------------------------
# Semantic Search (TF-IDF)
# ---------------------------------------------------------------------------

def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant past conversations using TF-IDF cosine similarity.

    Returns a list of dicts sorted by relevance:
        [{"role": ..., "message": ..., "persona": ..., "timestamp": ..., "score": float}, ...]
    """
    if not _SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for semantic_search. Install it with: pip install scikit-learn")
        return []

    # Fetch all stored messages
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, message, persona, timestamp FROM conversations ORDER BY id"
        ).fetchall()

    if not rows:
        logger.info("No conversations in memory yet — nothing to search.")
        return []

    conversations = [dict(r) for r in rows]
    corpus = [c["message"] for c in conversations]

    # Build TF-IDF matrix with the query appended at the end
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])

    # Cosine similarity between query vector (last) and every conversation
    query_vec = tfidf_matrix[-1]
    similarities = cosine_similarity(query_vec, tfidf_matrix[:-1]).flatten()

    # Attach scores and sort descending
    for idx, conv in enumerate(conversations):
        conv["score"] = round(float(similarities[idx]), 4)

    ranked = sorted(conversations, key=lambda c: c["score"], reverse=True)
    results = [r for r in ranked[:top_k] if r["score"] > 0.0]

    logger.debug(f"semantic_search('{query}') → {len(results)} results (top score: {results[0]['score'] if results else 0})")
    return results


# ---------------------------------------------------------------------------
# Session context  (in-memory, not persisted — reset on each process start)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Tracks intent and entity continuity across conversation turns.

    Used by the router to resolve follow-up questions like
    "What about tomorrow?" when the previous intent was 'weather'.
    """

    last_intent: str = "general"
    last_entities: str = ""      # stripped query from the previous turn
    last_raw_input: str = ""
    turn_count: int = 0
    _history: list[dict] = field(default_factory=list, repr=False)

    def update(self, intent: str, entities: str, raw_input: str) -> None:
        """Record the result of a completed turn."""
        self._history.append({
            "intent": self.last_intent,
            "entities": self.last_entities,
            "raw": self.last_raw_input,
        })
        self.last_intent = intent
        self.last_entities = entities
        self.last_raw_input = raw_input
        self.turn_count += 1
        logger.debug(
            f"SessionContext updated | turn={self.turn_count} "
            f"intent={intent} entities='{entities[:40]}'"
        )

    def has_context(self) -> bool:
        """True when there is a non-trivial previous intent to inherit."""
        return self.last_intent != "general" and bool(self.last_entities)

    def clear(self) -> None:
        """Reset all session state (e.g. when the user starts a new topic)."""
        self.last_intent = "general"
        self.last_entities = ""
        self.last_raw_input = ""
        self.turn_count = 0
        self._history.clear()
        logger.info("SessionContext cleared")

    def recent_history(self, n: int = 3) -> list[dict]:
        """Return the last *n* recorded turns (oldest first)."""
        return self._history[-n:]


# Module-level singleton imported by router.py and other modules
session_ctx = SessionContext()


# ---------------------------------------------------------------------------
# Auto-initialise on first import
# ---------------------------------------------------------------------------
init_db()
