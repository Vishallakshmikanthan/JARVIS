from __future__ import annotations

import sqlite3
import json
from datetime import datetime
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
    timestamp   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT    NOT NULL DEFAULT 'general',
    content     TEXT    NOT NULL,
    source      TEXT,
    created_at  TEXT    NOT NULL
);
"""


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
    """Create tables if they don't already exist."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info(f"Memory database initialised at {DB_PATH}")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def log_interaction(role: str, message: str, persona: str = "JARVIS") -> None:
    """Store a single conversation turn (user or assistant)."""
    ts = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (role, message, persona, timestamp) VALUES (?, ?, ?, ?)",
            (role, message, persona, ts),
        )
    logger.debug(f"Logged [{role}] message ({len(message)} chars)")


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

def add_fact(content: str, category: str = "general", source: Optional[str] = None) -> None:
    """Store a new knowledge fact."""
    ts = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO facts (category, content, source, created_at) VALUES (?, ?, ?, ?)",
            (category, content, source, ts),
        )
    logger.debug(f"Fact added [{category}]: {content[:60]}...")


def search_facts(query: str, category: Optional[str] = None, limit: int = 10) -> list[dict]:
    """
    Simple keyword search across facts using SQL LIKE.
    For semantic search, use semantic_search() instead.
    """
    pattern = f"%{query}%"
    with _connect() as conn:
        if category:
            rows = conn.execute(
                "SELECT id, category, content, source, created_at FROM facts "
                "WHERE content LIKE ? AND category = ? ORDER BY id DESC LIMIT ?",
                (pattern, category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category, content, source, created_at FROM facts "
                "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
    return [dict(r) for r in rows]


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
# Auto-initialise on first import
# ---------------------------------------------------------------------------
init_db()
