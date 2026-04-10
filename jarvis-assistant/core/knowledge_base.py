"""
core/knowledge_base.py — Local Knowledge Base

Indexes PDFs and plain-text notes stored in the configured KB_DIR and
answers queries via semantic search (TF-IDF cosine similarity by default,
drop-in-upgradeable to sentence-transformers dense embeddings).

Public API
----------
kb.add_document(path)            – index a single file (PDF or .txt/.md)
kb.index_directory(directory)    – recursively index all supported files
kb.ask_local_knowledge(query)    – semantic search → grounded LLM answer
kb.save() / kb.load()            – persist / restore the index to disk
"""
from __future__ import annotations

import json
import pickle
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import config

# ---------------------------------------------------------------------------
# Optional dense-embedding upgrade (sentence-transformers)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _ST
    _DENSE_AVAILABLE = True
except ImportError:
    _DENSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional PDF support (pypdf)
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    _PDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader      # older fallback
        _PDF_AVAILABLE = True
    except ImportError:
        _PDF_AVAILABLE = False
        logger.warning("pypdf not installed — PDF ingestion will be unavailable. Run: pip install pypdf")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # character overlap between consecutive chunks
TOP_K = 5                 # number of chunks returned per query
MIN_SCORE = 0.10          # minimum cosine similarity to include a chunk
INDEX_FILENAME = "kb_index.pkl"
META_FILENAME  = "kb_meta.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One text segment extracted from a source document."""
    text: str
    source: str           # original file path (str)
    chunk_id: int         # sequential index within the source


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(path: Path) -> str:
    if not _PDF_AVAILABLE:
        logger.warning(f"Skipping PDF '{path.name}' — pypdf not installed.")
        return ""
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        logger.error(f"PDF extraction error for '{path}': {e}")
        return ""


def _extract_text(path: Path) -> str:
    """Return raw text from a file regardless of format."""
    if path.suffix.lower() == ".pdf":
        return _extract_text_from_pdf(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Text read error for '{path}': {e}")
        return ""


def _clean(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, source: str) -> list[Chunk]:
    """Split *text* into overlapping character-level chunks."""
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
            if snippet := text[start:end].strip():


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Local knowledge base backed by TF-IDF sparse embeddings (no GPU required).
    Automatically upgrades to dense sentence-transformer embeddings when
    the `sentence-transformers` package is installed.
    """

    def __init__(
        self,
        kb_dir: Path | str = config.KB_DIR,
        index_dir: Path | str = config.KB_INDEX_DIR,
        use_dense: bool = False,
        dense_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.kb_dir    = Path(kb_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self._chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None   # (n_chunks, n_features)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._indexed_sources: set[str] = set()

        # Dense embedding backend (optional)
        self._use_dense = use_dense and _DENSE_AVAILABLE
        self._dense_model: Optional[object] = None
        if self._use_dense:
            logger.info(f"Loading dense embedding model: {dense_model}")
            self._dense_model = _ST(dense_model)

        logger.info(
            f"KnowledgeBase initialised | kb_dir={self.kb_dir} "
            f"mode={'dense' if self._use_dense else 'tfidf'}"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(self, path: Path | str) -> int:
        """
        Index a single file. Returns the number of new chunks added.
        Safe to call multiple times on the same file (deduplication by path).
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"KB: file not found — {path}")
            return 0
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"KB: unsupported file type '{path.suffix}' — {path.name}")
            return 0
        if str(path) in self._indexed_sources:
            logger.debug(f"KB: already indexed — {path.name}")
            return 0

        raw = _extract_text(path)
        if not raw.strip():
            logger.warning(f"KB: no text extracted from '{path.name}'")
            return 0

        cleaned = _clean(raw)
        new_chunks = _chunk_text(cleaned, str(path))
        self._chunks.extend(new_chunks)
        self._indexed_sources.add(str(path))
        self._matrix = None   # invalidate index — will be rebuilt on next query
        logger.success(f"KB: indexed '{path.name}' → {len(new_chunks)} chunks")
        return len(new_chunks)

    def index_directory(self, directory: Path | str | None = None) -> int:
        """
        Recursively index all supported files in *directory* (defaults to kb_dir).
        Returns the total number of new chunks added.
        """
        directory = Path(directory) if directory else self.kb_dir
        if not directory.exists():
            logger.warning(f"KB: directory not found — {directory}")
            return 0

        total = sum(
            self.add_document(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        self._rebuild_index()
        logger.info(f"KB: directory indexed — {total} total new chunks from {directory}")
        return total

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Recompute the embedding matrix from all current chunks."""
        if not self._chunks:
            logger.debug("KB: no chunks to index.")
            return

        texts = [c.text for c in self._chunks]

        if self._use_dense and self._dense_model:
            self._matrix = self._dense_model.encode(texts, show_progress_bar=False)
            logger.debug(f"KB: dense index rebuilt — shape={self._matrix.shape}")
        else:
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=50_000,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(texts)
            logger.debug(f"KB: TF-IDF index rebuilt — shape={self._matrix.shape}")

    def _ensure_index(self) -> None:
        """Build the index lazily if it has been invalidated."""
        if self._matrix is None and self._chunks:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        """
        Return the *top_k* most relevant chunks for *query*.
        Returns an empty list when the KB is empty.
        """
        if not self._chunks:
            return []

        self._ensure_index()

        if self._use_dense and self._dense_model:
            q_vec = self._dense_model.encode([query])
        elif self._vectorizer is None:
            return []
        else:
            q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]

        # Rank and filter
        ranked = sorted(
            ((float(scores[i]), self._chunks[i]) for i in range(len(scores))),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            SearchResult(chunk=chunk, score=score)
            for score, chunk in ranked[:top_k]
            if score >= MIN_SCORE
        ]

    # ------------------------------------------------------------------
    # Public query entry-point
    # ------------------------------------------------------------------

    def ask_local_knowledge(self, query: str) -> Optional[str]:
        """
        Semantic search over indexed documents. If relevant chunks are found,
        returns a grounded answer string. Returns None when no relevant
        content exists so the caller can fall through to a web/LLM path.
        """
        results = self.search(query)
        if not results:
            logger.debug(f"KB: no relevant chunks for query '{query[:60]}'")
            return None

        # Build context block from top results
        context_parts: list[str] = []
        seen_sources: set[str] = set()
        for r in results:
            source_name = Path(r.chunk.source).name
            context_parts.append(
                f"[{source_name} | score={r.score:.2f}]\n{r.chunk.text}"
            )
            seen_sources.add(source_name)

        context_block = "\n\n---\n\n".join(context_parts)

        return self._synthesise_answer(query, context_block, seen_sources, len(results))

    def _synthesise_answer(
        self,
        query: str,
        context_block: str,
        seen_sources: set[str],
        n_chunks: int,
    ) -> str:
        """Call the local LLM to produce a grounded answer from retrieved context."""
        try:
            import ollama as _ollama
            client = _ollama.Client(host=config.OLLAMA_BASE_URL)
            synthesis_prompt = textwrap.dedent(f"""\
                You are JARVIS, an AI assistant. Answer the user's question using ONLY the
                context excerpts below. If the context does not contain the answer, say so.

                USER QUESTION: {query}

                CONTEXT:
                {context_block}

                Answer concisely and cite the source file names when relevant.
            """)
            response = client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                options={"temperature": 0.2, "num_predict": 512},
            )
            answer: str = response["message"]["content"].strip()
            logger.info(
                f"KB: answered query from {n_chunks} chunk(s) "
                f"(sources: {', '.join(seen_sources)})"
            )
            return answer
        except Exception as e:
            logger.error(f"KB: LLM synthesis failed: {e}")
            return f"Here is what I found in your knowledge base:\n\n{context_block}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist chunks + vectorizer/model to the index directory."""
        index_path = self.index_dir / INDEX_FILENAME
        meta_path  = self.index_dir / META_FILENAME

        self._ensure_index()

        try:
            with index_path.open("wb") as f:
                pickle.dump(
                    {
                        "chunks": self._chunks,
                        "matrix": self._matrix,
                        "vectorizer": self._vectorizer,
                        "indexed_sources": self._indexed_sources,
                        "use_dense": self._use_dense,
                    },
                    f,
                )
            meta_path.write_text(
                json.dumps(
                    {
                        "chunk_count": len(self._chunks),
                        "sources": sorted(self._indexed_sources),
                        "mode": "dense" if self._use_dense else "tfidf",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.success(f"KB: index saved → {index_path}")
        except Exception as e:
            logger.error(f"KB: save failed: {e}")

    def load(self) -> bool:
        """
        Restore a previously saved index. Returns True on success.
        Safe to call even when no saved index exists.
        """
        index_path = self.index_dir / INDEX_FILENAME
        if not index_path.exists():
            logger.debug("KB: no saved index found — starting fresh.")
            return False

        try:
            with index_path.open("rb") as f:
                data = pickle.load(f)
            self._restore_from_data(data)
            logger.success(
                f"KB: loaded index — {len(self._chunks)} chunks, "
                f"{len(self._indexed_sources)} source(s)"
            )
            return True
        except Exception as e:
            logger.error(f"KB: load failed: {e}")
            return False

    def _restore_from_data(self, data: dict) -> None:
        """Apply a deserialized index snapshot to this instance."""
        self._chunks          = data["chunks"]
        self._matrix          = data["matrix"]
        self._vectorizer      = data.get("vectorizer")
        self._indexed_sources = data.get("indexed_sources", set())
        self._use_dense       = data.get("use_dense", False)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> str:
        return (
            f"Knowledge base: {len(self._chunks)} chunks indexed from "
            f"{len(self._indexed_sources)} source(s). "
            f"Mode: {'dense' if self._use_dense else 'TF-IDF'}."
        )

    def __repr__(self) -> str:
        return (
            f"<KnowledgeBase chunks={len(self._chunks)} "
            f"sources={len(self._indexed_sources)} "
            f"mode={'dense' if self._use_dense else 'tfidf'}>"
        )


# ---------------------------------------------------------------------------
# Singleton — load any previously saved index at import time
# ---------------------------------------------------------------------------
knowledge_base = KnowledgeBase()
knowledge_base.load()
# Auto-index the KB directory at startup (adds new files since last run)
knowledge_base.index_directory()
