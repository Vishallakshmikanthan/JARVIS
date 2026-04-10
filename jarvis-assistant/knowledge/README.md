# JARVIS Local Knowledge Base

Drop any of the following file types into this folder and they will be
automatically indexed at startup:

- `.pdf`   — research papers, manuals, reports
- `.txt`   — plain text notes
- `.md`    — markdown notes
- `.rst`   — reStructuredText documents

Sub-directories are scanned recursively.

## How it works

1. Files are chunked into overlapping 400-character segments.
2. Chunks are indexed with TF-IDF embeddings (upgradeable to dense
   sentence-transformers by installing `sentence-transformers`).
3. When a query is routed through `brain.think()`, the KB is checked
   first via cosine similarity before falling through to the LLM.
4. Top matching chunks are fed into the LLM as grounded context.

## Programmatic usage

```python
from core.knowledge_base import knowledge_base

# Add a single file
knowledge_base.add_document("path/to/file.pdf")

# Direct semantic search
results = knowledge_base.search("How does X work?")

# Full grounded answer
answer = knowledge_base.ask_local_knowledge("Summarise the key points about X")

# Persist index to disk
knowledge_base.save()
```
