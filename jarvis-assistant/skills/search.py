from __future__ import annotations

from typing import Optional
from duckduckgo_search import DDGS
from loguru import logger

# ---------------------------------------------------------------------------
# Search Skill
# ---------------------------------------------------------------------------

def web_search(user_input: str) -> str:
    """
    Search the web via DuckDuckGo and return a top summary result.
    """
    query = user_input.replace("search for", "").replace("find", "").strip("? .")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return f"No relevant web results found for '{query}'."

            summary = results[0]["body"]
            title = results[0]["title"]
            href = results[0]["href"]
            
            return f"{title}: {summary}\nSource: {href}"

    except Exception as e:
        logger.error(f"Search error: {e}")
        return "I was unable to search the web at this time."
