from __future__ import annotations

import wikipedia
from loguru import logger

# ---------------------------------------------------------------------------
# Wiki Skill (Wikipedia)
# ---------------------------------------------------------------------------

wikipedia.set_lang("en")

def wiki_search(user_input: str) -> str:
    """
    Search Wikipedia for terms or topics.
    """
    query = user_input.replace("wiki", "").replace("wikipedia", "").strip("? .")
    
    try:
        results = wikipedia.search(query, results=5)
        
        if not results:
            return f"I couldn't find a Wikipedia page for '{query}'."

        # Attempt to get summary for the most relevant match
        summary = wikipedia.summary(results[0], sentences=2)
        return summary

    except wikipedia.DisambiguationError as de:
        return f"Multiple matches found: {', '.join(de.options[:5])}."
    except wikipedia.PageError:
        return f"I found the page {query} but it cannot be loaded."
    except Exception as e:
        logger.error(f"Wiki error: {e}")
        return "I was unable to search Wikipedia for you at this time."
