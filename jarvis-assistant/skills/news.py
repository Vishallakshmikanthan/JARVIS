from __future__ import annotations

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# News Skill (NewsAPI)
# ---------------------------------------------------------------------------

def get_news(user_input: str) -> str:
    """
    Get top headlines or search news by topic.
    Example: "What's the news in technology"
    """
    topic = "general"
    if "about" in user_input.lower():
        topic = user_input.lower().split("about")[-1].strip("? .")
    elif "in " in user_input.lower():
        topic = user_input.lower().split("in ")[-1].strip("? .")

    # This requires NewsAPI Key in config, mock logic follows
    try:
        # url = f"https://newsapi.org/v2/top-headlines?category={topic}&apiKey=YOUR_KEY"
        # Mocking for demo completion
        return f"I see headlines about {topic} currently. (NewsAPI requires an API key in your .env)."

    except Exception as e:
        logger.error(f"News error: {e}")
        return "I could not retrieve the news at this moment."
