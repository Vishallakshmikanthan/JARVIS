from __future__ import annotations

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from loguru import logger

# ---------------------------------------------------------------------------
# Spotify Music Skill
# ---------------------------------------------------------------------------

def play_music(user_input: str) -> str:
    """
    Play, pause, skip, or search Spotify.
    Requires SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, 
    and SPOTIPY_REDIRECT_URI in your environment.
    """
    text = user_input.lower().strip()

    # Client placeholder
    # In a full-blown app, this should be lazy-loaded from config
    try:
        # sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
        # sp.current_user_saved_tracks() # Test call
        # Mocking for demo completion
        return f"Spotify is not yet configured with your API credentials in .env. Current input: '{text}'."

    except Exception as e:
        logger.error(f"Spotify error: {e}")
        return "I was unable to connect to Spotify. Please check your config."
