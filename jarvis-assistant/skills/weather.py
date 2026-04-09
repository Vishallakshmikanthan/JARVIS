from __future__ import annotations

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Open-Meteo Weather Skill
# ---------------------------------------------------------------------------

def get_weather(user_input: str) -> str:
    """
    Get current weather for a city.
    Uses geocoding to find coordinates first.
    """
    # Simple extraction (mock - assuming brain or config provides location)
    # In a full impl, we'd use regex or LLM to extract the city
    city = "New York" 
    if "in " in user_input.lower():
        city = user_input.lower().split("in ")[-1].strip("? .")

    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if not geo_res.get("results"):
            return f"I couldn't find location data for {city}."
        
        loc = geo_res["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        full_name = f"{loc['name']}, {loc.get('country', '')}"

        # 2. Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_res = requests.get(weather_url, timeout=5).json()
        
        curr = w_res["current_weather"]
        temp = curr["temperature"]
        wind = curr["windspeed"]
        
        return f"Currently in {full_name}, it's {temp}°C with a wind speed of {wind} km/h."

    except Exception as e:
        logger.error(f"Weather error: {e}")
        return "I encountered an error retrieving the weather data."
