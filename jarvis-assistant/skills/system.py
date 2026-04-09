from __future__ import annotations

import psutil
import pyautogui
from loguru import logger

# ---------------------------------------------------------------------------
# System Control Skill
# ---------------------------------------------------------------------------

def system_command(user_input: str) -> str:
    """
    Handle system commands (Battery, CPU, Memory, Screenshot, Volume).
    """
    text = user_input.lower().strip()

    # --- Battery status ---
    if "battery" in text:
        bat = psutil.sensors_battery()
        if bat:
            pcent = bat.percentage
            is_plugged = "plugged in" if bat.power_plugged else "not plugged in"
            return f"Your battery is at {pcent}% and it is {is_plugged}."
        return "I couldn't retrieve battery status."

    # --- CPU and RAM status ---
    if "cpu" in text or "processor" in text:
        cpu_usage = psutil.cpu_percent(interval=1)
        return f"Current CPU usage is at {cpu_usage}%."

    if "ram" in text or "memory" in text:
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        return f"Your RAM usage is at {ram_percent}%."

    # --- Screenshot ---
    if "screenshot" in text or "take a picture" in text:
        try:
            # Assume screenshot path is defined or in current directory
            pyautogui.screenshot("screenshot.png")
            return "Screenshot saved as 'screenshot.png' in the root directory."
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return "I could not take a screenshot at this time."

    return "I couldn't find a matching system command." 
