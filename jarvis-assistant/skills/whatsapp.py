from __future__ import annotations

import pywhatkit
from loguru import logger

# ---------------------------------------------------------------------------
# WhatsApp Messaging Skill (PyWhatKit)
# ---------------------------------------------------------------------------

def send_whatsapp(user_input: str) -> str:
    """
    Send WhatsApp messages via PyWhatKit.
    Example: "send Hi to +1234567890" or "whatsapp message to ..."
    """
    text = user_input.lower().strip()

    # Regex or splitting to find number and text
    # Very simple extraction for demo:
    # "send X to +N"
    try:
        if "to " in text:
            parts = text.split("to ")
            number = parts[-1].strip("? .")
            msg = parts[0].replace("send", "").replace("whatsapp", "").strip()
            
            # Send instantly (requires WhatsApp Web logged in)
            pywhatkit.sendwhatmsg_instantly(number, msg, wait_time=15)
            return f"Opening WhatsApp Web to send '{msg}' to {number}."
        
        return "I couldn't parse a message and number. Try: 'Send Hello to +1234567890'."

    except Exception as e:
        logger.error(f"WhatsApp error: {e}")
        return "I encountered an error trying to send the WhatsApp message."
