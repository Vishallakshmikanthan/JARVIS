from __future__ import annotations

"""
skills/assistant_control.py
Dynamically enable or disable skill modules at runtime.
"""

import importlib
import re
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Module registry  –  tracks enabled / disabled state for every known skill
# ---------------------------------------------------------------------------

_KNOWN_MODULES: list[str] = [
    "weather",
    "search",
    "news",
    "wiki",
    "system",
    "spotify",
    "whatsapp",
    "calculator",
    "timer",
    "jokes",
    "automation",
    "file_system",
    "context_awareness",
    "assistant_control",
]

# module name → enabled flag
_module_state: dict[str, bool] = {m: True for m in _KNOWN_MODULES}

# module name → cached Python module object (populated on first enable/use)
_module_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_name(text: str) -> str | None:
    """Return the canonical module name if mentioned in *text*."""
    text_clean = text.lower()
    for name in _KNOWN_MODULES:
        # Match "timer", "timer module", "the timer skill" etc.
        if re.search(rf"\b{re.escape(name)}\b", text_clean):
            return name
    return None


def _load_module(name: str) -> Any | None:
    """Import and cache a skill module; return it or None on failure."""
    if name in _module_cache:
        return _module_cache[name]
    try:
        mod = importlib.import_module(f"skills.{name}")
        _module_cache[name] = mod
        return mod
    except ImportError as exc:
        logger.error(f"Cannot import skills.{name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API (used by other modules)
# ---------------------------------------------------------------------------

def is_enabled(module_name: str) -> bool:
    """Return True if *module_name* is currently enabled."""
    return _module_state.get(module_name, False)


def get_module(module_name: str) -> Any | None:
    """Return the loaded module if enabled, otherwise None."""
    if not is_enabled(module_name):
        logger.warning(f"Module '{module_name}' is disabled.")
        return None
    return _load_module(module_name)


def list_modules() -> dict[str, bool]:
    """Return a snapshot of the current module-state registry."""
    return dict(_module_state)


# ---------------------------------------------------------------------------
# Public skill function
# ---------------------------------------------------------------------------

def manage_module(user_input: str) -> str:
    """
    Enable, disable, or list skill modules via natural language.

    Examples::

        "disable weather module"
        "enable the timer skill"
        "enable automation"
        "list all modules"
        "show module status"
        "what modules are enabled"
    """
    text = user_input.lower().strip()

    # --- List / status ---
    if re.search(r"\b(list|show|status|what|which)\b", text):
        enabled  = [m for m, on in _module_state.items() if on]
        disabled = [m for m, on in _module_state.items() if not on]
        lines = ["Module status:"]
        lines.append(f"  Enabled ({len(enabled)}):  {', '.join(enabled) or 'none'}")
        lines.append(f"  Disabled ({len(disabled)}): {', '.join(disabled) or 'none'}")
        return "\n".join(lines)

    # --- Enable ---
    if re.search(r"\b(enable|activate|turn\s+on|load)\b", text):
        name = _resolve_name(text)
        if not name:
            return "Which module would you like to enable? E.g. 'enable weather module'."
        if _module_state.get(name):
            return f"Module '{name}' is already enabled."
        _module_state[name] = True
        _load_module(name)          # warm up the import
        logger.info(f"Module '{name}' enabled.")
        return f"Module '{name}' has been enabled."

    # --- Disable ---
    if re.search(r"\b(disable|deactivate|turn\s+off|unload)\b", text):
        name = _resolve_name(text)
        if not name:
            return "Which module would you like to disable? E.g. 'disable search module'."
        if name == "assistant_control":
            return "I can't disable myself — that would be rather recursive."
        if not _module_state.get(name, True):
            return f"Module '{name}' is already disabled."
        _module_state[name] = False
        logger.info(f"Module '{name}' disabled.")
        return f"Module '{name}' has been disabled."

    return (
        "I can enable, disable, or list modules. "
        "Try: 'enable weather', 'disable search', or 'list modules'."
    )
