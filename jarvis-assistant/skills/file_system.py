from __future__ import annotations

"""
skills/file_system.py
Read, write, list, and delete files via natural-language commands.
All paths are resolved relative to a configurable sandbox directory to
prevent access outside the workspace.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Sandbox root  –  all file operations are confined here
# ---------------------------------------------------------------------------

try:
    from config import config
    _SANDBOX: Path = Path(config.DB_PATH).parent / "workspace"
except Exception:
    _SANDBOX: Path = Path.home() / "jarvis_workspace"

_SANDBOX.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _safe_path(relative: str) -> Optional[Path]:
    """
    Resolve *relative* to an absolute path inside the sandbox.
    Returns None if the resolved path would escape the sandbox.
    """
    try:
        resolved = (_SANDBOX / relative).resolve()
        resolved.relative_to(_SANDBOX.resolve())   # raises if outside
        return resolved
    except (ValueError, OSError):
        logger.warning(f"Path escape attempt blocked: '{relative}'")
        return None


def _extract_filename(text: str) -> Optional[str]:
    """Extract a quoted or bare filename / path from *text*."""
    # Quoted first
    q = re.search(r"['\"]([^'\"]+)['\"]", text)
    if q:
        return q.group(1).strip()
    # Bare word that looks like a file (contains a dot or slash)
    bare = re.search(r"(\S+\.\w+)", text)
    if bare:
        return bare.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _read_file(filename: str) -> str:
    path = _safe_path(filename)
    if path is None:
        return "Access denied: path is outside the workspace."
    if not path.exists():
        return f"File '{filename}' not found in the workspace."
    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        preview = "\n".join(lines[:40])
        suffix = f"\n… ({len(lines) - 40} more lines)" if len(lines) > 40 else ""
        return f"Contents of '{filename}':\n{preview}{suffix}"
    except Exception as e:
        logger.error(f"Read error: {e}")
        return f"I couldn't read '{filename}': {e}"


def _write_file(filename: str, content: str, append: bool = False) -> str:
    path = _safe_path(filename)
    if path is None:
        return "Access denied: path is outside the workspace."
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as fh:
            fh.write(content)
        action = "appended to" if append else "written to"
        logger.info(f"File {action}: {path}")
        return f"Content successfully {action} '{filename}'."
    except Exception as e:
        logger.error(f"Write error: {e}")
        return f"I couldn't write to '{filename}': {e}"


def _delete_file(filename: str) -> str:
    path = _safe_path(filename)
    if path is None:
        return "Access denied: path is outside the workspace."
    if not path.exists():
        return f"'{filename}' does not exist."
    try:
        if path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Directory deleted: {path}")
            return f"Directory '{filename}' and all its contents have been deleted."
        path.unlink()
        logger.info(f"File deleted: {path}")
        return f"File '{filename}' has been deleted."
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return f"I couldn't delete '{filename}': {e}"


def _list_dir(subdir: str = "") -> str:
    path = _safe_path(subdir) if subdir else _SANDBOX
    if path is None:
        return "Access denied."
    if not path.exists():
        return f"Directory '{subdir}' does not exist in the workspace."
    try:
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not items:
            return f"The directory '{subdir or 'workspace'}' is empty."
        lines = [f"Contents of '{subdir or 'workspace'}'  ({len(items)} items):"]
        for item in items:
            prefix = "📁" if item.is_dir() else "📄"
            size   = f"  ({item.stat().st_size} B)" if item.is_file() else ""
            lines.append(f"  {prefix} {item.name}{size}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"List error: {e}")
        return f"I couldn't list the directory: {e}"


# ---------------------------------------------------------------------------
# Public skill function
# ---------------------------------------------------------------------------

def file_command(user_input: str) -> str:
    """
    Perform file system operations via natural language.

    Examples::

        "read notes.txt"
        "open my file 'report.md'"
        "write 'Hello World' to hello.txt"
        "append 'new line' to log.txt"
        "delete old_file.txt"
        "list files"
        "show workspace"
        "what files do I have"
    """
    text = user_input.strip()
    lower = text.lower()

    # --- List / show directory ---
    if re.search(r"\b(list|show|workspace|what files|directory|folder)\b", lower):
        # Optional subfolder
        sub_match = re.search(r"(?:in|inside|folder)\s+(['\"]?)(\S+)\1", lower)
        subdir = sub_match.group(2) if sub_match else ""
        return _list_dir(subdir)

    # --- Read / open ---
    if re.search(r"\b(read|open|show\s+me|display|view|get)\b", lower):
        filename = _extract_filename(text)
        if not filename:
            return "Which file would you like me to read? E.g. 'read notes.txt'."
        return _read_file(filename)

    # --- Append ---
    if re.search(r"\b(append|add\s+to|insert\s+into)\b", lower):
        filename = _extract_filename(text)
        if not filename:
            return "Please specify a file to append to."
        # Extract content between quotes preferring the second quoted group
        parts = re.findall(r"['\"]([^'\"]+)['\"]", text)
        content = parts[0] if parts else re.sub(
            r"append|add\s+to|insert\s+into|to\s+\S+", "", lower
        ).strip()
        return _write_file(filename, content + "\n", append=True)

    # --- Write / create / save ---
    if re.search(r"\b(write|create|save|make\s+a\s+file)\b", lower):
        filename = _extract_filename(text)
        if not filename:
            return "Please specify a filename. E.g. 'write 'Hello' to notes.txt'."
        # Content is the first quoted string; filename is the last quoted string
        parts = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if len(parts) >= 2:
            content, filename = parts[0], parts[1]
        elif len(parts) == 1:
            content = parts[0]
        else:
            content = re.sub(
                r"write|create|save|make\s+a\s+file|to\s+\S+", "", lower
            ).strip()
        return _write_file(filename, content)

    # --- Delete / remove ---
    if re.search(r"\b(delete|remove|erase|wipe)\b", lower):
        filename = _extract_filename(text)
        if not filename:
            return "Which file would you like to delete?"
        return _delete_file(filename)

    return (
        "I can read, write, append, delete, or list files in the workspace. "
        "Try: 'read notes.txt', 'write 'Hello' to hello.txt', or 'list files'."
    )
