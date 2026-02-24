"""
Clipboard Read tool — standalone module.

Reads the current system clipboard contents, allowing the AI
to work with text the user has copied.
"""

import logging
import platform
import subprocess
from typing import Any, Dict

from tools.models import ReadClipboardArgs

logger = logging.getLogger(__name__)

_is_macos = platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# macOS clipboard reader
# ---------------------------------------------------------------------------


def _read_clipboard_macos() -> str:
    """Read clipboard contents on macOS using pbpaste."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError("Clipboard read timed out")
    except FileNotFoundError:
        raise RuntimeError("pbpaste command not found - are you on macOS?")
    except Exception as e:
        raise RuntimeError(f"Failed to read clipboard: {e}")


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: ReadClipboardArgs) -> dict:
    """Execute the clipboard read tool."""
    try:
        logger.debug("Executing Tool: Clipboard -- max_length=%s", args.max_length)

        if not _is_macos:
            return {
                "success": False,
                "error": "Clipboard reading is only supported on macOS",
                "platform": platform.system(),
            }

        max_length = args.max_length if args.max_length is not None else 8000

        content = _read_clipboard_macos()

        if not content:
            return {
                "success": True,
                "content": "",
                "is_empty": True,
                "message": "The clipboard is empty - nothing has been copied.",
            }

        original_length = len(content)
        was_truncated = original_length > max_length

        if was_truncated:
            content = content[:max_length]
            truncation_message = f"Content truncated from {original_length} to {max_length} characters."
        else:
            truncation_message = None

        return {
            "success": True,
            "content": content,
            "is_empty": False,
            "character_count": len(content),
            "original_length": original_length,
            "was_truncated": was_truncated,
            "truncation_message": truncation_message,
        }

    except Exception as e:
        logger.error("Failed to read clipboard: %s", e)
        return {
            "success": False,
            "error": f"Failed to read clipboard: {str(e)}",
        }
