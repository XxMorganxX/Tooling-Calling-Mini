"""
Send SMS/iMessage tool — standalone module.

Sends text messages via iMessage on macOS.
Only available on macOS — will error on other platforms.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import Any, Dict

from tools.models import SendSmsArgs

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"

_default_phone_number: str = os.environ.get("SMS_PHONE_NUMBER", "")


# ---------------------------------------------------------------------------
# iMessage helper
# ---------------------------------------------------------------------------


def _send_imessage(message: str, phone_number: str) -> bool:
    """Send an iMessage via AppleScript."""
    escaped_message = message.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{phone_number}" of targetService
    send "{escaped_message}" to targetBuddy
    delay 1
    quit
end tell
'''

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            subprocess.run(
                ["osascript", script_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.info("Message sent to %s", phone_number)
            return True
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except subprocess.CalledProcessError as e:
        logger.error("Failed to send message: %s", e.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Message sending timed out")
        return False
    except Exception as e:
        logger.error("Unexpected error sending message: %s", e)
        return False


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: SendSmsArgs) -> dict:
    """Execute the SMS tool to send a message."""
    try:
        if not IS_MACOS:
            return {
                "success": False,
                "error": "This tool is only available on macOS",
                "platform": platform.system(),
            }

        message = args.message.strip()
        phone_number = _default_phone_number

        logger.debug("Executing Tool: SMS -- Sending to %s...", phone_number[:6] if phone_number else "?")

        if not message:
            return {"success": False, "error": "Message cannot be empty"}

        if not phone_number:
            return {"success": False, "error": "SMS tool not configured with a phone number"}

        if not phone_number.startswith("+"):
            return {
                "success": False,
                "error": "Configured phone number invalid (must include country code)",
            }

        success = _send_imessage(message, phone_number)

        if success:
            return {
                "success": True,
                "message_sent": message,
                "recipient": phone_number,
                "method": "iMessage",
                "info": "Message sent successfully via iMessage",
            }
        else:
            return {
                "success": False,
                "error": "Failed to send message via iMessage",
                "recipient": phone_number,
                "suggestion": "Ensure Messages app is configured and recipient uses iMessage",
            }

    except Exception as e:
        logger.error("Error executing SMS tool: %s", e)
        return {
            "success": False,
            "error": str(e),
            "recipient": "unknown",
        }
