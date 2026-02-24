"""
Cursor Composer Control tool — standalone module.

Sends prompts to Cursor's Composer interface via clipboard + AppleScript.
Only available on macOS.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import Any, Dict

from tools.models import CursorComposerArgs

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# AppleScript helper
# ---------------------------------------------------------------------------


def _send_to_cursor_composer(prompt: str) -> bool:
    """
    Send a prompt to Cursor's Composer via clipboard and AppleScript.

    1. Copies the prompt to the clipboard with pbcopy
    2. Activates Cursor with AppleScript, opens Composer (Cmd+Shift+I), pastes
    3. Does NOT press Enter — user reviews first
    """
    try:
        subprocess.run(
            ["pbcopy"],
            input=prompt.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=5,
        )

        script = '''
-- Basic activation (works with all apps including Electron)
tell application "Cursor" to activate

-- Give macOS time to complete the app switch
delay 0.5

-- Use System Events for all window manipulation (works with Electron apps)
tell application "System Events"
    tell process "Cursor"
        set frontmost to true
        
        try
            if (count of windows) > 0 then
                perform action "AXRaise" of window 1
            end if
        end try
    end tell
end tell

delay 0.3

-- Verify Cursor is now the frontmost app and retry if needed
repeat 3 times
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    
    if frontApp is "Cursor" then
        exit repeat
    else
        tell application "Cursor" to activate
        delay 0.2
        tell application "System Events"
            tell process "Cursor"
                set frontmost to true
                try
                    if (count of windows) > 0 then
                        perform action "AXRaise" of window 1
                    end if
                end try
            end tell
        end tell
        delay 0.3
    end if
end repeat

-- Final check - if still not frontmost, abort with error
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
end tell

if frontApp is not "Cursor" then
    return "error: Could not bring Cursor to front. Current app: " & frontApp
end if

tell application "System Events"
    tell process "Cursor"
        -- Press Escape first to dismiss any open menus/dropdowns/dialogs
        key code 53
        delay 0.15
        
        -- Press Escape again to ensure we're in a clean state
        key code 53
        delay 0.15
        
        -- Open Composer with Cmd+Shift+I (opens NEW pane, doesn't toggle)
        keystroke "i" using {command down, shift down}
        delay 0.6
        
        -- Wait a bit more for the Composer input to be ready
        delay 0.3
        
        -- Paste from clipboard
        keystroke "v" using command down
        delay 0.1
    end tell
end tell

return "success"
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["osascript", script_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )

            output = result.stdout.strip().lower()

            if "error:" in output:
                logger.error("AppleScript activation failed: %s", result.stdout.strip())
                return False

            if "success" in output:
                logger.info("Prompt sent to Cursor Composer successfully")
                return True
            else:
                logger.warning("AppleScript returned unexpected result: %s", result.stdout)
                return True

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except subprocess.CalledProcessError as e:
        logger.error("Failed to send to Cursor: %s", getattr(e, "stderr", e))
        return False
    except subprocess.TimeoutExpired:
        logger.error("Cursor control timed out")
        return False
    except Exception as e:
        logger.error("Unexpected error sending to Cursor: %s", e)
        return False


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(args: CursorComposerArgs) -> dict:
    """Execute the tool to send a prompt to Cursor's Composer."""
    try:
        if not IS_MACOS:
            return {
                "success": False,
                "error": "This tool is only available on macOS",
                "platform": platform.system(),
            }

        prompt = args.prompt.strip()

        if not prompt:
            return {"success": False, "error": "Prompt cannot be empty"}

        logger.debug("Executing Tool: Cursor Composer -- Sending prompt (%d chars)", len(prompt))

        success = _send_to_cursor_composer(prompt)

        if success:
            return {
                "success": True,
                "prompt_sent": prompt,
                "method": "Clipboard + AppleScript",
                "info": "Prompt sent to Cursor Composer. Review and press Enter to execute.",
            }
        else:
            return {
                "success": False,
                "error": "Failed to send prompt to Cursor Composer",
                "suggestion": "Ensure Cursor is installed and running",
            }

    except Exception as e:
        logger.error("Error executing Cursor Composer tool: %s", e)
        return {"success": False, "error": str(e)}
