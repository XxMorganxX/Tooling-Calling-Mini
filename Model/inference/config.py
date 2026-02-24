"""
Configuration for the inference server and its tools.

Calendar user data is loaded from ``calendar_users.json`` (gitignored) so that
personal email addresses and calendar IDs stay out of version control.  Copy
``calendar_users.example.json`` → ``calendar_users.json`` and fill in your values.
"""

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = Path(__file__).parent

# =============================================================================
# Google Calendar
# =============================================================================

_DEFAULT_SA_FILE = str(_PROJECT_ROOT / "creds" / "homeassist-google-service.json")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", _DEFAULT_SA_FILE)

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

_CALENDAR_USERS_FILE = _CONFIG_DIR / "calendar_users.json"

if _CALENDAR_USERS_FILE.exists():
    with open(_CALENDAR_USERS_FILE, encoding="utf-8") as _f:
        CALENDAR_USERS: dict = json.load(_f)
else:
    CALENDAR_USERS = {}


def get_calendar_alias_map() -> dict:
    """Build alias -> canonical calendar key mapping."""
    alias_map: dict[str, str] = {}
    for key, cfg in CALENDAR_USERS.items():
        alias_map[key.lower()] = key
        if cfg.get("display_name"):
            alias_map[cfg["display_name"].lower()] = key
        for alias in cfg.get("aliases", []):
            alias_map[alias.lower()] = key
    return alias_map


CALENDAR_ALIAS_MAP = get_calendar_alias_map()

DEFAULT_TIME_ZONE = os.environ.get("DEFAULT_TIME_ZONE", "America/New_York")
