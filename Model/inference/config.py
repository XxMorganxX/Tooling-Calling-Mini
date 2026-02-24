"""
Configuration for the inference server and its tools.
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Google Calendar
# =============================================================================

# Service Account credentials (recommended - never expires, no user interaction)
# The same service account can access multiple calendars if each is shared with it
_DEFAULT_SA_FILE = str(_PROJECT_ROOT / "creds" / "homeassist-google-service.json")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", _DEFAULT_SA_FILE)

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# All calendars use the same service account - just share each calendar with:
#   calendar-homeassist@homeassist-465018.iam.gserviceaccount.com
CALENDAR_USERS = {
    "morgan_personal": {
        "calendar_id": "morgannstuart@gmail.com",
        "display_name": "Personal",
        "aliases": ["personal", "main", "gmail", "my calendar"],
    },
    "morgan_school": {
        "calendar_id": "mns66@cornell.edu",
        "display_name": "School",
        "aliases": ["school", "cornell", "university", "class", "classes"],
    },
    "Gen_AI": {
        "calendar_id": "c_a9971ca39f405c3c8c855332f1d5bc8721378fef7845b1e3fc7b62601e0911bc@group.calendar.google.com",
        "display_name": "Gen AI Class",
        "aliases": ["genai", "gen ai", "gen_ai", "ai", "ai class", "generative ai"],
    },
    "homeassist": {
        "calendar_id": "bd7409eb309d624908ee53c2adf02cfc3d087e50dd1139909df8d13e2b8bb8e4@group.calendar.google.com",
        "display_name": "HomeAssist",
        "aliases": ["assistant", "home assistant", "homeassist", "reminders"],
    },
}


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
