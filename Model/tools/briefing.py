"""
Briefing Management Tool – standalone module.

Ported from MCP BaseTool class to module-level functions.
Creates, lists, and dismisses briefing announcements stored in Supabase.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from tools.models import BriefingArgs

try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

TABLE_NAME = "briefing_announcements"

_supabase_client: Optional[Any] = None
_supabase_available = False

# User config from env
_users_env = os.getenv("NOTIFICATION_USERS", "Morgan")
_configured_users = [u.strip() for u in _users_env.split(",") if u.strip()]
_default_user = os.getenv("DEFAULT_NOTIFICATION_USER", _configured_users[0] if _configured_users else "Morgan")

# Timezone
_DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")
_tz = ZoneInfo(_DEFAULT_TIMEZONE)


def _init_supabase() -> None:
    global _supabase_client, _supabase_available

    if not SUPABASE_AVAILABLE:
        logger.debug("Supabase package not installed")
        return

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.debug("SUPABASE_URL or SUPABASE_KEY not set")
        return

    try:
        _supabase_client = create_client(url, key)
        _supabase_available = True
        logger.debug("Supabase client initialized for briefings")
    except Exception as e:
        logger.warning("Failed to initialize Supabase client: %s", e)


_init_supabase()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(args: BriefingArgs) -> dict:
    """Execute the briefing tool."""
    try:
        params = args.model_dump()
        action = params.get("action", "create")
        user = params.get("user", _default_user)
        user = _normalize_user(user)

        if action == "create":
            return _create_briefing(params, user)
        elif action == "list":
            return _list_briefings(params, user)
        elif action == "dismiss":
            return _dismiss_briefing(params, user)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: create, list, dismiss",
            }

    except Exception as e:
        logger.error("Error executing briefing tool: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# User normalization
# ---------------------------------------------------------------------------


def _normalize_user(user: str) -> str:
    if not user:
        return _default_user

    user_lower = user.lower().strip()

    for configured_user in _configured_users:
        if user_lower == configured_user.lower():
            return configured_user

    if user_lower in {"me", "my", "default"}:
        return _default_user

    return user.title()


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _parse_datetime(dt_string: str) -> datetime:
    if not dt_string:
        raise ValueError("Datetime string is empty")

    if "T" in dt_string:
        dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz)
    else:
        dt = datetime.strptime(dt_string, "%Y-%m-%d")
        dt = dt.replace(tzinfo=_tz)

    return dt


def _format_time_until(minutes: int) -> str:
    if minutes >= 1440:
        days = minutes // 1440
        hours = (minutes % 1440) // 60
        if hours > 0:
            return f"{days} day{'s' if days > 1 else ''} and {hours} hour{'s' if hours > 1 else ''}"
        return f"{days} day{'s' if days > 1 else ''}"
    elif minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        if mins > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} and {mins} minute{'s' if mins > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''}"
    else:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _create_briefing(params: Dict[str, Any], user: str) -> Dict[str, Any]:
    message = params.get("message")

    if not message:
        return {
            "success": False,
            "error": "Message is required to create a briefing. Please provide a message for the announcement (e.g., 'Call Merrick').",
        }

    event_time = params.get("event_time")
    remind_at = params.get("remind_at")
    remind_before_minutes = params.get("remind_before_minutes")
    priority = params.get("priority", "normal")

    if not remind_at and not (event_time and remind_before_minutes):
        return {
            "success": False,
            "error": (
                "Timing is required. Please provide either:\n"
                "1. 'remind_at' - exact time to remind (e.g., '2026-01-13T12:30:00'), OR\n"
                "2. 'event_time' AND 'remind_before_minutes' - to remind X minutes before the event\n\n"
                "Examples:\n"
                "- 'Remind me at 12:30' -> remind_at='2026-01-13T12:30:00'\n"
                "- 'Remind me 30 minutes before' -> event_time='2026-01-13T13:00:00', remind_before_minutes=30"
            ),
        }

    event_dt = None
    active_from_dt = None

    try:
        if remind_at:
            active_from_dt = _parse_datetime(remind_at)
            if event_time:
                event_dt = _parse_datetime(event_time)
        else:
            event_dt = _parse_datetime(event_time)
            active_from_dt = event_dt - timedelta(minutes=remind_before_minutes)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid datetime format: {e}. Use ISO 8601 format like '2026-01-13T13:00:00'",
        }

    now = datetime.now(timezone.utc)
    if active_from_dt.astimezone(timezone.utc) < now:
        return {
            "success": False,
            "error": f"Reminder time {active_from_dt.isoformat()} is in the past. Please provide a future time.",
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    briefing_id = f"manual_{timestamp}_{uuid.uuid4().hex[:8]}"

    if event_dt:
        display_message = f"{message}. Your event is in {{{{TIME_UNTIL_EVENT}}}}."
    else:
        display_message = message

    content = {
        "message": display_message,
        "llm_instructions": f"Remind the user: {message}. Be concise and natural.",
        "active_from": active_from_dt.isoformat(),
        "meta": {
            "source": "assistant_created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_message": message,
        },
    }

    if event_dt:
        content["meta"]["event_datetime_iso"] = event_dt.isoformat()
        content["meta"]["remind_before_minutes"] = remind_before_minutes

    record = {
        "id": briefing_id,
        "user_id": user,
        "content": content,
        "priority": priority,
        "status": "pending",
    }

    if not _supabase_available:
        return {"success": False, "error": "Supabase is not available. Cannot store briefing."}

    try:
        record_for_db = record.copy()
        record_for_db["content"] = json.dumps(content)
        _supabase_client.table(TABLE_NAME).insert(record_for_db).execute()

        response: Dict[str, Any] = {
            "success": True,
            "action": "create",
            "briefing_id": briefing_id,
            "user": user,
            "message": message,
            "priority": priority,
            "status": "pending",
            "remind_at": active_from_dt.isoformat(),
        }

        if event_dt:
            response["event_time"] = event_dt.isoformat()
            response["remind_before_minutes"] = remind_before_minutes
            response["note"] = (
                f"Reminder set for {_format_time_until(remind_before_minutes)} before your event "
                f"at {event_dt.strftime('%I:%M %p on %b %d')}"
            )
        else:
            response["note"] = f"Reminder set for {active_from_dt.strftime('%I:%M %p on %b %d')}"

        return response

    except Exception as e:
        logger.error("Failed to store briefing: %s", e)
        return {"success": False, "error": f"Failed to store briefing: {str(e)}"}


def _list_briefings(params: Dict[str, Any], user: str) -> Dict[str, Any]:
    limit = params.get("limit", 10)

    if not _supabase_available:
        return {"success": False, "error": "Supabase is not available. Cannot retrieve briefings."}

    try:
        response = (
            _supabase_client.table(TABLE_NAME)
            .select("id, content, priority, status, created_at")
            .eq("user_id", user)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        briefings = []
        for row in response.data or []:
            content = row.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {}

            meta = content.get("meta", {})

            briefings.append({
                "id": row.get("id"),
                "message": meta.get("original_message") or content.get("message", ""),
                "priority": row.get("priority", "normal"),
                "created_at": row.get("created_at"),
                "remind_at": content.get("active_from"),
                "event_time": meta.get("event_datetime_iso"),
                "remind_before_minutes": meta.get("remind_before_minutes"),
                "source": meta.get("source", "unknown"),
            })

        return {
            "success": True,
            "action": "list",
            "user": user,
            "count": len(briefings),
            "briefings": briefings,
        }

    except Exception as e:
        logger.error("Failed to list briefings: %s", e)
        return {"success": False, "error": f"Failed to list briefings: {str(e)}"}


def _dismiss_briefing(params: Dict[str, Any], user: str) -> Dict[str, Any]:
    briefing_id = params.get("briefing_id")

    if not briefing_id:
        return {
            "success": False,
            "error": "briefing_id is required to dismiss a briefing. Use action='list' to see available briefings.",
        }

    if not _supabase_available:
        return {"success": False, "error": "Supabase is not available. Cannot dismiss briefing."}

    try:
        now = datetime.now(timezone.utc).isoformat()

        _supabase_client.table(TABLE_NAME).update({
            "status": "dismissed",
            "dismissed_at": now,
        }).eq("id", briefing_id).eq("user_id", user).execute()

        return {
            "success": True,
            "action": "dismiss",
            "briefing_id": briefing_id,
            "user": user,
            "status": "dismissed",
            "dismissed_at": now,
        }

    except Exception as e:
        logger.error("Failed to dismiss briefing: %s", e)
        return {"success": False, "error": f"Failed to dismiss briefing: {str(e)}"}
