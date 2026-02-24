"""
Calendar Data Tool – standalone module.

Ported from MCP BaseTool class to module-level functions.
Provides comprehensive Google Calendar access with command normalization,
natural-language date parsing, and multi-calendar read/write support.
"""

import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from inference.config import CALENDAR_USERS, CALENDAR_ALIAS_MAP

try:
    from tools.models import CalendarDataArgs
except ImportError:
    from inference.tools.models import CalendarDataArgs

try:
    from clients.calendar_client import CalendarComponent
except ImportError:
    CalendarComponent = None

try:
    from googleapiclient.discovery import build as _build_service
except ImportError:
    _build_service = None

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    dateparser = None
    DATEPARSER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_calendar_instances: Dict[str, Any] = {}
_CALENDAR_USERS = CALENDAR_USERS
_CALENDAR_ALIAS_MAP = CALENDAR_ALIAS_MAP

_available_calendars: Optional[List[str]] = None
_available_actions = ["read", "create_event"]
_available_read_types = ["next_events", "day_summary", "week_summary", "specific_date"]

_shared_creds = None
_shared_creds_lock = threading.Lock()


def _get_available_calendars() -> List[str]:
    global _available_calendars
    if _available_calendars is None:
        if _CALENDAR_USERS:
            _available_calendars = list(_CALENDAR_USERS.keys())
        else:
            _available_calendars = []
    return _available_calendars


def _get_shared_creds():
    """Get authenticated service-account credentials, lazily initialized once."""
    global _shared_creds
    if _shared_creds is not None:
        return _shared_creds
    with _shared_creds_lock:
        if _shared_creds is not None:
            return _shared_creds
        first_user = _get_available_calendars()[0] if _get_available_calendars() else None
        if not first_user:
            return None
        instance = _get_calendar_instance(first_user)
        if instance and instance.creds:
            _shared_creds = instance.creds
        return _shared_creds


def _fetch_events_for_calendar(
    creds,
    cal_name: str,
    cal_id: str,
    time_min: Optional[str],
    time_max: Optional[str],
    max_results: int,
) -> Tuple[str, List[Dict], Optional[str]]:
    """Fetch events for one calendar ID. Runs in a worker thread with its own
    service object (httplib2 is not thread-safe so each thread builds one)."""
    try:
        service = _build_service("calendar", "v3", credentials=creds, cache_discovery=False)
        result = service.events().list(
            calendarId=cal_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        for e in events:
            e["source_calendar"] = cal_name
        return cal_name, events, None
    except Exception as exc:
        return cal_name, [], str(exc)


# ---------------------------------------------------------------------------
# Calendar alias resolution
# ---------------------------------------------------------------------------


def resolve_calendar_alias(name: str) -> str:
    """Resolve a calendar name/alias to the canonical calendar key."""
    if not name:
        return name
    if name in _CALENDAR_USERS:
        return name
    return _CALENDAR_ALIAS_MAP.get(name.lower(), name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(args: CalendarDataArgs) -> dict:
    """Execute calendar operations."""
    try:
        params = args.model_dump()
        commands = params.get("commands", [])
        commands = [_normalize_command(cmd) for cmd in commands]

        if not commands:
            return {
                "success": False,
                "error": "No commands provided. You must include a 'commands' array with at least one command object.",
                "example_read": {
                    "commands": [{"read_or_write": "read", "read_type": "next_events", "calendar": "all"}]
                },
                "example_create": {
                    "commands": [{"read_or_write": "create_event", "event_title": "Meeting", "date": "2026-01-23", "start_time": "14:00", "end_time": "15:00"}]
                },
                "available_calendars": _get_available_calendars(),
                "available_actions": _available_actions,
            }

        if len(commands) > 1:
            return {
                "success": False,
                "error": "Only ONE command per tool call. Do not make multiple calendar tool calls for the same request.",
                "available_calendars": _get_available_calendars(),
            }

        write_commands = [c for c in commands if c.get("read_or_write") == "create_event"]
        if len(write_commands) > 1:
            return {
                "success": False,
                "error": "Only ONE write/create_event command allowed per call. To add an event to multiple calendars, use the 'calendars' array parameter instead of multiple commands.",
                "available_calendars": _get_available_calendars(),
            }

        validation_errors = _validate_commands(commands)
        if validation_errors:
            missing_info_errors = [e for e in validation_errors if "MISSING_INFO:" in e]
            format_errors = [e for e in validation_errors if "INVALID_FORMAT:" in e]

            if missing_info_errors:
                guidance = missing_info_errors[0].replace("Command 1: MISSING_INFO: ", "")
                return {
                    "success": False,
                    "error": f"Cannot create event: {guidance}",
                    "hint": "Ask the user to provide this information before trying again.",
                }
            if format_errors:
                details = "; ".join(
                    e.split("INVALID_FORMAT: ", 1)[-1]
                    for e in format_errors
                )
                return {
                    "success": False,
                    "error": f"Cannot create event due to invalid field format: {details}",
                }
            return {
                "success": False,
                "error": f"Calendar command validation failed: {'; '.join(validation_errors)}",
                "available_calendars": _get_available_calendars(),
                "available_actions": _available_actions,
            }

        results = []
        for i, cmd in enumerate(commands):
            try:
                result = _execute_single_command(cmd, i)
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "command_index": i,
                    "command": cmd,
                    "error": str(e),
                })

        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        calendars_used = set()
        for cmd in commands:
            cal = cmd.get("calendar") or cmd.get("user")
            if cal:
                calendars_used.add(cal)

        return {
            "success": len(failed) == 0,
            "total_commands": len(commands),
            "successful_commands": len(successful),
            "failed_commands": len(failed),
            "results": results,
            "calendars": list(calendars_used),
            "timestamp": _get_current_timestamp(),
        }
    except Exception as e:
        logger.error("Error executing calendar operations: %s", e)
        return {
            "success": False,
            "error": f"Calendar execution failed: {str(e)}",
            "total_commands": len((args.commands if args else []) or []),
            "successful_commands": 0,
            "failed_commands": len((args.commands if args else []) or []),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_commands(commands: List[Dict[str, Any]]) -> List[str]:
    errors = []
    valid_calendars = _get_available_calendars() + ["all"]

    for i, cmd in enumerate(commands):
        cmd_errors: List[str] = []

        if "read_or_write" not in cmd:
            cmd_errors.append(
                f"Missing required 'read_or_write' parameter. "
                f"Set it to one of: {_available_actions} "
                f"('read' to query events, 'create_event' to add a new event)"
            )
        elif cmd["read_or_write"] not in _available_actions:
            cmd_errors.append(
                f"Invalid read_or_write value '{cmd['read_or_write']}'. "
                f"Must be one of: {_available_actions}. "
                f"Use 'read' to query events or 'create_event' to add a new event"
            )

        calendar = cmd.get("calendar") or cmd.get("user")
        if not calendar:
            cmd_errors.append(
                f"Missing required 'calendar' parameter. "
                f"Must be one of: {valid_calendars}. "
                f"Use 'all' to read from all calendars, or a specific calendar name to target one"
            )
        elif calendar not in valid_calendars:
            cmd_errors.append(
                f"Unknown calendar '{calendar}'. "
                f"Available calendars: {valid_calendars}"
            )
        elif calendar == "all" and cmd.get("read_or_write") in ["write", "create_event"]:
            cmd_errors.append(
                f"Cannot create events on 'all' calendars. "
                f"Specify a single target calendar: {[c for c in valid_calendars if c != 'all']}"
            )

        if cmd.get("read_or_write") == "read":
            if "read_type" not in cmd:
                cmd_errors.append(
                    f"Read operations require a 'read_type' parameter. "
                    f"Must be one of: {_available_read_types}"
                )
            elif cmd["read_type"] not in _available_read_types:
                cmd_errors.append(
                    f"Invalid read_type '{cmd['read_type']}'. "
                    f"Must be one of: {_available_read_types}"
                )
            if cmd.get("read_type") == "specific_date" and "date" not in cmd:
                cmd_errors.append(
                    "read_type 'specific_date' requires a 'date' parameter in YYYY-MM-DD format"
                )

        elif cmd.get("read_or_write") == "create_event":
            missing_fields = []
            if not cmd.get("event_title"):
                missing_fields.append("event_title (the name/title of the event)")
            if not cmd.get("date"):
                missing_fields.append("date (in YYYY-MM-DD format, e.g. '2026-03-15')")
            if not cmd.get("start_time"):
                missing_fields.append("start_time (in HH:MM format, e.g. '14:00')")
            if not cmd.get("end_time"):
                missing_fields.append("end_time (in HH:MM format, e.g. '15:00')")

            if missing_fields:
                fields_str = ", ".join(missing_fields)
                cmd_errors.append(
                    f"MISSING_INFO: Cannot create event because the following required fields are missing: {fields_str}. "
                    f"Ask the user to provide them."
                )
            else:
                if not _is_valid_time_format(cmd["start_time"]):
                    cmd_errors.append(
                        f"INVALID_FORMAT: start_time '{cmd['start_time']}' is not valid. "
                        f"Must be in HH:MM 24-hour format (e.g., '14:00' for 2pm, '09:30' for 9:30am)"
                    )
                if not _is_valid_time_format(cmd["end_time"]):
                    cmd_errors.append(
                        f"INVALID_FORMAT: end_time '{cmd['end_time']}' is not valid. "
                        f"Must be in HH:MM 24-hour format (e.g., '15:00' for 3pm, '17:30' for 5:30pm)"
                    )
                if not _is_valid_date_format(cmd["date"]):
                    cmd_errors.append(
                        f"INVALID_FORMAT: date '{cmd['date']}' is not valid. "
                        f"Must be in YYYY-MM-DD format (e.g., '2026-03-15')"
                    )

        if cmd_errors:
            errors.append(f"Command {i + 1}: {'; '.join(cmd_errors)}")

    return errors


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


def _execute_single_command(cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
    try:
        calendar = cmd.get("calendar") or cmd.get("user")
        action = cmd.get("read_or_write", "read")

        if action in ["write", "create_event"]:
            return _handle_write_command(cmd, cmd_index)

        if action == "read" and (calendar == "all" or calendar is None):
            return _handle_read_all_calendars(cmd, cmd_index)

        calendar_instance = _get_calendar_instance(calendar)
        if not calendar_instance:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Failed to initialize calendar: {calendar}",
                "command": cmd,
            }

        if action == "read":
            return _handle_read_command(calendar_instance, cmd, cmd_index)
        else:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Unknown action: {action}",
                "command": cmd,
            }

    except Exception as e:
        return {"success": False, "command_index": cmd_index, "command": cmd, "error": str(e)}


def _compute_time_bounds(
    read_type: str, cmd: Dict[str, Any], include_past: bool, limit: int,
) -> Tuple[Optional[str], Optional[str], int]:
    """Compute (time_min, time_max, max_results) for a given read_type."""
    now = datetime.now(timezone.utc)

    if read_type == "next_events":
        time_min = None if include_past else now.isoformat()
        return time_min, None, limit

    if read_type in ("day_summary", "specific_date"):
        date_str = cmd.get("date", now.strftime("%Y-%m-%d"))
        target = datetime.strptime(date_str, "%Y-%m-%d")
        t_min = target.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        t_max = target.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
        return t_min.isoformat(), t_max.isoformat(), 50

    if read_type == "week_summary":
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
        t_min = start_of_week.isoformat() if include_past else now.isoformat()
        return t_min, end_of_week.isoformat(), 100

    return now.isoformat(), None, limit


def _handle_read_all_calendars(cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
    read_type = cmd.get("read_type", "next_events")
    limit = cmd.get("limit", 10)
    include_past = cmd.get("include_past_events", False)

    creds = _get_shared_creds()
    if creds is None or _build_service is None:
        return _handle_read_all_calendars_sequential(cmd, cmd_index)

    time_min, time_max, max_results = _compute_time_bounds(read_type, cmd, include_past, limit)

    calendars = [
        (name, cfg.get("calendar_id", "primary"))
        for name, cfg in _CALENDAR_USERS.items()
    ]

    all_events: List[Dict] = []
    calendars_read: List[str] = []
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=len(calendars)) as pool:
        futures = {
            pool.submit(
                _fetch_events_for_calendar,
                creds, cal_name, cal_id, time_min, time_max, max_results,
            ): cal_name
            for cal_name, cal_id in calendars
        }
        for future in as_completed(futures):
            cal_name, events, error = future.result()
            if error:
                errors.append(f"{cal_name}: {error}")
            else:
                calendars_read.append(cal_name)
                all_events.extend(events)

    def _sort_key(event: Any) -> str:
        if isinstance(event, dict):
            start = event.get("start_time") or event.get("start") or ""
            if isinstance(start, dict):
                start = start.get("dateTime") or start.get("date") or ""
            return str(start)
        return str(event)

    all_events.sort(key=_sort_key)
    limited_events = all_events[:limit]
    cleaned_events = [_clean_event_for_agent(e) for e in limited_events]

    partial_success = bool(calendars_read) and bool(errors)
    result: Dict[str, Any] = {
        "success": bool(calendars_read) or len(errors) == 0,
        "operation": "read",
        "command_index": cmd_index,
        "command": cmd,
        "read_type": read_type,
        "events": cleaned_events,
        "event_count": len(cleaned_events),
        "total_events_found": len(all_events),
        "calendars_read": calendars_read,
        "calendar": "all",
        "note": "This was a READ operation - no events were created.",
    }

    if errors:
        result["warnings"] = errors
    if partial_success:
        result["partial_success"] = True
    if read_type in ["day_summary", "specific_date"]:
        result["date"] = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))

    return result


def _handle_read_all_calendars_sequential(cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
    """Fallback: sequential reads when shared credentials are unavailable."""
    read_type = cmd.get("read_type", "next_events")
    limit = cmd.get("limit", 10)
    include_past = cmd.get("include_past_events", False)

    all_events: List[Dict] = []
    calendars_read: List[str] = []
    errors: List[str] = []

    for cal_name in _get_available_calendars():
        try:
            calendar_instance = _get_calendar_instance(cal_name)
            if not calendar_instance:
                errors.append(f"Failed to initialize {cal_name}")
                continue

            cal_config = _CALENDAR_USERS.get(cal_name, {})
            calendar_id = cal_config.get("calendar_id", "primary")

            if read_type == "next_events":
                events = calendar_instance.get_upcoming_events(
                    calendar_name=calendar_id, max_results=limit, include_past=include_past
                )
            elif read_type == "day_summary":
                target_date = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
                events = calendar_instance.get_day_events(
                    date=target_date, calendar_name=calendar_id, include_past=include_past
                )
            elif read_type == "week_summary":
                events = calendar_instance.get_week_events(
                    calendar_name=calendar_id, include_past=include_past
                )
            elif read_type == "specific_date":
                events = calendar_instance.get_day_events(
                    date=cmd["date"], calendar_name=calendar_id, include_past=include_past
                )
            else:
                events = []

            cal_error = getattr(calendar_instance, "error_message", None)
            if cal_error:
                errors.append(f"{cal_name}: {cal_error}")
                continue

            calendars_read.append(cal_name)
            for event in events:
                if isinstance(event, dict):
                    event["source_calendar"] = cal_name
            all_events.extend(events)

        except Exception as e:
            errors.append(f"{cal_name}: {str(e)}")

    def _sort_key(event: Any) -> str:
        if isinstance(event, dict):
            start = event.get("start_time") or event.get("start") or ""
            if isinstance(start, dict):
                start = start.get("dateTime") or start.get("date") or ""
            return str(start)
        return str(event)

    all_events.sort(key=_sort_key)
    limited_events = all_events[:limit]
    cleaned_events = [_clean_event_for_agent(e) for e in limited_events]

    partial_success = bool(calendars_read) and bool(errors)
    result: Dict[str, Any] = {
        "success": bool(calendars_read) or len(errors) == 0,
        "operation": "read",
        "command_index": cmd_index,
        "command": cmd,
        "read_type": read_type,
        "events": cleaned_events,
        "event_count": len(cleaned_events),
        "total_events_found": len(all_events),
        "calendars_read": calendars_read,
        "calendar": "all",
        "note": "This was a READ operation - no events were created.",
    }

    if errors:
        result["warnings"] = errors
    if partial_success:
        result["partial_success"] = True
    if read_type in ["day_summary", "specific_date"]:
        result["date"] = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))

    return result


def _handle_read_command(calendar_instance: Any, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
    read_type = cmd["read_type"]
    calendar = cmd.get("calendar") or cmd.get("user")
    limit = cmd.get("limit", 10)
    include_past = cmd.get("include_past_events", False)

    cal_config = _CALENDAR_USERS.get(calendar, {})
    calendar_id = cal_config.get("calendar_id", "primary")

    try:
        creds = _get_shared_creds()
        if creds is not None and _build_service is not None:
            time_min, time_max, max_results = _compute_time_bounds(
                read_type, cmd, include_past, limit,
            )
            _, events, error = _fetch_events_for_calendar(
                creds, calendar, calendar_id, time_min, time_max, max_results,
            )
            if error:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Read operation failed: {error}",
                    "command": cmd,
                }
        else:
            if read_type == "next_events":
                events = calendar_instance.get_upcoming_events(
                    calendar_name=calendar_id, max_results=limit, include_past=include_past
                )
            elif read_type == "day_summary":
                target_date = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
                events = calendar_instance.get_day_events(
                    date=target_date, calendar_name=calendar_id, include_past=include_past
                )
            elif read_type == "week_summary":
                events = calendar_instance.get_week_events(
                    calendar_name=calendar_id, include_past=include_past
                )
            elif read_type == "specific_date":
                target_date = cmd["date"]
                events = calendar_instance.get_day_events(
                    date=target_date, calendar_name=calendar_id, include_past=include_past
                )
            else:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Unknown read_type: {read_type}",
                    "command": cmd,
                }

            cal_error = getattr(calendar_instance, "error_message", None)
            if cal_error:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Read operation failed: {cal_error}",
                    "command": cmd,
                }

        cleaned_events = [_clean_event_for_agent(e) for e in events]
        result: Dict[str, Any] = {
            "success": True,
            "operation": "read",
            "command_index": cmd_index,
            "command": cmd,
            "read_type": read_type,
            "events": cleaned_events,
            "event_count": len(cleaned_events),
            "calendar": calendar,
            "note": "This was a READ operation - no events were created.",
        }
        if read_type in ["day_summary", "specific_date"]:
            result["date"] = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
        return result

    except Exception as e:
        return {
            "success": False,
            "command_index": cmd_index,
            "error": f"Read operation failed: {str(e)}",
            "command": cmd,
        }


def _handle_write_command(cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
    target_calendars = cmd.get("calendars")
    if not target_calendars:
        single_cal = cmd.get("calendar") or cmd.get("user") or "morgan_personal"
        target_calendars = [single_cal]

    for cal in target_calendars:
        if cal not in _CALENDAR_USERS:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Unknown calendar: {cal}. Available: {list(_CALENDAR_USERS.keys())}",
                "command": cmd,
            }

    created_events: List[Dict] = []
    errors: List[str] = []

    for cal_name in target_calendars:
        try:
            cal_instance = _get_calendar_instance(cal_name)
            if not cal_instance:
                errors.append(f"Failed to initialize {cal_name}")
                continue

            cal_config = _CALENDAR_USERS.get(cal_name, {})
            calendar_id = cal_config.get("calendar_id", "primary")

            event_data = {
                "title": cmd["event_title"],
                "description": cmd.get("event_description", ""),
                "date": cmd["date"],
                "start_time": cmd["start_time"],
                "end_time": cmd["end_time"],
                "location": cmd.get("location", ""),
                "attendees": cmd.get("attendees", []),
                "calendar_name": calendar_id,
                "time_zone": cmd.get("time_zone"),
            }

            created = cal_instance.create_event(event_data)
            created_events.append({
                "calendar": cal_name,
                "event_id": created.get("id"),
                "link": created.get("htmlLink"),
            })
        except Exception as e:
            errors.append(f"{cal_name}: {str(e)}")

    if not created_events:
        return {
            "success": False,
            "command_index": cmd_index,
            "error": f"Failed to create event on any calendar: {errors}",
            "command": cmd,
        }

    result: Dict[str, Any] = {
        "success": True,
        "command_index": cmd_index,
        "command": cmd,
        "operation": "create_event",
        "event_title": cmd["event_title"],
        "event_date": cmd["date"],
        "event_time": f"{cmd['start_time']} - {cmd['end_time']}",
        "created_on_calendars": created_events,
        "calendars_count": len(created_events),
    }

    if errors:
        result["warnings"] = errors

    # Briefing creation is not available in standalone mode
    create_briefing = cmd.get("create_briefing", True)
    if create_briefing and created_events:
        logger.info("Briefing creation is not available in standalone mode")
        result["briefing"] = {"success": False, "error": "Briefing creation not available in standalone mode"}

    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize inputs for calendar commands."""
    try:
        normalized = dict(cmd)

        # Handle nested 'create_event' object format
        if "create_event" in normalized and isinstance(normalized["create_event"], dict):
            create_obj = normalized.pop("create_event")
            normalized["read_or_write"] = "create_event"
            if "calendar" in create_obj:
                normalized["calendar"] = create_obj["calendar"]
                normalized["user"] = create_obj["calendar"]
            if "title" in create_obj:
                normalized["event_title"] = create_obj["title"]
            for field in ("start_time", "end_time", "date", "location", "attendees"):
                if field in create_obj:
                    normalized[field] = create_obj[field]
            if "description" in create_obj:
                normalized["event_description"] = create_obj["description"]

        # Resolve calendar aliases
        if normalized.get("calendar"):
            normalized["calendar"] = resolve_calendar_alias(normalized["calendar"])
            normalized["user"] = normalized["calendar"]
        elif normalized.get("user"):
            normalized["user"] = resolve_calendar_alias(normalized["user"])
            normalized["calendar"] = normalized["user"]

        if normalized.get("calendars") and isinstance(normalized["calendars"], list):
            normalized["calendars"] = [resolve_calendar_alias(c) for c in normalized["calendars"]]

        # Map common title synonyms
        if not normalized.get("event_title"):
            for alt in ("event_name", "title", "summary", "name"):
                if isinstance(normalized.get(alt), str) and normalized.get(alt).strip():
                    normalized["event_title"] = normalized[alt].strip()
                    break

            if not normalized.get("event_title") and isinstance(normalized.get("event"), dict):
                event_obj = normalized["event"]
                for alt in ("title", "event_name", "summary", "name"):
                    if isinstance(event_obj.get(alt), str) and event_obj.get(alt).strip():
                        normalized["event_title"] = event_obj[alt].strip()
                        break
                for key in ("start_time", "start"):
                    if key in event_obj:
                        val = event_obj[key]
                        if isinstance(val, dict):
                            if "time" in val:
                                normalized["start_time"] = val["time"]
                                if "date" in val and not normalized.get("date"):
                                    normalized["date"] = val["date"]
                                break
                            val = val.get("dateTime") or val.get("date") or ""
                        if isinstance(val, str) and val:
                            normalized["start_time"] = val
                            break
                for key in ("end_time", "end"):
                    if key in event_obj:
                        val = event_obj[key]
                        if isinstance(val, dict):
                            if "time" in val:
                                normalized["end_time"] = val["time"]
                                break
                            val = val.get("dateTime") or val.get("date") or ""
                        if isinstance(val, str) and val:
                            normalized["end_time"] = val
                            break
                if "date" in event_obj and not normalized.get("date"):
                    val = event_obj["date"]
                    if isinstance(val, dict):
                        val = val.get("dateTime") or val.get("date") or ""
                    if isinstance(val, str):
                        normalized["date"] = val
                if "description" in event_obj:
                    normalized["event_description"] = event_obj["description"]
                if "location" in event_obj:
                    normalized["location"] = event_obj["location"]

        # Top-level start/end aliases
        if not normalized.get("start_time") and normalized.get("start"):
            val = normalized.pop("start")
            if isinstance(val, dict):
                if "time" in val:
                    normalized["start_time"] = val["time"]
                    if "date" in val and not normalized.get("date"):
                        normalized["date"] = val["date"]
                    val = None
                else:
                    val = val.get("dateTime") or val.get("date") or ""
            if isinstance(val, str) and val:
                normalized["start_time"] = val
        if not normalized.get("end_time") and normalized.get("end"):
            val = normalized.pop("end")
            if isinstance(val, dict):
                if "time" in val:
                    normalized["end_time"] = val["time"]
                    val = None
                else:
                    val = val.get("dateTime") or val.get("date") or ""
            if isinstance(val, str) and val:
                normalized["end_time"] = val

        # Handle "action" field as alias
        if normalized.get("action") and not normalized.get("read_or_write"):
            action = normalized["action"]
            if action in ["create_event", "write"]:
                normalized["read_or_write"] = action
            elif action == "read":
                normalized["read_or_write"] = "read"

        wt = normalized.get("write_type")
        if wt and not normalized.get("read_or_write"):
            normalized["read_or_write"] = "create_event"

        if normalized.get("read_or_write") == "write":
            normalized["read_or_write"] = "create_event"

        if not normalized.get("read_or_write"):
            has_event_fields = normalized.get("event_title") or normalized.get("event_name")
            normalized["read_or_write"] = "create_event" if has_event_fields else "read"

        # Set default calendar based on operation type
        is_write = normalized.get("read_or_write") in ["write", "create_event"]
        current_calendar = normalized.get("calendar") or normalized.get("user")

        if is_write:
            if not current_calendar or current_calendar == "all":
                normalized["calendar"] = "morgan_personal"
                normalized["user"] = "morgan_personal"
        else:
            if not current_calendar:
                normalized["calendar"] = "all"
                normalized["user"] = "all"

        if normalized.get("read_or_write") == "read" and not normalized.get("read_type"):
            normalized["read_type"] = "next_events"

        # Normalize time strings
        for key in ("start_time", "end_time"):
            val = normalized.get(key)
            if not val or not isinstance(val, str):
                continue
            parsed = _parse_time_like(val)
            if parsed:
                if not normalized.get("date") and parsed.get("date"):
                    normalized["date"] = parsed["date"]
                if parsed.get("time"):
                    normalized[key] = parsed["time"]

        # Normalize date
        dval = normalized.get("date")
        if isinstance(dval, dict):
            dval = dval.get("dateTime") or dval.get("date") or ""
            if isinstance(dval, str):
                normalized["date"] = dval
        if isinstance(dval, str) and "T" in dval:
            try:
                iso = dval.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                normalized["date"] = dt.date().isoformat()
            except Exception:
                try:
                    normalized["date"] = dval.split("T", 1)[0]
                except Exception:
                    pass

        # Parse natural language dates
        dval = normalized.get("date")
        if dval and isinstance(dval, str):
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", dval):
                parsed_date = _parse_natural_date(dval)
                if parsed_date:
                    normalized["date"] = parsed_date

        # Legacy calendar_name defaults
        if not normalized.get("calendar_name"):
            if normalized.get("read_or_write") in ["write", "create_event"]:
                normalized["calendar_name"] = "morgan_personal"
            else:
                normalized["calendar_name"] = "primary"

        calendar_name_mapping = {
            "default": "primary",
            "main": "primary",
            "default_calendar": "primary",
            "assistant": "homeassist",
        }
        if normalized.get("calendar_name") in calendar_name_mapping:
            normalized["calendar_name"] = calendar_name_mapping[normalized["calendar_name"]]

        return normalized
    except Exception:
        return cmd


# ---------------------------------------------------------------------------
# Time / date parsing helpers
# ---------------------------------------------------------------------------


def _parse_time_like(value: str) -> Dict[str, Any]:
    try:
        v = value.strip().rstrip(".,;:!")
        if "T" in v or (len(v) >= 10 and v[:10].count("-") == 2):
            try:
                iso = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                return {"date": dt.date().isoformat(), "time": dt.strftime("%H:%M")}
            except Exception:
                pass

        m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", v)
        if m:
            hh, mm = int(m.group(1)), int(m.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return {"time": f"{hh:02d}:{mm:02d}"}

        m2 = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*([ap]m)$", v, flags=re.IGNORECASE)
        if m2:
            hh = int(m2.group(1))
            mm = int(m2.group(2) or 0)
            ap = m2.group(3).lower()
            if hh == 12:
                hh = 0
            if ap == "pm":
                hh += 12
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return {"time": f"{hh:02d}:{mm:02d}"}

        return {}
    except Exception:
        return {}


def _parse_natural_date(value: str) -> Optional[str]:
    """Parse natural language date strings into YYYY-MM-DD format."""
    if not value or not isinstance(value, str):
        return None

    v = value.strip().lower()

    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            pass

    if re.match(r"^\d{1,2}:\d{2}", v) or re.match(r"^\d{1,2}\s*[ap]m$", v, re.IGNORECASE):
        return None

    if DATEPARSER_AVAILABLE and dateparser:
        try:
            settings = {
                "PREFER_DATES_FROM": "future",
                "PREFER_DAY_OF_MONTH": "first",
                "RETURN_AS_TIMEZONE_AWARE": False,
            }
            parsed = dateparser.parse(value, settings=settings)
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

    today = datetime.now().date()

    if v in ("today", "tonight"):
        return today.isoformat()
    elif v == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    elif v == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    days_of_week = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    for day_name, day_num in days_of_week.items():
        if day_name in v:
            current_day = today.weekday()
            days_ahead = day_num - current_day
            if "next" in v:
                days_ahead += 7
            elif days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).isoformat()

    in_match = re.match(r"in\s+(\d+)\s+(day|days|week|weeks)", v)
    if in_match:
        num = int(in_match.group(1))
        unit = in_match.group(2)
        if "week" in unit:
            num *= 7
        return (today + timedelta(days=num)).isoformat()

    return None


# ---------------------------------------------------------------------------
# Calendar instance management
# ---------------------------------------------------------------------------


def _get_calendar_instance(user: str) -> Optional[Any]:
    if CalendarComponent is None:
        logger.error("CalendarComponent not available")
        return None
    if user not in _calendar_instances:
        try:
            _calendar_instances[user] = CalendarComponent(user=user)
        except Exception as e:
            logger.error("Failed to create calendar instance for %s: %s", user, e)
            return None
    return _calendar_instances[user]


# ---------------------------------------------------------------------------
# Format / validation helpers
# ---------------------------------------------------------------------------


def _is_valid_time_format(time_str: str) -> bool:
    if not isinstance(time_str, str):
        return False
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def _is_valid_date_format(date_str: str) -> bool:
    if not isinstance(date_str, str):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _get_current_timestamp() -> str:
    try:
        return datetime.now().isoformat()
    except Exception:
        return "unknown"


def _clean_event_for_agent(event: Dict[str, Any]) -> Dict[str, Any]:
    """Strip unnecessary Google Calendar API metadata from events."""
    if not isinstance(event, dict):
        return event

    cleaned: Dict[str, Any] = {}

    if "summary" in event:
        cleaned["summary"] = event["summary"]

    if "start" in event:
        start = event["start"]
        if isinstance(start, dict):
            cleaned["start"] = {k: v for k, v in start.items() if k in ("dateTime", "date", "timeZone")}
        else:
            cleaned["start"] = start

    if "end" in event:
        end = event["end"]
        if isinstance(end, dict):
            cleaned["end"] = {k: v for k, v in end.items() if k in ("dateTime", "date", "timeZone")}
        else:
            cleaned["end"] = end

    if event.get("location"):
        cleaned["location"] = event["location"]
    if event.get("description"):
        cleaned["description"] = event["description"]
    if "source_calendar" in event:
        cleaned["source_calendar"] = event["source_calendar"]

    return cleaned
