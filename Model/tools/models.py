"""
Pydantic argument models for tool call validation.

Defines one model per tool, a registry mapping tool names to models,
and a validate_tool_call() dispatcher used before execution.

Each model includes a ``model_validator(mode="before")`` pre-validator
that normalises common model-output inaccuracies (synonym actions,
missing inferrable fields, flat-vs-nested structures) so the execution
layer receives clean, canonical arguments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Per-tool argument models
# ---------------------------------------------------------------------------


_WEATHER_DEFAULTS = {"hours": 24}


class WeatherArgs(BaseModel):
    hours: int | None = Field(None, ge=1, le=168)
    days: int | None = Field(None, ge=1, le=7)
    specific_date: str | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_timeframe(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        has_hours = data.get("hours") is not None
        has_days = data.get("days") is not None
        has_date = data.get("specific_date") is not None
        if not has_hours and not has_days and not has_date:
            data["hours"] = _WEATHER_DEFAULTS["hours"]
        # Coerce string numerics the model sometimes emits
        for key in ("hours", "days"):
            val = data.get(key)
            if isinstance(val, str):
                try:
                    data[key] = int(val)
                except ValueError:
                    pass
        return data

    @model_validator(mode="after")
    def require_one_timeframe(self):
        if self.hours is None and self.days is None and self.specific_date is None:
            raise ValueError("Provide at least one of: hours, days, specific_date")
        return self


_SPOTIFY_ACTION_MAP = {
    "search": "search_track",
    "find": "search_track",
    "find_track": "search_track",
    "find_song": "search_track",
    "play_track": "search_track",
    "play_song": "search_track",
    "find_artist": "search_artist",
    "play_artist": "search_artist",
    "resume": "play",
    "unpause": "play",
    "continue": "play",
    "stop": "pause",
    "skip": "next",
    "skip_track": "next",
    "forward": "next",
    "back": "previous",
    "prev": "previous",
    "rewind": "previous",
    "vol": "volume",
    "set_volume": "volume",
    "now_playing": "status",
    "current": "status",
    "info": "status",
    "list_devices": "devices",
    "toggle_shuffle": "shuffle",
    "toggle_repeat": "repeat",
}


class SpotifyPlaybackArgs(BaseModel):
    action: str
    user: str | None = None
    query: str | None = None
    search_type: str | None = "track"
    volume_level: int | None = Field(None, ge=0, le=100)
    device_id: str | None = None
    shuffle_state: bool | None = None
    repeat_mode: str | None = None
    search_limit: int | None = Field(None, ge=1, le=20)

    @model_validator(mode="before")
    @classmethod
    def normalise_action(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        action = (data.get("action") or "").strip().lower()
        data["action"] = _SPOTIFY_ACTION_MAP.get(action, action)

        # "play" with a query but no search fields → search_track
        if data["action"] == "play" and data.get("query") and not data.get("search_type"):
            data["action"] = "search_track"

        # Handle "song"/"track"/"artist" as field alias for query
        for alias in ("song", "track", "artist_name", "song_name", "track_name"):
            if not data.get("query") and data.get(alias):
                data["query"] = data.pop(alias)

        return data


_KASA_INTERACTION_SYNONYMS = {
    "control": "direct",
    "switch": "direct",
    "toggle": "direct",
    "on": "direct",
    "off": "direct",
    "set_scene": "scene",
    "apply_scene": "scene",
    "activate_scene": "scene",
    "mood": "scene",
}


class KasaLightingArgs(BaseModel):
    interaction: str
    light_name: str | None = None
    room: str | None = None
    action: str | None = None
    scene_name: str | None = None
    light_names: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_interaction(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        interaction = (data.get("interaction") or "").strip().lower()

        # Map synonyms
        if interaction in _KASA_INTERACTION_SYNONYMS:
            data["interaction"] = _KASA_INTERACTION_SYNONYMS[interaction]
            return data

        # Already canonical
        if interaction in ("direct", "scene"):
            return data

        # Infer from other fields
        if data.get("scene_name"):
            data["interaction"] = "scene"
        elif data.get("action") and str(data["action"]).lower() in ("on", "off"):
            data["interaction"] = "direct"
        elif data.get("light_name"):
            data["interaction"] = "direct"
        elif data.get("light_names") or data.get("room"):
            data["interaction"] = "scene"
        else:
            data["interaction"] = "direct"

        # If interaction was "on"/"off", that was meant as the action
        if interaction in ("on", "off") and not data.get("action"):
            data["action"] = interaction

        return data


_CALENDAR_FIELD_KEYS = {
    "read_or_write", "calendar", "calendars", "read_type", "limit",
    "date", "event_title", "event_description", "event_name", "title",
    "summary", "start_time", "end_time", "location", "attendees",
    "create_briefing", "action", "user", "write_type",
}


class CalendarCommand(BaseModel):
    read_or_write: str | None = "read"
    calendar: str | None = "all"
    calendars: list[str] | None = None
    read_type: str | None = "next_events"
    limit: int | None = Field(10, ge=1, le=50)
    date: str | None = None
    event_title: str | None = None
    event_description: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    location: str | None = None
    attendees: list[str] | None = None
    create_briefing: bool | None = True


class CalendarDataArgs(BaseModel):
    commands: list[CalendarCommand] = []

    @model_validator(mode="before")
    @classmethod
    def wrap_flat_fields(cls, data: Any) -> Any:
        """If the model emitted flat calendar fields instead of commands: [...], wrap them."""
        if not isinstance(data, dict):
            return data
        if data.get("commands"):
            return data
        # Check if the top-level dict looks like a calendar command
        if any(k in data for k in _CALENDAR_FIELD_KEYS):
            cmd = {k: v for k, v in data.items() if k != "commands"}
            return {"commands": [cmd]}
        return data


_STICKIES_ACTION_MAP = {
    "edit": "write",
    "update": "write",
    "modify": "write",
    "set": "write",
    "append": "write",
    "add": "write",
    "delete": "write",
    "remove": "write",
    "get": "read",
    "view": "read",
    "show": "read",
    "open": "read",
    "all": "list",
    "show_all": "list",
    "list_all": "list",
    "ls": "list",
}


class StickiesArgs(BaseModel):
    action: str
    sticky_id: str | None = None
    content: str | None = None
    format: str | None = "plain"

    @model_validator(mode="before")
    @classmethod
    def normalise_action(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        action = (data.get("action") or "").strip().lower()
        data["action"] = _STICKIES_ACTION_MAP.get(action, action)

        # Handle field aliases
        for alias in ("edits", "text", "note", "body", "new_content"):
            if not data.get("content") and data.get(alias):
                data["content"] = data.pop(alias)

        return data


class SendSmsArgs(BaseModel):
    message: str

    @model_validator(mode="before")
    @classmethod
    def normalise_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for alias in ("text", "body", "content", "sms"):
            if not data.get("message") and data.get(alias):
                data["message"] = data.pop(alias)
        return data


class GoogleSearchArgs(BaseModel):
    query: str
    query_type: str | None = "general"

    @model_validator(mode="before")
    @classmethod
    def normalise_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for alias in ("search", "search_query", "q", "term"):
            if not data.get("query") and data.get(alias):
                data["query"] = data.pop(alias)
        return data


class ReadClipboardArgs(BaseModel):
    max_length: int | None = 8000


_BRIEFING_ACTION_MAP = {
    "get": "list",
    "read": "list",
    "show": "list",
    "view": "list",
    "fetch": "list",
    "set": "create",
    "add": "create",
    "new": "create",
    "remind": "create",
    "schedule": "create",
    "remove": "delete",
    "cancel": "delete",
    "dismiss": "delete",
    "clear": "delete",
}


class BriefingArgs(BaseModel):
    action: str
    user: str | None = None
    message: str | None = None
    remind_at: str | None = None
    remind_before_minutes: int | None = Field(None, ge=1, le=10080)
    event_time: str | None = None
    priority: str | None = "normal"
    briefing_id: str | None = None
    limit: int | None = Field(None, ge=1, le=50)

    @model_validator(mode="before")
    @classmethod
    def normalise_action(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        action = (data.get("action") or "").strip().lower()
        data["action"] = _BRIEFING_ACTION_MAP.get(action, action)
        for alias in ("text", "content", "body", "note"):
            if not data.get("message") and data.get(alias):
                data["message"] = data.pop(alias)
        return data


class GetNotificationsArgs(BaseModel):
    user: str | None = None
    type_filter: str | None = "all"
    limit: int | None = Field(10, ge=1, le=50)
    include_metadata: bool | None = True
    priority_filter: str | None = "all"


class SystemInfoArgs(BaseModel):
    section: str | None = "overview"


class CursorComposerArgs(BaseModel):
    prompt: str

    @model_validator(mode="before")
    @classmethod
    def normalise_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for alias in ("command", "instruction", "code", "message", "request"):
            if not data.get("prompt") and data.get(alias):
                data["prompt"] = data.pop(alias)
        return data


# ---------------------------------------------------------------------------
# Registry + validation
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, type[BaseModel]] = {
    "weather": WeatherArgs,
    "spotify_playback": SpotifyPlaybackArgs,
    "kasa_lighting": KasaLightingArgs,
    "calendar_data": CalendarDataArgs,
    "stickies": StickiesArgs,
    "send_sms": SendSmsArgs,
    "google_search": GoogleSearchArgs,
    "read_clipboard": ReadClipboardArgs,
    "briefing": BriefingArgs,
    "get_notifications": GetNotificationsArgs,
    "system_info": SystemInfoArgs,
    "cursor_composer": CursorComposerArgs,
}


class ValidatedToolCall(BaseModel):
    name: str
    arguments: Any


class ToolValidationError(BaseModel):
    tool_name: str
    error: str
    raw_arguments: dict


def _format_validation_error(exc: Exception) -> str:
    """Turn a Pydantic ValidationError into a human-readable string."""
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return str(exc)

    parts: list[str] = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        if loc:
            parts.append(f"'{loc}': {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts)


def validate_tool_call(
    name: str, arguments: dict
) -> ValidatedToolCall | ToolValidationError:
    """Validate raw tool-call arguments against the registered Pydantic model.

    Returns ValidatedToolCall on success, ToolValidationError on failure.
    """
    model_cls = TOOL_REGISTRY.get(name)
    if model_cls is None:
        return ToolValidationError(
            tool_name=name,
            error=f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY.keys())}",
            raw_arguments=arguments,
        )
    try:
        validated_args = model_cls.model_validate(arguments)
        return ValidatedToolCall(name=name, arguments=validated_args)
    except Exception as e:
        return ToolValidationError(
            tool_name=name,
            error=f"Argument validation failed for '{name}': {_format_validation_error(e)}",
            raw_arguments=arguments,
        )
