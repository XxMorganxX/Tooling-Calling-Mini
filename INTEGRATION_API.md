# Tool-Calling Mini — External Integration Spec

> **Auto-generated** from [`tool_calling_config.json`](https://raw.githubusercontent.com/XxMorganxX/Tooling-Calling-Mini/master/Model/model_qwen4_finetuning/tool_calling_config.json)
> on 2026-03-16T04:17:52Z. Do not edit by hand — run `python scripts/generate_openclaw_spec.py` to regenerate.

This document gives an external agent everything it needs to delegate tool calls
to the Tool-Calling Mini inference API.

---

## 1. API Endpoint

| Item | Value |
|---|---|
| Base URL | `https://inference.stuart-labs.com` |
| Chat endpoint | `POST /v1/chat/completions` |
| Stream endpoint | `POST /v1/chat/completions/stream` |
| Health check | `GET /health` (no auth) |
| Interactive docs | `https://inference.stuart-labs.com/docs` (Swagger UI) |

Served via Cloudflare Tunnel. The server listens on `localhost:8000` internally.

---

## 2. Authentication

The API uses **rotating 5-minute API keys** derived from a permanent refresh token.

### Flow

1. **Exchange refresh token for API key:**

```http
POST /auth/token
Content-Type: application/json

{"refresh_token": "<REFRESH_TOKEN>"}
```

Response:
```json
{
  "api_key": "a3f8c1...64-hex-chars",
  "expires_at": "2026-02-16T20:35:00+00:00"
}
```

2. **Use API key in all subsequent requests:**

```
X-API-Key: <api_key>
```

3. **Refresh on expiry or 403.** The server accepts keys from the current and previous
   5-minute window as a grace period.

---

## 3. Request Format

```http
POST /v1/chat/completions
Content-Type: application/json
X-API-Key: <api_key>
```

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather tomorrow?"}
  ],
  "enable_thinking": true,
  "execute_tools": true,
  "generation": {
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `messages` | array | **yes** | — | Conversation history. Each: `{"role": "user"|"assistant"|"tool"|"system", "content": "..."}` |
| `enable_thinking` | bool | no | `true` | Model produces a reasoning trace before responding. |
| `execute_tools` | bool | no | `true` | Execute server-side tools automatically. Set to `false` to get tool calls without execution. |
| `generation` | object | no | server defaults | Override sampling params (all sub-fields optional). |

The server is **stateless** — send the full conversation history each request.

**Context window**: 4096 tokens. Tool schemas consume ~2661 tokens, leaving ~1435 for
conversation + response. Keep history short.

---

## 4. Response Format

```json
{
  "content": "I'll check the weather for you.",
  "thinking": "The user wants tomorrow's weather...",
  "tool_calls": [
    {
      "name": "weather",
      "arguments": {"specific_date": "tomorrow"}
    }
  ],
  "tool_results": [
    {
      "tool_name": "weather",
      "success": true,
      "result": {"temperature": 72, "conditions": "Partly cloudy"},
      "error": null,
      "duration_ms": 230.5
    }
  ],
  "usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 67,
    "tokens_per_second": 48.3
  }
}
```

| Field | Type | Nullable | Description |
|---|---|---|---|
| `content` | string | no | Cleaned response text (tool-call artifacts stripped). |
| `thinking` | string | yes | Reasoning trace. Null if disabled or not produced. |
| `tool_calls` | array | yes | Parsed tool calls. **Null when no tools needed.** Includes both server-side and client-side tool calls. |
| `tool_results` | array | yes | Results from server-side tool execution. Null if `execute_tools` is false or no server-side tools were called. |
| `usage` | object | yes | Token counts and generation speed. |

Each `tool_calls` entry:

| Field | Type |
|---|---|
| `name` | Tool name (string) |
| `arguments` | JSON object matching the tool's schema |

Each `tool_results` entry:

| Field | Type | Description |
|---|---|---|
| `tool_name` | string | Name of the tool that was executed |
| `success` | bool | Whether execution succeeded |
| `result` | any | Tool output (structure varies by tool). Null on failure. |
| `error` | string | Error message. Null on success. |
| `duration_ms` | float | Execution time in milliseconds |

### Error codes

| Status | Meaning |
|---|---|
| 403 | Expired/invalid API key or refresh token |
| 422 | Malformed request body |
| 500 | Server misconfiguration |
| 502 | llama-server unreachable |
| 504 | llama-server timeout (>300s) |

---

## 5. When to Defer

Delegate to this API when the user's request matches one of the **12 registered tools**:

| Tool | Description |
|---|---|
| `weather` | Get weather forecast for your current location. Uses hourly forecasts up to 36 hours or daily forecasts beyond that (max 7 days). |
| `spotify_playback` | Control Spotify playback: play/pause, track navigation, volume, search, shuffle, repeat. |
| `kasa_lighting` | Control Kasa smart lights. Direct control (on/off) for individual lights, or apply scenes across rooms. |
| `calendar_data` | Google Calendar access. Read events or create new events. Use 'all' to read from all calendars. |
| `stickies` | Read and edit the user's desktop sticky note (macOS Stickies app). For notes, to-do lists, things written down. |
| `send_sms` | Send a text message (iMessage) to the user's phone. For reminders or notifications to mobile device. |
| `google_search` | Search the web for up-to-date information. Only call ONCE per request. |
| `read_clipboard` | Read the current contents of the user's system clipboard. |
| `briefing` | Create and manage briefing announcements spoken to users when they wake up the assistant. |
| `get_notifications` | Check for pending notifications like email summaries and news. |
| `system_info` | Get information about how this voice assistant works internally. |
| `cursor_composer` | Send a coding request to Cursor's Composer for multi-file edits. Use when the user wants to write, refactor, or modify code in their codebase. The prompt will be pasted into Cursor's Composer interface for review. Only works on macOS. |

**Decision rule**: If the user's intent involves any of the above capabilities
(home automation, media control, calendar, weather, web search, notifications,
notes, SMS, briefings, clipboard, system info, or coding tasks), send the message
to this API. The model handles tool selection and argument extraction internally —
just forward the natural-language request.

If `tool_calls` is null in the response, the model determined no tool was needed
and answered conversationally — use `content` directly.

---

## 6. Integration Pattern

```
User message arrives
    │
    ├─ Does it involve one of the 12 tool domains? ──► NO ──► Handle normally
    │
    ▼ YES
POST /v1/chat/completions  (with message history)
    │
    ▼
Check response.tool_calls
    │
    ├─ null ──► Use response.content as conversational answer
    │
    ▼ non-null
Execute each tool call locally (you own the tool implementations)
    │
    ▼
Optionally feed results back as a follow-up message
    │
    ▼
Synthesize final response for the user
```

### Python integration example

```python
import requests
from datetime import datetime, timezone

BASE_URL = "https://inference.stuart-labs.com"
REFRESH_TOKEN = "<your-refresh-token>"

api_key = None
expires_at = None


def get_api_key():
    global api_key, expires_at
    now = datetime.now(timezone.utc)
    if api_key and expires_at and now < expires_at:
        return api_key
    resp = requests.post(f"{BASE_URL}/auth/token", json={"refresh_token": REFRESH_TOKEN})
    resp.raise_for_status()
    data = resp.json()
    api_key = data["api_key"]
    expires_at = datetime.fromisoformat(data["expires_at"])
    return api_key


def delegate(messages: list[dict]) -> dict:
    headers = {"Content-Type": "application/json", "X-API-Key": get_api_key()}
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": messages},
        headers=headers,
        timeout=120,
    )
    if resp.status_code == 403:
        api_key = None
        headers["X-API-Key"] = get_api_key()
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"messages": messages},
            headers=headers,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()
```

---

## 7. Full Tool Schemas

Canonical definitions from
[`tool_calling_config.json`](https://raw.githubusercontent.com/XxMorganxX/Tooling-Calling-Mini/master/Model/model_qwen4_finetuning/tool_calling_config.json).

### `weather`

Get weather forecast for your current location. Uses hourly forecasts up to 36 hours or daily forecasts beyond that (max 7 days).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `hours` | integer | no | Timeframe in hours (1-168). Uses hourly forecasts when <= 36 hours. (min=1, max=168) |
| `days` | integer | no | Timeframe in days (1-7). (min=1, max=7) |
| `specific_date` | string | no | Get weather for a specific date. Format: 'YYYY-MM-DD' or relative terms like 'today', 'tomorrow', 'monday'. |

**Constraint**: Provide one of: `hours`, `days`, `specific_date`.

---

### `spotify_playback`

Control Spotify playback: play/pause, track navigation, volume, search, shuffle, repeat.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | The Spotify action to perform. Values: `play`, `pause`, `next`, `previous`, `volume`, `search_track`, `search_artist`, `status`, `devices`, `shuffle`, `repeat`. |
| `query` | string | no | Search query for track or artist when using search actions or 'play' with a query. |
| `search_type` | string | no | When using 'play' with a query, choose whether to search tracks or artists. Values: `track`, `artist`. Default: `"track"`. |
| `volume_level` | integer | no | Volume level percentage (0-100) when action is 'volume'. (min=0, max=100) |
| `shuffle_state` | boolean | no | For shuffle action: true enables, false disables. If omitted, toggles. |
| `repeat_mode` | string | no | For repeat action: 'off', 'track' (loop one song), 'context' (loop playlist). If omitted, cycles. Values: `off`, `track`, `context`. |

---

### `kasa_lighting`

Control Kasa smart lights. Direct control (on/off) for individual lights, or apply scenes across rooms.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `interaction` | string | yes | 'direct' to control a single light, 'scene' to apply a preset. Values: `direct`, `scene`. |
| `light_name` | string | no | Specific light to control (direct mode). e.g., 'Light 1', 'Light 2'. |
| `room` | string | no | Room name for scene target. e.g., 'living room', 'bedroom'. |
| `action` | string | no | Direct mode action: 'on' or 'off'. Values: `on`, `off`. |
| `scene_name` | string | no | Scene to apply (scene mode). Values: `movie`, `work`, `mood`, `reading`, `party`, `relax`. |
| `light_names` | array | no | Optional list of specific lights for scene mode instead of room. |

---

### `calendar_data`

Google Calendar access. Read events or create new events. Use 'all' to read from all calendars.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `commands` | array of objects | no | Array with ONE command object. |

**`commands` object fields:**

| Field | Type | Description |
|---|---|---|
| `read_or_write` | string | 'read' to view events, 'create_event' to add an event. Values: `read`, `create_event`. |
| `calendar` | string | 'all' for reading all calendars, or specific calendar like 'morgan_personal'. |
| `read_type` | string | What to fetch for read operations. Values: `next_events`, `day_summary`, `week_summary`, `specific_date`. |
| `limit` | integer | Max events to return (default: 10). |
| `date` | string | Date in YYYY-MM-DD or natural language like 'tomorrow'. |
| `event_title` | string | For create_event: the event title. |
| `event_description` | string | For create_event: optional event details/notes. |
| `start_time` | string | For create_event: start time like '14:00' or '2pm'. |
| `end_time` | string | For create_event: end time. |
| `location` | string | Optional event location. |

---

### `stickies`

Read and edit the user's desktop sticky note (macOS Stickies app). For notes, to-do lists, things written down.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | 'read' returns note content; 'write' applies edits. Values: `read`, `write`. |
| `section` | string | no | For read: which section to return. Values: `notes`, `todo`, `both`. |
| `edits` | array of objects | no | For write: array of edit operations. |

**`edits` object fields:**

| Field | Type | Description |
|---|---|---|
| `op` | string | The edit operation to perform. Values: `add_todo`, `remove_todo`, `edit_todo`, `add_note`, `remove_note`, `edit_note`. **(required)** |
| `item` | string | For add_todo: the task text. |
| `due` | string | For add_todo: optional due date (e.g., 'Feb 12', 'Monday'). |
| `match` | string | For remove operations: text to match. |
| `old` | string | For edit operations: existing text to find. |
| `new` | string | For edit operations: replacement text. |
| `subheading` | string | For add_note: the note subheading. |
| `content` | string | For add_note: the note content. |

---

### `send_sms`

Send a text message (iMessage) to the user's phone. For reminders or notifications to mobile device.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The text message to send. |

---

### `google_search`

Search the web for up-to-date information. Only call ONCE per request.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | The search query. Be specific for better results. |
| `query_type` | string | no | 'general' for info, 'link' for URLs, 'directions' for navigation. Values: `general`, `link`, `directions`. |

---

### `read_clipboard`

Read the current contents of the user's system clipboard.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `max_length` | integer | no | Max characters to return (default: 8000). |

---

### `briefing`

Create and manage briefing announcements spoken to users when they wake up the assistant.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | 'create' new briefing, 'list' pending, 'dismiss' cancel. Values: `create`, `list`, `dismiss`. |
| `message` | string | no | For create: the briefing message. |
| `remind_at` | string | no | Absolute reminder time in ISO format or natural language like '9am', 'tomorrow 3pm'. |
| `remind_before_minutes` | integer | no | Minutes before event_time to remind. |
| `event_time` | string | no | When the actual event is (for relative reminders). |
| `priority` | string | no | Priority level. Default: 'normal'. Values: `high`, `normal`, `low`. |
| `briefing_id` | string | no | For dismiss: the briefing ID. |

---

### `get_notifications`

Check for pending notifications like email summaries and news.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `type_filter` | string | no | 'email' for email summaries, 'news' for news, 'all' for everything. Values: `email`, `news`, `other`, `all`. |
| `limit` | integer | no | Max notifications to return. |
| `priority_filter` | string | no | Filter by priority. Default: 'all'. Values: `high`, `normal`, `low`, `all`. |

---

### `system_info`

Get information about how this voice assistant works internally.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `section` | string | no | Specific documentation section. Values: `overview`, `architecture`, `providers`, `orchestrator`, `audio`, `tools`, `memory`, `config`, `all`. |

---

### `cursor_composer`

Send a coding request to Cursor's Composer for multi-file edits. Use when the user wants to write, refactor, or modify code in their codebase. The prompt will be pasted into Cursor's Composer interface for review. Only works on macOS.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | The coding task to send to Cursor Composer. Be specific about what files, functions, or changes are needed. This will be pasted into Cursor's Composer for review before execution. |

---

## 8. Keeping This Spec Up to Date

This file is generated from the canonical tool definitions at:

```
Model/model_qwen4_finetuning/tool_calling_config.json
```

When tools are added or modified:

1. Edit `tool_calling_config.json`
2. Run `python scripts/generate_openclaw_spec.py`
3. Commit and push

The raw-file URL for periodic polling:

```
https://raw.githubusercontent.com/XxMorganxX/Tooling-Calling-Mini/master/INTEGRATION_API.md
```

The raw tool config JSON (if you prefer to parse schemas directly):

```
https://raw.githubusercontent.com/XxMorganxX/Tooling-Calling-Mini/master/Model/model_qwen4_finetuning/tool_calling_config.json
```

### Automation options

- **Git pre-commit hook**: Add `python scripts/generate_openclaw_spec.py` to
  `.git/hooks/pre-commit` so the spec regenerates on every commit that touches
  `tool_calling_config.json`.
- **GitHub Action**: Trigger on pushes to `Model/model_qwen4_finetuning/tool_calling_config.json`
  and auto-commit the regenerated `INTEGRATION_API.md`.
- **Polling**: Have openclaw fetch the raw URL on a schedule (e.g. every 15 minutes)
  and diff against its cached copy.

---

*Generated 2026-03-16T04:17:52Z — 12 tools registered*
