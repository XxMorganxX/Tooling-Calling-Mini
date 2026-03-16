# Tools

Server-side tool implementations, argument validation, and the execution engine.

## Architecture

```
Model output → parse_tool_calls() → validate_tool_call() → ToolExecutor → tool function → ToolResult
```

1. **`models.py`** — Pydantic models for each tool's arguments + a `validate_tool_call()` dispatcher
2. **`executor.py`** — `ToolExecutor` class that dispatches validated calls to tool functions with timeout handling
3. **Individual tool files** — Each tool's `execute()` function

## Files

| File | Tool Name | Description |
|------|-----------|-------------|
| `models.py` | — | Pydantic argument models, validation registry, synonym normalization |
| `executor.py` | — | `ToolExecutor` — dispatch, timeout, batch execution, `ToolResult` model |
| `weather.py` | `weather` | Weather forecast (hourly/daily, up to 7 days) |
| `spotify.py` | `spotify_playback` | Spotify control (play, pause, skip, volume, search, shuffle, repeat) |
| `kasa_lighting.py` | `kasa_lighting` | Kasa smart light control (on/off, scenes by room) |
| `calendar_tool.py` | `calendar_data` | Google Calendar (read events, create events) |
| `stickies.py` | `stickies` | macOS Stickies app (read/write/list notes) |
| `sms.py` | `send_sms` | Send iMessage/SMS |
| `google_search.py` | `google_search` | Web search (general, links, directions) |
| `clipboard.py` | `read_clipboard` | Read system clipboard contents |
| `briefing.py` | `briefing` | Spoken briefing announcements (create/list/delete) |
| `notifications.py` | `get_notifications` | Check pending notifications (email, news) |
| `system_info.py` | `system_info` | Internal system architecture documentation |
| `cursor.py` | `cursor_composer` | Send coding tasks to Cursor Composer |

## Validation Layer (`models.py`)

Each tool has a Pydantic `BaseModel` subclass with `model_validator(mode="before")` pre-validators that normalize common model-output quirks:

- **Synonym mapping** — e.g., Spotify `"resume"` → `"play"`, `"skip"` → `"next"`
- **Field aliasing** — e.g., SMS `"text"` / `"body"` → `"message"`
- **Structure inference** — e.g., flat calendar fields auto-wrapped into `commands: [...]`
- **Type coercion** — e.g., string `"24"` → int `24` for weather hours

The `TOOL_REGISTRY` dict maps tool names to their argument model classes. `validate_tool_call(name, arguments)` returns either a `ValidatedToolCall` or `ToolValidationError`.

## Executor (`executor.py`)

`ToolExecutor` accepts a mapping of tool names to callables. Key behavior:

- Per-tool timeout (configurable via `agent.tool_timeout` in `config.yaml`)
- Returns structured `ToolResult` objects with `success`, `result`, `error`, `duration_ms`
- `execute_batch()` runs multiple tool calls sequentially
- Comprehensive error string construction merging `error`, `hint`, `suggestion`, etc.

## Adding a New Tool

1. Add the tool's schema to `../model_qwen4_finetuning/tool_calling_config.json`
2. Create `<tool_name>.py` with an `execute(params)` function
3. Add a Pydantic argument model to `models.py` and register it in `TOOL_REGISTRY`
4. Register the execute function in `executor.py` (or in `server.py` where `ToolExecutor` is instantiated)
5. Add training examples to `../model_qwen4_finetuning/data/training_data.jsonl`

## Dependencies

Tool implementations may depend on external API clients in `../clients/`. The tools themselves are registered and invoked by `server.py` at the `Model/` root.
