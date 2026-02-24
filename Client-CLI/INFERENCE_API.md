# Inference API -- Client Reference

## System Overview

This server wraps a fine-tuned **Qwen3-4B-Instruct** model that has been specialized for tool-calling through QLoRA training. The base model was fine-tuned on ~498 examples of tool-use conversations, then merged and quantized to GGUF (q4_k_m) for fast GPU inference via llama.cpp.

The model can do two things:
1. **Respond conversationally** when no tool is needed.
2. **Emit structured tool calls** when a user request maps to one of its 12 trained tools (weather, Spotify, smart lighting, calendar, sticky notes, SMS, web search, clipboard, briefings, notifications, system info, Cursor Composer).

The server handles all prompt formatting, tool-call parsing, and thinking/reasoning extraction internally. Clients send plain conversation messages and receive clean, structured JSON.

### Architecture

```
Client (remote)
    │
    ▼  HTTP + X-API-Key header
FastAPI  (server.py :8000)
    │
    ▼  localhost HTTP
llama-server  (llama.cpp :8080, GPU inference)
```

The FastAPI layer is stateless. It builds the model prompt from the conversation you send, forwards it to the local llama-server, then parses and validates the raw output before returning it. llama-server must be started separately before the FastAPI server.

---

## Authentication

The server uses a **rotating API key** system with two tokens:

- **Refresh token** -- A permanent secret set via the `INFERENCE_REFRESH_TOKEN` environment variable on the server. This is what your client stores long-term. It never changes unless manually rotated.
- **API key** -- A short-lived token that rotates every **5 minutes**. Derived deterministically from the refresh token and the current time window using HMAC-SHA256. You obtain it by calling `POST /auth/token`.

### How it works

1. Your client calls `POST /auth/token` with the refresh token to get the current API key and its expiry time.
2. Use that API key in the `X-API-Key` header for all `/v1/chat/completions` requests.
3. When the key expires (or a request returns `403`), call `/auth/token` again to get the new key.

The server accepts keys from the current **and** previous 5-minute window as a grace period, so a key obtained near the end of a window remains valid briefly after rotation.

The `/health` endpoint does **not** require authentication. The `/auth/token` endpoint does **not** require the `X-API-Key` header (it authenticates via the refresh token in the request body instead).

| Scenario | HTTP Status | Detail |
|---|---|---|
| Missing `X-API-Key` header | `422` | Validation error (header required) |
| Expired or wrong API key | `403` | `"Invalid or expired API key"` |
| Wrong refresh token on `/auth/token` | `403` | `"Invalid refresh token"` |
| `INFERENCE_REFRESH_TOKEN` env var not set | `500` | `"INFERENCE_REFRESH_TOKEN not configured"` |

---

## Endpoints

### `GET /health`

Unauthenticated. Returns the status of both the FastAPI server and the underlying llama-server.

**Response:**

```json
{
  "status": "ok",
  "llama_server": "reachable"
}
```

| Field | Type | Values |
|---|---|---|
| `status` | string | `"ok"` or `"degraded"` |
| `llama_server` | string | `"reachable"` or `"unreachable"` |

`"degraded"` means the FastAPI layer is up but llama-server is not responding.

---

### `POST /auth/token`

Unauthenticated (no `X-API-Key` header needed). Exchange your refresh token for the current short-lived API key.

**Request:**

```json
{
  "refresh_token": "your-permanent-secret"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `refresh_token` | string | **yes** | The permanent secret from `INFERENCE_REFRESH_TOKEN`. |

**Response:**

```json
{
  "api_key": "a3f8c1...64 hex chars",
  "expires_at": "2026-02-16T20:35:00+00:00"
}
```

| Field | Type | Description |
|---|---|---|
| `api_key` | string | The current 5-minute API key (64-character hex string). Use in `X-API-Key` header. |
| `expires_at` | string | ISO-8601 UTC timestamp when this key expires. Fetch a new one before or after this time. |

---

### `POST /v1/chat/completions`

Authenticated. Send a conversation, receive the model's response.

#### Request

**Headers:**

```
Content-Type: application/json
X-API-Key: <your-api-key>
```

**Body:**

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather tomorrow?"}
  ],
  "enable_thinking": true,
  "generation": {
    "max_tokens": 512,
    "temperature": 0.6
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `messages` | array of `ChatMessage` | **yes** | -- | The conversation history. |
| `enable_thinking` | bool | no | `true` | Whether the model should produce a reasoning trace. |
| `generation` | `GenerationParams` or null | no | `null` | Override sampling parameters. When null, server defaults from config.yaml are used. |

**ChatMessage:**

| Field | Type | Allowed Values |
|---|---|---|
| `role` | string | `"user"` or `"assistant"` |
| `content` | string | The message text. |

**GenerationParams** (all optional -- only include fields you want to override):

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | int | `512` | Maximum tokens to generate. |
| `temperature` | float | `0.6` | Sampling temperature (0.0 = deterministic). |
| `top_p` | float | `0.95` | Nucleus sampling cutoff. |
| `top_k` | int | `20` | Top-k token sampling limit. |
| `min_p` | float | `0.0` | Minimum probability threshold. |
| `repeat_penalty` | float | `1.0` | Repetition penalty (1.0 = disabled). |

#### Response

```json
{
  "content": "I'll check the weather for you.",
  "thinking": "The user wants tomorrow's weather. I should call the weather tool with specific_date='tomorrow'.",
  "tool_calls": [
    {
      "name": "weather",
      "arguments": {"specific_date": "tomorrow"}
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
| `content` | string | no | The cleaned response text (tool-call XML and artifacts stripped). |
| `thinking` | string or null | yes | The model's internal reasoning trace (between `<think>` tags). Null if thinking was disabled or the model didn't produce one. |
| `tool_calls` | array of `ToolCall` or null | yes | Parsed tool calls. Null if the model didn't call any tools. |
| `usage` | `Usage` or null | yes | Token counts and generation speed. Null if timings were unavailable. |

**ToolCall:**

| Field | Type | Description |
|---|---|---|
| `name` | string | The tool name (e.g. `"weather"`, `"spotify_playback"`). |
| `arguments` | object | The arguments as a JSON object matching the tool's parameter schema. |

**Usage:**

| Field | Type | Description |
|---|---|---|
| `prompt_tokens` | int | Number of tokens in the prompt. |
| `completion_tokens` | int | Number of tokens generated. |
| `tokens_per_second` | float | Generation speed. |

---

## Error Responses

All errors return a JSON body with a `detail` field:

```json
{"detail": "error description"}
```

| Status | Meaning |
|---|---|
| `403` | Invalid or expired API key, or invalid refresh token. |
| `422` | Request validation failed (malformed body, missing required fields, wrong types). |
| `500` | Server misconfiguration (`INFERENCE_REFRESH_TOKEN` env var not set). |
| `502` | llama-server is unreachable. |
| `504` | llama-server timed out (>300 seconds). |

---

## Client Examples

### Python (requests) -- full flow with token refresh

```python
import requests
from datetime import datetime, timezone

BASE_URL = "http://<server-ip>:8000"
REFRESH_TOKEN = "your-permanent-secret"

api_key = None
expires_at = None


def get_api_key():
    """Fetch or refresh the short-lived API key."""
    global api_key, expires_at
    now = datetime.now(timezone.utc)
    if api_key and expires_at and now < expires_at:
        return api_key
    resp = requests.post(
        f"{BASE_URL}/auth/token",
        json={"refresh_token": REFRESH_TOKEN},
    )
    resp.raise_for_status()
    data = resp.json()
    api_key = data["api_key"]
    expires_at = datetime.fromisoformat(data["expires_at"])
    return api_key


def chat(messages):
    """Send a chat request, auto-refreshing the key on 403."""
    headers = {"Content-Type": "application/json", "X-API-Key": get_api_key()}
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": messages},
        headers=headers,
        timeout=120,
    )
    if resp.status_code == 403:
        global api_key
        api_key = None  # force refresh
        headers["X-API-Key"] = get_api_key()
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"messages": messages},
            headers=headers,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


# Health check (no auth needed)
print(requests.get(f"{BASE_URL}/health").json())

# Chat
data = chat([{"role": "user", "content": "Play some jazz music"}])
print(data["content"])

if data.get("tool_calls"):
    for tc in data["tool_calls"]:
        print(f"Tool: {tc['name']}, Args: {tc['arguments']}")
```

### Multi-turn conversation

The server is stateless. To maintain a conversation, accumulate messages on the client side and send the full history each time:

```python
messages = []

# Turn 1
messages.append({"role": "user", "content": "What's on my calendar today?"})
data = chat(messages)
messages.append({"role": "assistant", "content": data["content"]})

# Turn 2
messages.append({"role": "user", "content": "Create an event at 3pm called Team Standup"})
data = chat(messages)
messages.append({"role": "assistant", "content": data["content"]})
```

### curl

```bash
# Health
curl http://<server-ip>:8000/health

# Get API key (exchange refresh token)
curl -X POST http://<server-ip>:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your-permanent-secret"}'
# Returns: {"api_key": "a3f8c1...", "expires_at": "2026-02-16T20:35:00+00:00"}

# Chat (use the api_key from above)
curl -X POST http://<server-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: a3f8c1..." \
  -d '{"messages": [{"role": "user", "content": "Turn off the bedroom lights"}]}'
```

---

## Available Tools

The model was trained to call these 12 tools. When a user message maps to one, the response will contain a `tool_calls` array. Your client is responsible for executing the tool and (optionally) feeding the result back as a follow-up message.

| Tool | Description |
|---|---|
| `weather` | Weather forecast (hourly or daily, up to 7 days). |
| `spotify_playback` | Spotify control (play, pause, skip, volume, search, shuffle, repeat). |
| `kasa_lighting` | Smart light control (on/off, scenes by room). |
| `calendar_data` | Google Calendar (read events, create events). |
| `stickies` | Desktop sticky notes (read/write notes and to-dos). |
| `send_sms` | Send an iMessage/text to the user's phone. |
| `google_search` | Web search (general, links, directions). |
| `read_clipboard` | Read system clipboard contents. |
| `briefing` | Create/list/dismiss spoken briefing announcements. |
| `get_notifications` | Check pending notifications (email, news). |
| `system_info` | Documentation about the assistant's own architecture. |
| `cursor_composer` | Send a coding task to Cursor Composer. |

If the model determines no tool is needed, `tool_calls` will be `null` and `content` will contain the conversational response.

---

## Notes

- **Stateless**: The server does not store conversation history. Send the full message list on every request.
- **Timeout**: The server allows up to 300 seconds for llama-server to respond. Long prompts with high `max_tokens` may take time.
- **Context window**: 4096 tokens. The 12 tool schemas consume ~2661 tokens of this, leaving ~1435 tokens for conversation + response. Keep message history short or truncate older turns on the client side.
- **Thinking mode**: When `enable_thinking` is true, the model produces a reasoning trace before responding. This uses extra tokens but generally improves tool-call accuracy. The thinking text is returned separately in the `thinking` field and is not included in `content`.
- **Interactive docs**: When the server is running, visit `http://<server-ip>:8000/docs` for the auto-generated Swagger UI where you can test requests directly.
