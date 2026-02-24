# Tool Mini-Model Client

Python client for a fine-tuned Qwen3-4B tool-calling model served via FastAPI + llama.cpp. Includes an importable SDK and an interactive terminal chat.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your refresh token

The server uses rotating API keys. You store a permanent **refresh token** in `.env`, and the client automatically exchanges it for short-lived API keys (rotated every 5 minutes).

Copy the example env file and fill in your token:

```bash
cp .env.example .env
```

Then edit `.env`:

```
INFERENCE_REFRESH_TOKEN=your-permanent-secret
```

### 3. Adjust settings (optional)

Edit `config.yaml` to change the server URL, generation parameters, or conversation limits:

```yaml
server:
  url: "http://localhost:8000"   # point at your inference server
  timeout: 120

model:
  enable_thinking: true

generation:
  max_tokens: 512
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  repeat_penalty: 1.0

conversation:
  max_history_messages: 10       # older turns are dropped to fit context window
```

## Interactive Chat

Start the REPL:

```bash
python main.py
```

Type messages to chat with the model. Responses, tool calls, and thinking traces are rendered in the terminal.

### Commands

| Command     | Description                        |
|-------------|------------------------------------|
| `/clear`    | Clear conversation history         |
| `/thinking` | Toggle thinking mode on/off        |
| `/config`   | Show current configuration         |
| `/help`     | Show available commands            |
| `/quit`     | Exit the client                    |

## Library Usage

Use the client programmatically in your own scripts:

```python
from src import load_config, InferenceClient, ConversationManager

config = load_config()
client = InferenceClient(config)

# Health check
health = client.health()
print(health.status, health.llama_server)

# Single request
from src.models import ChatMessage
response = client.chat([ChatMessage(role="user", content="What's the weather?")])
print(response.content)
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"  {tc.name}: {tc.arguments}")

# Multi-turn conversation
manager = ConversationManager(client, config)
r1 = manager.send("What's on my calendar today?")
r2 = manager.send("Create an event at 3pm called Team Standup")
manager.clear()
```

## Project Structure

```
├── src/
│   ├── __init__.py           # Package exports
│   ├── client.py             # InferenceClient -- HTTP calls to the API
│   ├── models.py             # Pydantic models for requests and responses
│   ├── config.py             # Config loader (YAML + .env)
│   └── conversation.py       # ConversationManager -- history and truncation
├── config.yaml               # Server URL, generation defaults, context limits
├── .env.example              # Refresh token template
├── main.py                   # Interactive terminal chat
├── requirements.txt          # Python dependencies
├── README.md
└── INFERENCE_API.md          # Server API reference
```

## Available Tools

The model was trained to call these 12 tools. When one is triggered, the response's `tool_calls` field will contain the tool name and arguments. Your code is responsible for executing the tool.

| Tool               | Description                                         |
|--------------------|-----------------------------------------------------|
| `weather`          | Weather forecast (hourly or daily, up to 7 days)    |
| `spotify_playback` | Spotify control (play, pause, skip, volume, search) |
| `kasa_lighting`    | Smart light control (on/off, scenes by room)        |
| `calendar_data`    | Google Calendar (read events, create events)        |
| `stickies`         | Desktop sticky notes (read/write notes and to-dos)  |
| `send_sms`         | Send an iMessage/text to the user's phone           |
| `google_search`    | Web search (general, links, directions)             |
| `read_clipboard`   | Read system clipboard contents                      |
| `briefing`         | Create/list/dismiss spoken briefing announcements   |
| `get_notifications`| Check pending notifications (email, news)           |
| `system_info`      | Documentation about the assistant's own architecture|
| `cursor_composer`  | Send a coding task to Cursor Composer               |

## Authentication

The server uses a **rotating API key** system:

1. Your `.env` stores a permanent **refresh token** (`INFERENCE_REFRESH_TOKEN`).
2. On startup (and transparently during use), the client calls `POST /auth/token` to exchange the refresh token for a short-lived API key that rotates every 5 minutes.
3. The client automatically re-authenticates when a key expires or a `403` is received.

You never need to manage API keys manually -- just set the refresh token once.

## Notes

- The server is **stateless** -- the client sends the full conversation history with every request.
- The model has a **4096-token context window**. The 12 tool schemas consume ~2661 tokens, leaving ~1435 for conversation + response. The `max_history_messages` config keeps history within this budget.
- **Thinking mode** adds a reasoning trace before the response at the cost of extra tokens. Toggle it at runtime with `/thinking`.
