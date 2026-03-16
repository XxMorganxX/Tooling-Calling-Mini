# Model (Server)

FastAPI inference server, training pipeline, tool execution engine, and all backend logic for the Qwen3 tool-calling system.

## Architecture

```
User Request
    ↓ HTTP + X-API-Key
server.py (FastAPI :8000)
    ↓ localhost HTTP
llama-server (llama.cpp :8080)
    ↓ GPU inference
Model Response → tool_calls?
    ↓ yes
ToolExecutor → tool implementations
    ↓
Final synthesised response
```

## Entry Points

| File | Purpose | How to Run |
|------|---------|------------|
| `server.py` | FastAPI inference server (primary entry point) | `python server.py` |
| `agent.py` | Multi-turn agent loop (tool-call → result → re-prompt cycle) | Imported by `server.py` |
| `context.py` | Server-side token-aware context manager | Imported by `agent.py` |

## Subfolders

| Folder | Purpose | Has README |
|--------|---------|------------|
| `inference/` | Core inference functions, GGUF export, validation | Yes |
| `model_qwen4_finetuning/` | Training pipeline, config, tool schemas, data prep | Yes |
| `tools/` | Tool implementations and execution engine | Yes |
| `clients/` | External API clients used by tools | Yes |
| `tool_calling_sdk/` | Shared SDK for building conversation context | Yes |
| `homeassist-ref/` | Git submodule — source of truth for tool definitions | Upstream repo |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check (FastAPI + llama-server reachability) |
| `/auth/token` | POST | No | Exchange refresh token for 5-minute API key |
| `/v1/chat/completions` | POST | Yes | Chat completion with tool-call support |
| `/v1/chat/completions/stream` | POST | Yes | Streaming chat completion (SSE) |

## Authentication

Uses HMAC-SHA256 rotating API keys derived from a permanent refresh token (`INFERENCE_REFRESH_TOKEN` in `.env`). Keys rotate every 5 minutes; the server accepts both the current and previous window for a grace period.

## Running

```bash
cd Model
pip install -r inference/requirements.txt

# Set refresh token
echo 'INFERENCE_REFRESH_TOKEN=your-secret' > .env

# Start (auto-launches llama-server if not running)
python server.py
# → FastAPI on http://0.0.0.0:8000
# → llama-server on http://127.0.0.1:8080
```

## Agent Loop (`agent.py`)

The `AgentLoop` class drives multi-turn tool execution:

1. Sends conversation to llama-server via `/completion`
2. Parses `<tool_call>` tags from the response
3. Validates arguments against Pydantic models in `tools/models.py`
4. Executes tools via `ToolExecutor`
5. Injects tool results back as `role=tool` messages
6. Repeats until the model produces a final text answer or max iterations hit

## Context Manager (`context.py`)

Server-side `ContextManager` handles:
- Token-aware message truncation to fit the 4096-token context window
- Tool-result injection into the conversation
- Prompt construction via `inference.build_prompt()`

## Configuration

All config is centralized in `model_qwen4_finetuning/config.yaml`. The server reads:
- `models[active_model]` — which GGUF to load
- `inference.*` — llama-server host/port, generation parameters
- `data.tool_config_path` — tool schemas injected into prompts
- `data.system_prompt` — system message
- `agent.*` — tool execution settings (timeouts, enable/disable)

## Dependencies

Server runtime: `fastapi`, `uvicorn`, `httpx`, `requests`, `pydantic`, `python-dotenv`, `pyyaml`

See `inference/requirements.txt` for the full list.
