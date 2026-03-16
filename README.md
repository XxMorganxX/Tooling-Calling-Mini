# Tool Calling Mini

A system for fine-tuning, serving, and interacting with Qwen3 models specialized in tool calling. The **server** handles model training, GGUF export, authenticated inference via llama.cpp, and server-side tool execution. Two **clients** provide terminal and web chat interfaces. A **test suite** validates tool-calling accuracy against training data.

```
Tool-Calling-Mini/
├── Model/                    ← Server: inference API + tool execution + training pipeline
├── Client-CLI/               ← Client: interactive terminal chat
├── Client-Web/               ← Client: web-based chat interface
├── Inference-Test-Suite/     ← Statistical accuracy validation
├── scripts/                  ← Maintenance scripts (spec generation, git hooks)
└── homeassist-ref/           ← Git submodule: source of truth for tool definitions
```

---

## Architecture

```
┌──────────────────────────┐  ┌──────────────────────────┐
│    Client-CLI             │  │    Client-Web             │
│    (main.py)              │  │    React (:5173)          │
│    Rich terminal REPL     │  │    FastAPI proxy (:8001)  │
└────────┬─────────────────┘  └────────┬─────────────────┘
         │  HTTP + X-API-Key           │  HTTP + cookies
         └──────────┬──────────────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│    Model/server.py          FastAPI gateway (:8000)   │
│    Auth, prompt building, response parsing            │
│    Server-side tool execution (agent loop)            │
└────────┬─────────────────────────────────────────────┘
         │  localhost HTTP
         ▼
┌──────────────────────────┐
│    llama-server           │  llama.cpp inference (:8080)
│    (GGUF model on GPU)    │  Token generation, GPU offload
└──────────────────────────┘
```

---

## Server — `Model/`

### What It Does

1. **Fine-tunes** Qwen3 models on ~498 tool-calling examples using QLoRA via Unsloth
2. **Exports** the trained LoRA adapter merged into a quantized GGUF model
3. **Serves** the model through a FastAPI API backed by llama.cpp for GPU-accelerated inference
4. **Executes tools** server-side via an agent loop (weather, calendar, Spotify, etc.) and passes client-side tool calls through for local execution
5. **Authenticates** clients with HMAC-SHA256 rotating API keys (5-minute windows)

### Structure

```
Model/
├── server.py                           # FastAPI inference server (entry point)
├── agent.py                            # Multi-turn agent loop (tool call → result → re-prompt)
├── context.py                          # Token-aware context manager for inference
├── .env                                # INFERENCE_REFRESH_TOKEN
├── inference/
│   ├── inference.py                    # Prompt building, tool-call parsing, llama-server mgmt
│   ├── export_gguf.py                  # LoRA merge + GGUF conversion
│   └── validate.py                     # Automated test suite (19 hardcoded cases)
├── model_qwen4_finetuning/
│   ├── config.yaml                     # Single source of truth for all config
│   ├── run.py                          # Full pipeline runner
│   ├── tool_calling_config.json        # 12 tool schemas
│   ├── data/
│   │   ├── prepare_dataset.py          # JSONL → tokenized parquet
│   │   └── training_data.jsonl         # Raw training examples
│   ├── training/
│   │   └── train.py                    # QLoRA fine-tuning script
│   └── output/                         # Trained weights, merged model, GGUF files
├── tools/
│   ├── models.py                       # Pydantic validation for all 12 tools
│   ├── executor.py                     # ToolExecutor — dispatch with timeout handling
│   ├── weather.py, spotify.py, ...     # Individual tool implementations
│   └── system_info.py                  # Internal system architecture docs
├── clients/
│   ├── weather_client.py               # Open-Meteo API client
│   ├── calendar_client.py              # Google Calendar API client
│   ├── kasa_lighting_client.py         # TP-Link Kasa LAN client
│   └── web_search_client.py            # Google Custom Search client
└── tool_calling_sdk/
    ├── __init__.py                     # Exports ContextManager, LocalToolExecutor
    ├── context.py                      # Shared conversation context management
    └── executor.py                     # Client-side tool executor for local tools
```

### Model Registry

The server supports multiple model profiles via `config.yaml`. The active model is selected by the `active_model` key.

| Profile | Model | Quantization |
|---|---|---|
| `qwen3-4b` | Qwen3-4B-Instruct-2507 | Q4_K_M |
| `qwen3-8b` (active) | Qwen3-8B | Q4_0 |

### API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check (FastAPI + llama-server status) |
| `/auth/token` | POST | No | Exchange refresh token for a 5-minute API key |
| `/v1/chat/completions` | POST | Yes | Chat completions with tool-call support |
| `/v1/chat/completions/stream` | POST | Yes | Streaming chat via Server-Sent Events |
| `/tools` | GET | Yes | List all registered tool schemas |
| `/models` | GET | No | List available models and active selection |

### Server-Side vs Client-Side Tools

Tools are split between server and client execution:

| Executed on Server | Passed Through to Client |
|---|---|
| `weather`, `spotify_playback`, `calendar_data`, `google_search`, `briefing`, `get_notifications`, `system_info` | `stickies`, `clipboard`, `send_sms`, `cursor_composer`, `kasa_lighting` |

Server-side tools are executed automatically after the model produces tool calls. Client-side tools are returned in `tool_calls` for the client to execute locally (they require macOS apps, LAN devices, or filesystem access).

### Running the Server

```bash
cd Model

# Set the refresh token
# .env: INFERENCE_REFRESH_TOKEN="your-secret"

pip install -r inference/requirements.txt

# Start the server (auto-launches llama-server if not running)
python server.py
# → FastAPI on http://0.0.0.0:8000
# → llama-server on http://127.0.0.1:8080
```

### Training Pipeline

```bash
cd Model/model_qwen4_finetuning

pip install -r requirements.txt

# Run the full pipeline: prepare data → train → export → validate
python run.py

# Or run specific stages
python run.py --start train       # Skip data prep
python run.py --stop export       # Stop before validation
```

### Tech Stack (Server)

- **FastAPI** + **Uvicorn** — API server
- **llama.cpp** — GPU-accelerated inference engine
- **Unsloth** — QLoRA training
- **HuggingFace Transformers** — Model loading and tokenization
- **Pydantic** — Request/response and tool-call validation
- **httpx** — Async HTTP for streaming proxied responses

---

## Client — `Client-CLI/`

### What It Does

1. **Interactive terminal chat** with Rich-formatted output (thinking traces, tool-call tables, markdown responses)
2. **Conversation management** with automatic history truncation to stay within the 4096-token context window
3. **Transparent authentication** with automatic key refresh on expiry
4. **Importable SDK** for programmatic access to the inference API

### Structure

```
Client-CLI/
├── main.py                 # Interactive terminal REPL (entry point)
├── config.yaml             # Server URL, generation params, conversation settings
├── .env                    # INFERENCE_REFRESH_TOKEN
├── requirements.txt        # Python dependencies
└── src/
    ├── __init__.py         # Package exports
    ├── client.py           # InferenceClient — HTTP client with auth
    ├── config.py           # YAML + .env config loader
    ├── models.py           # Pydantic request/response models
    └── conversation.py     # ConversationManager — history tracking + truncation
```

### Running the Client

```bash
cd Client-CLI

pip install -r requirements.txt

# Set the same refresh token as the server
# .env: INFERENCE_REFRESH_TOKEN="your-secret"

python main.py
```

### Slash Commands

| Command | Description |
|---|---|
| `/clear` | Clear conversation history |
| `/thinking` | Toggle thinking mode on/off |
| `/config` | Show current configuration |
| `/help` | Show available commands |
| `/quit` | Exit the client |

### Programmatic Usage

```python
from src import load_config, InferenceClient, ConversationManager

config = load_config()
client = InferenceClient(config)
manager = ConversationManager(client, config)

response = manager.send("What's the weather tomorrow?")
print(response.content)

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")
```

### Tech Stack (Client-CLI)

- **Requests** — HTTP client
- **Rich** — Terminal UI (panels, tables, markdown, spinners)
- **Pydantic** — Data validation
- **PyYAML** + **python-dotenv** — Configuration

---

## Client — `Client-Web/`

### What It Does

1. **Web chat interface** with streaming responses via Server-Sent Events
2. **Backend proxy** that manages sessions, conversation history, and auth with the inference server
3. **Settings panel** for adjusting generation parameters and switching models
4. **Training data approval** workflow for curating good prompt/response pairs

### Structure

```
Client-Web/
├── backend/
│   └── main.py             # FastAPI proxy (:8001) — sessions, config, model switching
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main chat component — rendering, streaming, settings
│   │   ├── api.ts           # API client functions
│   │   ├── types.ts         # TypeScript interfaces for API types
│   │   └── main.tsx         # React entry point
│   └── vite.config.ts       # Vite config with backend proxy
├── config.yaml              # Backend configuration
└── data/
    └── approved_samples.jsonl  # Approved training data from the UI
```

### Running

```bash
# Terminal 1: Backend
cd Client-Web/backend
pip install -r requirements.txt
python main.py                    # → http://localhost:8001

# Terminal 2: Frontend (dev)
cd Client-Web/frontend
npm install && npm run dev        # → http://localhost:5173
```

### Tech Stack (Client-Web)

- **React** + **TypeScript** + **Vite** — Frontend
- **FastAPI** — Backend proxy with session management

---

## Inference Test Suite — `Inference-Test-Suite/`

Validates tool-calling accuracy by sampling from the training dataset, querying the model, and comparing predicted tool calls against ground truth.

### Accuracy Tiers

| Tier | What It Checks |
|---|---|
| **Strict** | Correct tool names AND matching argument values |
| **Routing** | Correct tool selected (ignoring argument values) |
| **Format** | Valid `{"name": ..., "arguments": ...}` JSON structure |

### Running

```bash
cd Inference-Test-Suite
pip install -r requirements.txt

python run.py                                    # 25 random samples
python run.py --samples 50 --seed 42 --verbose   # Reproducible with detail
```

Requires a running llama-server instance (talks directly to `/completion`, not the FastAPI server).

---

## Supported Tools

The model is fine-tuned to call 12 tools:

| Tool | Description |
|---|---|
| `weather` | Weather forecast (hourly/daily, up to 7 days) |
| `spotify_playback` | Spotify control (play, pause, volume, search, shuffle, repeat) |
| `kasa_lighting` | Kasa smart light control (on/off, scenes) |
| `calendar_data` | Google Calendar (read events, create events) |
| `stickies` | Desktop sticky notes (read/edit) |
| `send_sms` | Send iMessage/text to phone |
| `google_search` | Web search |
| `clipboard` | Read system clipboard contents |
| `briefing` | Spoken briefing announcements |
| `get_notifications` | Check pending notifications (email, news) |
| `system_info` | Internal system architecture docs |
| `cursor_composer` | Send coding tasks to Cursor Composer |

---

## Authentication Flow

Both the server and clients share a permanent **refresh token** (`INFERENCE_REFRESH_TOKEN`). The auth flow works as follows:

1. Client sends `POST /auth/token` with the refresh token
2. Server derives a short-lived API key using HMAC-SHA256 over a 5-minute time window
3. Client includes the key in `X-API-Key` header on subsequent requests
4. Keys rotate every 5 minutes; the server accepts both the current and previous window for grace

---

## Configuration

### Server (`Model/model_qwen4_finetuning/config.yaml`)

Central config for model registry, LoRA parameters, training hyperparameters, export settings, inference/generation defaults, and agent tool execution settings. All paths are relative to this file.

### Client-CLI (`Client-CLI/config.yaml`)

```yaml
server:
  url: "http://localhost:8000"
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
  max_history_messages: 10
```

---

## Scripts — `scripts/`

| Script | Purpose |
|---|---|
| `generate_openclaw_spec.py` | Regenerates `INTEGRATION_API.md` from `tool_calling_config.json` |
| `install-hooks.sh` | Installs git hooks (run once after cloning) |
| `pre-commit` | Auto-regenerates the integration spec on relevant commits |

---

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA (recommended for inference; CPU works but is slow)
- **llama.cpp** installed (provides `llama-server` and `llama-quantize`)
- For training: **16+ GB VRAM** (QLoRA with 4-bit base weights)
