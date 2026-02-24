# Tool Calling Mini

A two-part system for fine-tuning, hosting, and interacting with a Qwen3-4B model specialized in tool calling. The **server** handles model training, GGUF export, and authenticated inference via llama.cpp. The **client** provides a terminal chat interface with conversation management and Rich-formatted output.

```
Tool Calling Mini/
├── Model/                ← Server: training pipeline + inference API
├── Client-CLI/           ← Client: interactive terminal chat
└── Client-Web/           ← Client: web-based chat interface
```

---

## Architecture

```
┌──────────────────────────┐
│    Client-CLI             │  Python terminal REPL
│    (main.py)              │  Rich UI, conversation history
└────────┬─────────────────┘
         │  HTTP + X-API-Key
         ▼
┌──────────────────────────┐
│    Model                  │  FastAPI gateway (:8000)
│    (server.py)            │  Auth, prompt building, response parsing
└────────┬─────────────────┘
         │  localhost HTTP
         ▼
┌──────────────────────────┐
│    llama-server           │  llama.cpp inference engine (:8080)
│    (GGUF model on GPU)    │  Token generation, GPU offload
└──────────────────────────┘
```

---

## Server — `Model/`

### What It Does

1. **Fine-tunes** Qwen3-4B-Instruct on ~498 tool-calling examples using QLoRA via Unsloth
2. **Exports** the trained LoRA adapter merged into a quantized GGUF model (Q4_K_M)
3. **Serves** the model through a FastAPI API backed by llama.cpp for GPU-accelerated inference
4. **Authenticates** clients with HMAC-SHA256 rotating API keys (5-minute windows)

### Structure

```
Model/
├── server.py                       # FastAPI inference server (entry point)
├── .env                            # INFERENCE_REFRESH_TOKEN
├── inference/
│   ├── inference.py                # Core functions: prompt building, tool-call parsing
│   ├── export_gguf.py              # LoRA merge + GGUF conversion
│   ├── validate.py                 # Automated test suite (19 cases)
│   └── convert_hf_to_gguf.py      # HuggingFace → GGUF converter
└── model_qwen4_finetuning/
    ├── config.yaml                 # Single source of truth for all config
    ├── run.py                      # Full pipeline runner
    ├── tool_calling_config.json    # 12 tool schemas
    ├── data/
    │   ├── prepare_dataset.py      # JSONL → tokenized parquet
    │   └── training_data.jsonl     # Raw training examples
    ├── training/
    │   └── train.py                # QLoRA fine-tuning script
    └── output/
        ├── lora_adapter/           # Trained LoRA weights + checkpoints
        ├── merged_model/           # Full merged model
        └── gguf/                   # Quantized GGUF files
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check (FastAPI + llama-server status) |
| `/auth/token` | POST | No | Exchange refresh token for a 5-minute API key |
| `/v1/chat/completions` | POST | Yes | Chat completions with tool-call support |

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
- **Pydantic** — Request/response validation

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

### Tech Stack (Client)

- **Requests** — HTTP client
- **Rich** — Terminal UI (panels, tables, markdown, spinners)
- **Pydantic** — Data validation
- **PyYAML** + **python-dotenv** — Configuration

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
| `read_clipboard` | Read system clipboard contents |
| `briefing` | Spoken briefing announcements |
| `get_notifications` | Check pending notifications (email, news) |
| `system_info` | Internal system architecture docs |
| `cursor_composer` | Send coding tasks to Cursor Composer |

---

## Authentication Flow

Both the server and client share a permanent **refresh token** (`INFERENCE_REFRESH_TOKEN`). The auth flow works as follows:

1. Client sends `POST /auth/token` with the refresh token
2. Server derives a short-lived API key using HMAC-SHA256 over a 5-minute time window
3. Client includes the key in `X-API-Key` header on subsequent requests
4. Keys rotate every 5 minutes; the server accepts both the current and previous window for grace

---

## Configuration

### Server (`Model/model_qwen4_finetuning/config.yaml`)

Central config for model selection, LoRA parameters, training hyperparameters, export settings, and inference/generation defaults. All paths are relative to this file.

### Client (`Client-CLI/config.yaml`)

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

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA (recommended for inference; CPU works but is slow)
- **llama.cpp** installed (provides `llama-server` and `llama-quantize`)
- For training: **16+ GB VRAM** (QLoRA with 4-bit base weights)
