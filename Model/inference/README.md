# Inference

Model export, interactive inference, validation, and API server for the fine-tuned Qwen3 tool-calling model.

## Scripts

| File | Purpose |
|------|---------|
| `export_gguf.py` | Merge LoRA adapter + convert to GGUF + quantize |
| `inference.py` | Interactive CLI chat with streaming + tool-call parsing |
| `validate.py` | Automated test suite (19 test cases) |
| `convert_hf_to_gguf.py` | HuggingFace-to-GGUF converter (from llama.cpp) |

The root-level `server.py` imports core functions from `inference.py` to run the FastAPI server (see [API Server](#api-server) below).

## Export

```bash
python export_gguf.py

# Override quantization type
python export_gguf.py --quant q8_0
```

Steps: merge LoRA weights into the base model (Unsloth), convert to f16 GGUF, quantize to target type (default: `q4_k_m`).

## Interactive Inference

```bash
# Auto-find GGUF and start llama-server
python inference.py

# Connect to an already-running server
python inference.py --server-running

# Specify a GGUF file directly
python inference.py --gguf path/to/model.gguf

# Custom port / disable thinking
python inference.py --port 9090 --no-think
```

Chat commands: `/reset`, `/system`, `/quit`

## Validation

```bash
python validate.py

# If llama-server is already running
python validate.py --server-running
```

Covers single tool calls, multi-tool calls, and no-tool conversational cases.

## API Server

`server.py` (at the project root) wraps the inference pipeline as a FastAPI service with rotating API key authentication.

```bash
# Set your refresh token in .env or environment
# INFERENCE_REFRESH_TOKEN=your-secret

python server.py
```

The server automatically starts llama-server if it is not already running, loads the model config and tool schemas, and exposes three endpoints:

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Server + llama-server status |
| `POST /auth/token` | No | Exchange refresh token for a 5-minute API key |
| `POST /v1/chat/completions` | Yes | Send conversation, receive model response |

### Authentication

The server uses a rotating API key that changes every 5 minutes:

1. Store `INFERENCE_REFRESH_TOKEN` in `.env` (permanent secret, never expires).
2. Call `POST /auth/token` with `{"refresh_token": "..."}` to get the current short-lived API key.
3. Pass the key in `X-API-Key` header for chat requests.
4. When the key expires (or you get a 403), call `/auth/token` again.

### Request format

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

Only `messages` is required. `enable_thinking` defaults to `true`. `generation` is optional (server defaults from `config.yaml` are used when omitted).

### Response format

```json
{
  "content": "I'll check the weather for you.",
  "thinking": "The user wants tomorrow's weather...",
  "tool_calls": [
    {"name": "weather", "arguments": {"specific_date": "tomorrow"}}
  ],
  "usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 67,
    "tokens_per_second": 48.3
  }
}
```

`content` is always present. `thinking`, `tool_calls`, and `usage` are null when not applicable.

## Prerequisites

- Trained LoRA adapter at `output/lora_adapter/` (for export)
- llama.cpp installed with `llama-server` and `llama-quantize` binaries
- NVIDIA GPU recommended for inference (CPU works but slow)

## Output

| Artifact | Location | Description |
|----------|----------|-------------|
| Merged model | `output/merged_model/` | Full-precision merged weights |
| GGUF model | `output/gguf/` | Quantized model for llama.cpp |

## Dependencies

```bash
pip install -r requirements.txt
```

Export requires `unsloth` and `transformers`. Inference requires `requests` and `pyyaml`. The API server additionally requires `fastapi`, `uvicorn`, and `python-dotenv`.

## Config

Reads from the shared `config.yaml` at the project root (`model_qwen4_finetuning/config.yaml`):

- `model.*` -- model name, sequence length
- `export.*` -- llama.cpp path, output directories, quantization type
- `inference.*` -- server settings, generation parameters
- `data.tool_config_path` -- tool schemas for prompts
- `data.system_prompt` -- system message
