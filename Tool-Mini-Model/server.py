"""
server.py -- FastAPI inference server with rotating API key auth.

Wraps the fine-tuned Qwen3 tool-calling model behind a single
authenticated endpoint. Automatically starts llama-server if it
is not already running.

Usage:
    1. Set INFERENCE_REFRESH_TOKEN in .env or environment
    2. python server.py
"""

import asyncio
import hashlib
import hmac
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

import requests as http_requests
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from inference.inference import (
    DEFAULT_CONFIG,
    build_prompt,
    clean_tool_artifacts,
    find_gguf,
    load_config,
    load_tools,
    parse_tool_calls,
    resolve_path,
    start_server,
    strip_thinking,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerationParams(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repeat_penalty: float = 1.0


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    enable_thinking: bool = True
    generation: GenerationParams | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float


class ChatResponse(BaseModel):
    content: str
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Authentication -- rotating API keys
# ---------------------------------------------------------------------------

REFRESH_TOKEN = os.environ.get("INFERENCE_REFRESH_TOKEN", "")
KEY_ROTATION_SECONDS = 300  # 5 minutes


def _derive_api_key(window_offset: int = 0) -> tuple[str, datetime]:
    """Derive a short-lived API key for a time window.

    Returns (hex_key, window_expiry_utc).
    window_offset=0 is the current window, -1 is the previous window.
    """
    window_index = int(time.time()) // KEY_ROTATION_SECONDS + window_offset
    digest = hmac.new(
        REFRESH_TOKEN.encode(), str(window_index).encode(), hashlib.sha256,
    ).hexdigest()
    expiry = datetime.fromtimestamp(
        (window_index + 1) * KEY_ROTATION_SECONDS, tz=timezone.utc,
    )
    return digest, expiry


async def _key_rotation_logger():
    """Background task that prints each new API key when the window rotates."""
    last_window = int(time.time()) // KEY_ROTATION_SECONDS
    while True:
        now = int(time.time())
        current_window = now // KEY_ROTATION_SECONDS
        next_rotation = (current_window + 1) * KEY_ROTATION_SECONDS
        await asyncio.sleep(next_rotation - now)

        new_window = int(time.time()) // KEY_ROTATION_SECONDS
        if new_window != last_window:
            api_key, expires_at = _derive_api_key(0)
            print(f"  [Key rotated] New API key: {api_key}")
            print(f"                Expires:     {expires_at.isoformat()}")
            last_window = new_window


async def verify_api_key(x_api_key: str = Header(...)):
    if not REFRESH_TOKEN:
        raise HTTPException(
            status_code=500, detail="INFERENCE_REFRESH_TOKEN not configured",
        )
    current_key, _ = _derive_api_key(0)
    previous_key, _ = _derive_api_key(-1)
    if hmac.compare_digest(x_api_key, current_key):
        return
    if hmac.compare_digest(x_api_key, previous_key):
        return
    raise HTTPException(status_code=403, detail="Invalid or expired API key")


class TokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    api_key: str
    expires_at: str


# ---------------------------------------------------------------------------
# Lifespan -- load config & verify llama-server on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    cfg = load_config(DEFAULT_CONFIG)
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    export_cfg = cfg["export"]

    host = infer_cfg["server_host"]
    port = infer_cfg["server_port"]

    tool_config_path = resolve_path(DEFAULT_CONFIG, data_cfg["tool_config_path"])
    tools = load_tools(tool_config_path)

    gen_params = {
        "max_tokens": infer_cfg.get("max_tokens", 512),
        "temperature": infer_cfg.get("temperature", 0.6),
        "top_p": infer_cfg.get("top_p", 0.95),
        "top_k": infer_cfg.get("top_k", 20),
        "min_p": infer_cfg.get("min_p", 0.0),
        "repeat_penalty": infer_cfg.get("repeat_penalty", 1.0),
    }

    # Check if llama-server is already running; if not, start it
    llama_proc = None
    llama_running = False
    try:
        resp = http_requests.get(f"http://{host}:{port}/health", timeout=3)
        if resp.status_code == 200:
            llama_running = True
            print(f"llama-server already running at http://{host}:{port}")
    except (http_requests.exceptions.ConnectionError, http_requests.exceptions.Timeout):
        pass

    if not llama_running:
        gguf_dir = resolve_path(DEFAULT_CONFIG, export_cfg["gguf_output_dir"])
        gguf_path = find_gguf(gguf_dir)
        if not gguf_path:
            print(f"ERROR: No GGUF file found in {gguf_dir}")
            print("Run inference/export_gguf.py first, or start llama-server manually.")
            sys.exit(1)

        llama_proc = start_server(
            llamacpp_path=export_cfg["llamacpp_path"],
            gguf_path=gguf_path,
            host=host,
            port=port,
            gpu_layers=infer_cfg["gpu_layers"],
            n_ctx=infer_cfg["n_ctx"],
            flash_attn=infer_cfg.get("flash_attn", False),
        )

    application.state.llama_host = host
    application.state.llama_port = port
    application.state.llama_proc = llama_proc
    application.state.system_prompt = data_cfg["system_prompt"]
    application.state.tools = tools
    application.state.gen_params = gen_params
    application.state.enable_thinking = infer_cfg.get("enable_thinking", True)

    print(f"FastAPI server ready -- llama-server at http://{host}:{port}")
    print(f"Loaded {len(tools)} tool schemas")

    if REFRESH_TOKEN:
        api_key, expires_at = _derive_api_key(0)
        print(f"  API key: {api_key}")
        print(f"  Expires: {expires_at.isoformat()}")

    rotation_task = asyncio.create_task(_key_rotation_logger())

    yield

    rotation_task.cancel()
    try:
        await rotation_task
    except asyncio.CancelledError:
        pass

    if llama_proc:
        print("Stopping llama-server...")
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except Exception:
            llama_proc.kill()
        print("llama-server stopped.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llama_server: Literal["reachable", "unreachable"]


@app.get("/health", response_model=HealthResponse)
async def health():
    state = app.state
    try:
        resp = http_requests.get(
            f"http://{state.llama_host}:{state.llama_port}/health", timeout=5,
        )
        llama_ok = resp.status_code == 200
    except (http_requests.exceptions.ConnectionError, http_requests.exceptions.Timeout):
        llama_ok = False

    return HealthResponse(
        status="ok" if llama_ok else "degraded",
        llama_server="reachable" if llama_ok else "unreachable",
    )


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------


@app.post("/auth/token", response_model=TokenResponse)
async def exchange_token(req: TokenRequest):
    if not REFRESH_TOKEN:
        raise HTTPException(
            status_code=500, detail="INFERENCE_REFRESH_TOKEN not configured",
        )
    if not hmac.compare_digest(req.refresh_token, REFRESH_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid refresh token")

    api_key, expires_at = _derive_api_key(0)
    return TokenResponse(api_key=api_key, expires_at=expires_at.isoformat())


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(req: ChatRequest):
    state = app.state
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    enable_thinking = req.enable_thinking and state.enable_thinking

    # Merge generation params: config defaults overridden by request
    gen = state.gen_params.copy()
    if req.generation:
        overrides = req.generation.model_dump(exclude_unset=True)
        gen.update(overrides)

    prompt = build_prompt(state.system_prompt, state.tools, messages, enable_thinking)

    # Non-streaming request to llama-server
    url = f"http://{state.llama_host}:{state.llama_port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": gen["max_tokens"],
        "temperature": gen["temperature"],
        "top_p": gen["top_p"],
        "top_k": gen["top_k"],
        "min_p": gen["min_p"],
        "repeat_penalty": gen["repeat_penalty"],
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": False,
    }

    try:
        resp = http_requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="llama-server unreachable")
    except http_requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="llama-server timed out")

    result = resp.json()
    raw_content = result.get("content", "")
    timings = result.get("timings", {})

    # Parse response
    thinking, response_text = strip_thinking(raw_content)
    tool_calls_raw = parse_tool_calls(response_text)
    content = clean_tool_artifacts(response_text)

    # Build validated response
    tool_calls = (
        [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in tool_calls_raw]
        if tool_calls_raw
        else None
    )

    usage = None
    if timings:
        usage = Usage(
            prompt_tokens=timings.get("prompt_n", 0),
            completion_tokens=timings.get("predicted_n", 0),
            tokens_per_second=timings.get("predicted_per_second", 0.0),
        )

    return ChatResponse(
        content=content,
        thinking=thinking or None,
        tool_calls=tool_calls,
        usage=usage,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
