"""
FastAPI backend for the Tool Mini-Model web client.

Acts as a proxy between the React frontend and the inference API server,
handling authentication, conversation history, and config management.

The refresh token is never stored on disk -- it is provided by the user
through the web interface and held only in process memory.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _load_config() -> dict[str, Any]:
    cfg_path = _ROOT / "config.yaml"
    defaults = {
        "server": {"url": "http://localhost:8000", "timeout": 120},
        "model": {"enable_thinking": True},
        "generation": {
            "max_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
        },
        "conversation": {"max_history_messages": 10},
    }
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section, vals in defaults.items():
            if section in raw and isinstance(vals, dict):
                defaults[section] = {**vals, **raw[section]}
            elif section in raw:
                defaults[section] = raw[section]
    return defaults


CFG = _load_config()
SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", CFG["server"]["url"]).rstrip("/")
TIMEOUT = int(CFG["server"]["timeout"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ConnectRequest(BaseModel):
    refresh_token: str


class UserMessage(BaseModel):
    content: str


class ConfigUpdate(BaseModel):
    enable_thinking: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# In-memory state (never written to disk)
# ---------------------------------------------------------------------------

_refresh_token: str | None = None
_api_key: str | None = None
_key_expires: datetime | None = None
_history: list[dict[str, str]] = []
_enable_thinking: bool = bool(CFG["model"]["enable_thinking"])

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_http = httpx.AsyncClient(timeout=TIMEOUT)


def _upstream_unreachable_detail(exc: Exception) -> str:
    return f"Cannot reach upstream inference server at {SERVER_URL}: {exc}"


async def _authenticate() -> None:
    global _api_key, _key_expires
    if not _refresh_token:
        raise HTTPException(status_code=401, detail="Not connected -- provide a refresh token first")
    try:
        resp = await _http.post(
            f"{SERVER_URL}/auth/token",
            json={"refresh_token": _refresh_token},
            timeout=10,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=_upstream_unreachable_detail(exc)) from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Authentication failed")
    data = resp.json()
    _api_key = data["api_key"]
    _key_expires = datetime.fromisoformat(data["expires_at"])


async def _ensure_key() -> str:
    global _api_key, _key_expires
    if _api_key is None or _key_expires is None or datetime.now(timezone.utc) >= _key_expires:
        await _authenticate()
    assert _api_key is not None
    return _api_key


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Tool Mini-Model Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    try:
        resp = await _http.get(f"{SERVER_URL}/health", timeout=10)
        return resp.json()
    except httpx.RequestError as exc:
        return {"status": "error", "detail": _upstream_unreachable_detail(exc)}


@app.get("/api/auth/status")
async def auth_status():
    """Check whether the backend currently holds a refresh token."""
    return {"authenticated": _refresh_token is not None}


@app.post("/api/auth/connect")
async def connect(req: ConnectRequest):
    """Accept a refresh token from the UI and verify it against the server."""
    global _refresh_token, _api_key, _key_expires

    try:
        resp = await _http.post(
            f"{SERVER_URL}/auth/token",
            json={"refresh_token": req.refresh_token},
            timeout=10,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=_upstream_unreachable_detail(exc)) from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=403, detail="Invalid refresh token")

    data = resp.json()
    _refresh_token = req.refresh_token
    _api_key = data["api_key"]
    _key_expires = datetime.fromisoformat(data["expires_at"])

    return {"status": "ok", "expires_at": data["expires_at"]}


@app.post("/api/auth/disconnect")
async def disconnect():
    """Clear the in-memory token and conversation history."""
    global _refresh_token, _api_key, _key_expires, _history
    _refresh_token = None
    _api_key = None
    _key_expires = None
    _history = []
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    return {
        "server_url": SERVER_URL,
        "enable_thinking": _enable_thinking,
        "generation": CFG["generation"],
        "max_history_messages": CFG["conversation"]["max_history_messages"],
    }


@app.patch("/api/config")
async def update_config(update: ConfigUpdate):
    global _enable_thinking
    if update.enable_thinking is not None:
        _enable_thinking = update.enable_thinking
    gen = CFG["generation"]
    if update.max_tokens is not None:
        gen["max_tokens"] = update.max_tokens
    if update.temperature is not None:
        gen["temperature"] = update.temperature
    if update.top_p is not None:
        gen["top_p"] = update.top_p
    if update.top_k is not None:
        gen["top_k"] = update.top_k
    if update.min_p is not None:
        gen["min_p"] = update.min_p
    if update.repeat_penalty is not None:
        gen["repeat_penalty"] = update.repeat_penalty
    return await get_config()


@app.post("/api/chat")
async def chat(msg: UserMessage):
    global _history

    if not _refresh_token:
        raise HTTPException(status_code=401, detail="Not connected")

    _history.append({"role": "user", "content": msg.content})

    max_msgs = int(CFG["conversation"]["max_history_messages"])
    if max_msgs > 0 and len(_history) > max_msgs:
        overflow = len(_history) - max_msgs
        if overflow % 2 != 0:
            overflow += 1
        _history = _history[overflow:]

    key = await _ensure_key()

    payload = {
        "messages": _history,
        "enable_thinking": _enable_thinking,
        "generation": CFG["generation"],
    }

    try:
        resp = await _http.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            headers={"X-API-Key": key},
            timeout=TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=_upstream_unreachable_detail(exc)) from exc

    if resp.status_code == 403:
        await _authenticate()
        key = _api_key
        try:
            resp = await _http.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                headers={"X-API-Key": key},
                timeout=TIMEOUT,
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=_upstream_unreachable_detail(exc)) from exc

    if resp.status_code != 200:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()

    _history.append({"role": "assistant", "content": data.get("content", "")})

    return data


@app.post("/api/clear")
async def clear():
    global _history
    _history = []
    return {"status": "ok"}


@app.get("/api/history")
async def get_history():
    return {"messages": _history}


# ---------------------------------------------------------------------------
# Serve frontend in production
# ---------------------------------------------------------------------------

_dist = _ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
