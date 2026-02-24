"""
FastAPI backend for the Tool Mini-Model web client.

Acts as a proxy between the React frontend and the inference API server,
handling authentication, conversation history, and config management.

Uses cookie-based sessions so multiple concurrent users each get their
own conversation state.  Refresh tokens are never stored on disk.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
# Training data approval config
# ---------------------------------------------------------------------------

_TOOL_CFG_PATH = _ROOT.parent / "Model" / "model_qwen4_finetuning" / "tool_calling_config.json"
_APPROVED_DIR = _ROOT / "data"
_APPROVED_FILE = _APPROVED_DIR / "approved_samples.jsonl"

_TOOL_CFG: dict[str, Any] | None = None
if _TOOL_CFG_PATH.exists():
    with open(_TOOL_CFG_PATH, encoding="utf-8") as f:
        _TOOL_CFG = json.load(f)

_FULL_SYSTEM_PROMPT: str | None = None
if _TOOL_CFG:
    _FULL_SYSTEM_PROMPT = _TOOL_CFG["system_prompt"].replace(
        "{tools}", json.dumps(_TOOL_CFG["tools"], indent=2),
    )

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


class ApproveRequest(BaseModel):
    prompt: str
    thinking: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

SESSION_COOKIE = "tmm_session"


@dataclass
class Session:
    refresh_token: str | None = None
    api_key: str | None = None
    key_expires: datetime | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    enable_thinking: bool = bool(CFG["model"]["enable_thinking"])
    generation: dict[str, Any] = field(default_factory=lambda: dict(CFG["generation"]))


_sessions: dict[str, Session] = {}


def _get_or_create_session(session_id: str | None, response: Response) -> tuple[str, Session]:
    """Return (session_id, session), creating a new one if necessary."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = secrets.token_urlsafe(32)
    session = Session()
    _sessions[sid] = session
    response.set_cookie(
        SESSION_COOKIE,
        sid,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return sid, session


def _require_session(session_id: str | None) -> Session:
    """Return an existing session or raise 401."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    raise HTTPException(status_code=401, detail="No active session")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_http = httpx.AsyncClient(timeout=TIMEOUT)


def _upstream_unreachable_detail(exc: Exception) -> str:
    return f"Cannot reach upstream inference server at {SERVER_URL}: {exc}"


async def _authenticate(session: Session) -> None:
    if not session.refresh_token:
        raise HTTPException(status_code=401, detail="Not connected -- provide a refresh token first")
    try:
        resp = await _http.post(
            f"{SERVER_URL}/auth/token",
            json={"refresh_token": session.refresh_token},
            timeout=10,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=_upstream_unreachable_detail(exc)) from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Authentication failed")
    data = resp.json()
    session.api_key = data["api_key"]
    session.key_expires = datetime.fromisoformat(data["expires_at"])


async def _ensure_key(session: Session) -> str:
    if (
        session.api_key is None
        or session.key_expires is None
        or datetime.now(timezone.utc) >= session.key_expires
    ):
        await _authenticate(session)
    assert session.api_key is not None
    return session.api_key


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Tool Mini-Model Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_html_cache(request: Request, call_next):
    response = await call_next(request)

    # Keep API behavior unchanged; only force fresh HTML shell fetches.
    if not request.url.path.startswith("/api/"):
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

    return response

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
async def auth_status(response: Response, tmm_session: str | None = Cookie(None)):
    if tmm_session and tmm_session in _sessions:
        session = _sessions[tmm_session]
        return {"authenticated": session.refresh_token is not None}
    return {"authenticated": False}


@app.post("/api/auth/connect")
async def connect(req: ConnectRequest, response: Response, tmm_session: str | None = Cookie(None)):
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
    sid, session = _get_or_create_session(tmm_session, response)
    session.refresh_token = req.refresh_token
    session.api_key = data["api_key"]
    session.key_expires = datetime.fromisoformat(data["expires_at"])

    return {"status": "ok", "expires_at": data["expires_at"]}


@app.post("/api/auth/disconnect")
async def disconnect(response: Response, tmm_session: str | None = Cookie(None)):
    if tmm_session and tmm_session in _sessions:
        del _sessions[tmm_session]
    response.delete_cookie(SESSION_COOKIE)
    return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    """Proxy model list from the inference server."""
    try:
        resp = await _http.get(f"{SERVER_URL}/models", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except httpx.RequestError:
        pass
    return {"active": None, "active_display_name": None, "models": []}


@app.get("/api/config")
async def get_config(tmm_session: str | None = Cookie(None)):
    session = _require_session(tmm_session)
    model_info: dict[str, Any] = {"active": None, "active_display_name": None, "models": []}
    try:
        resp = await _http.get(f"{SERVER_URL}/models", timeout=10)
        if resp.status_code == 200:
            model_info = resp.json()
    except httpx.RequestError:
        pass
    return {
        "server_url": SERVER_URL,
        "enable_thinking": session.enable_thinking,
        "generation": session.generation,
        "max_history_messages": CFG["conversation"]["max_history_messages"],
        "active_model": model_info.get("active"),
        "active_model_display_name": model_info.get("active_display_name"),
        "models": model_info.get("models", []),
    }


@app.patch("/api/config")
async def update_config(update: ConfigUpdate, tmm_session: str | None = Cookie(None)):
    session = _require_session(tmm_session)
    if update.enable_thinking is not None:
        session.enable_thinking = update.enable_thinking
    gen = session.generation
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
    return {
        "server_url": SERVER_URL,
        "enable_thinking": session.enable_thinking,
        "generation": session.generation,
        "max_history_messages": CFG["conversation"]["max_history_messages"],
    }


@app.post("/api/chat")
async def chat(msg: UserMessage, tmm_session: str | None = Cookie(None)):
    session = _require_session(tmm_session)
    if not session.refresh_token:
        raise HTTPException(status_code=401, detail="Not connected")

    session.history.append({"role": "user", "content": msg.content})

    max_msgs = int(CFG["conversation"]["max_history_messages"])
    if max_msgs > 0 and len(session.history) > max_msgs:
        overflow = len(session.history) - max_msgs
        if overflow % 2 != 0:
            overflow += 1
        session.history = session.history[overflow:]

    key = await _ensure_key(session)

    payload = {
        "messages": session.history,
        "enable_thinking": session.enable_thinking,
        "execute_tools": True,
        "generation": session.generation,
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
        await _authenticate(session)
        key = session.api_key
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

    session.history.append({"role": "assistant", "content": data.get("content", "")})

    return data


@app.post("/api/chat/stream")
async def chat_stream(msg: UserMessage, tmm_session: str | None = Cookie(None)):
    """Streaming chat endpoint -- proxies SSE from the inference server."""
    session = _require_session(tmm_session)
    if not session.refresh_token:
        raise HTTPException(status_code=401, detail="Not connected")

    session.history.append({"role": "user", "content": msg.content})

    max_msgs = int(CFG["conversation"]["max_history_messages"])
    if max_msgs > 0 and len(session.history) > max_msgs:
        overflow = len(session.history) - max_msgs
        if overflow % 2 != 0:
            overflow += 1
        session.history = session.history[overflow:]

    key = await _ensure_key(session)

    payload = {
        "messages": session.history,
        "enable_thinking": session.enable_thinking,
        "execute_tools": True,
        "generation": session.generation,
    }

    async def proxy_stream() -> AsyncGenerator[bytes, None]:
        import json as _json

        async with _http.stream(
            "POST",
            f"{SERVER_URL}/v1/chat/completions/stream",
            json=payload,
            headers={"X-API-Key": key},
            timeout=TIMEOUT,
        ) as resp:
            final_content = ""
            current_event = ""

            async for line in resp.aiter_lines():
                yield (line + "\n").encode()

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: ") and current_event == "done":
                    try:
                        data = _json.loads(line[6:])
                        final_content = data.get("content", "")
                    except (ValueError, KeyError):
                        pass
                    current_event = ""
                elif not line.strip():
                    current_event = ""

            if final_content:
                session.history.append({"role": "assistant", "content": final_content})

    return StreamingResponse(
        proxy_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/training/approve")
async def approve_training(req: ApproveRequest, tmm_session: str | None = Cookie(None)):
    _require_session(tmm_session)
    if not _FULL_SYSTEM_PROMPT:
        raise HTTPException(
            status_code=500,
            detail="Tool config not loaded -- cannot save training data",
        )

    # Reconstruct the assistant content in training-data format
    assistant_content = ""
    if req.thinking:
        assistant_content = f"<think>\n{req.thinking}\n</think>\n"

    if req.tool_calls:
        assistant_content += json.dumps(req.tool_calls, separators=(",", ":"))
    else:
        assistant_content += "[]"

    entry = {
        "messages": [
            {"role": "system", "content": _FULL_SYSTEM_PROMPT},
            {"role": "user", "content": req.prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }

    _APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(_APPROVED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "ok", "file": str(_APPROVED_FILE)}


@app.post("/api/clear")
async def clear(tmm_session: str | None = Cookie(None)):
    session = _require_session(tmm_session)
    session.history = []
    return {"status": "ok"}


@app.get("/api/history")
async def get_history(tmm_session: str | None = Cookie(None)):
    session = _require_session(tmm_session)
    return {"messages": session.history}


# ---------------------------------------------------------------------------
# Serve frontend in production
# ---------------------------------------------------------------------------

_dist = _ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
