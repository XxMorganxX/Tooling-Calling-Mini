"""Pydantic models mirroring the inference API request/response schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Authentication models ─────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    """Body for ``POST /auth/token``."""

    refresh_token: str


class TokenResponse(BaseModel):
    """Response from ``POST /auth/token``."""

    api_key: str
    expires_at: datetime


# ── Request models ────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class GenerationParams(BaseModel):
    """Optional sampling overrides sent to the server.

    Any field left as *None* tells the server to use its own default.
    """

    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate.")
    temperature: Optional[float] = Field(None, ge=0.0, description="Sampling temperature.")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling cutoff.")
    top_k: Optional[int] = Field(None, ge=0, description="Top-k token sampling limit.")
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum probability threshold.")
    repeat_penalty: Optional[float] = Field(None, ge=0.0, description="Repetition penalty (1.0 = disabled).")


class ChatRequest(BaseModel):
    """Body for ``POST /v1/chat/completions``."""

    messages: list[ChatMessage]
    enable_thinking: Optional[bool] = True
    generation: Optional[GenerationParams] = None


# ── Response models ───────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A single tool invocation returned by the model."""

    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """Token-usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float


class ChatResponse(BaseModel):
    """Parsed response from ``POST /v1/chat/completions``."""

    content: str
    thinking: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    usage: Optional[Usage] = None


class HealthResponse(BaseModel):
    """Response from ``GET /health``."""

    status: Literal["ok", "degraded"]
    llama_server: Literal["reachable", "unreachable"]
