"""HTTP client for the Qwen3-4B tool-calling inference API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

from .config import ClientConfig
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerationParams,
    HealthResponse,
    TokenRequest,
    TokenResponse,
)


# ── Custom exceptions ─────────────────────────────────────────────────────────


class APIError(Exception):
    """Base class for inference-API errors."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AuthenticationError(APIError):
    """Raised on 403 -- invalid/expired API key or invalid refresh token."""


class ValidationError(APIError):
    """Raised on 422 -- malformed request."""


class ServerError(APIError):
    """Raised on 500 -- server misconfiguration."""


class BadGatewayError(APIError):
    """Raised on 502 -- llama-server unreachable."""


class GatewayTimeoutError(APIError):
    """Raised on 504 -- llama-server timed out."""


_STATUS_EXCEPTION_MAP: dict[int, type[APIError]] = {
    403: AuthenticationError,
    422: ValidationError,
    500: ServerError,
    502: BadGatewayError,
    504: GatewayTimeoutError,
}


# ── Client ────────────────────────────────────────────────────────────────────


class InferenceClient:
    """Synchronous client for the inference API with rotating-key auth.

    On construction the client does **not** authenticate automatically.
    Call :meth:`authenticate` explicitly, or let :meth:`chat` handle it
    transparently (it calls :meth:`_ensure_valid_key` before every request
    and retries once on ``403``).

    Parameters
    ----------
    config:
        A :class:`ClientConfig` instance (usually built by :func:`load_config`).
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._base_url = config.server_url

        self._api_key: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    # ── Authentication ────────────────────────────────────────────────────

    def authenticate(self) -> TokenResponse:
        """Exchange the refresh token for a short-lived API key.

        The key is stored internally and used for subsequent requests.
        Returns the :class:`TokenResponse` so callers can inspect the
        expiry if needed.
        """
        body = TokenRequest(refresh_token=self._config.refresh_token)

        resp = self._session.post(
            f"{self._base_url}/auth/token",
            json=body.model_dump(),
            timeout=min(self._config.timeout, 10),
        )
        self._raise_for_status(resp)

        token = TokenResponse.model_validate(resp.json())
        self._api_key = token.api_key
        self._expires_at = token.expires_at
        return token

    @property
    def is_authenticated(self) -> bool:
        """True if we hold a key that has not yet expired."""
        if self._api_key is None or self._expires_at is None:
            return False
        return datetime.now(timezone.utc) < self._expires_at

    def _ensure_valid_key(self) -> str:
        """Return a valid API key, refreshing if necessary."""
        if not self.is_authenticated:
            self.authenticate()
        assert self._api_key is not None
        return self._api_key

    def _auth_header(self) -> dict[str, str]:
        """Build an ``X-API-Key`` header with the current key."""
        return {"X-API-Key": self._ensure_valid_key()}

    # ── Public API ────────────────────────────────────────────────────────

    def health(self) -> HealthResponse:
        """Check server and llama-server health (no auth required)."""
        resp = self._session.get(
            f"{self._base_url}/health",
            timeout=min(self._config.timeout, 10),
        )
        self._raise_for_status(resp)
        return HealthResponse.model_validate(resp.json())

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        enable_thinking: Optional[bool] = None,
        generation: Optional[GenerationParams] = None,
    ) -> ChatResponse:
        """Send a conversation and receive a model response.

        Automatically refreshes the API key before the request if it has
        expired, and retries once on ``403`` (in case the key rotated
        between the expiry check and the actual request).

        Parameters
        ----------
        messages:
            The full conversation history.
        enable_thinking:
            Override the config-level thinking toggle for this call.
            *None* means use the value from config.
        generation:
            Override generation/sampling parameters.  *None* means use the
            defaults from config.  Pass a :class:`GenerationParams` with only
            the fields you want to change.
        """
        if enable_thinking is None:
            enable_thinking = self._config.enable_thinking

        if generation is None:
            g = self._config.generation
            generation = GenerationParams(
                max_tokens=g.max_tokens,
                temperature=g.temperature,
                top_p=g.top_p,
                top_k=g.top_k,
                min_p=g.min_p,
                repeat_penalty=g.repeat_penalty,
            )

        request_body = ChatRequest(
            messages=messages,
            enable_thinking=enable_thinking,
            generation=generation,
        )
        payload = request_body.model_dump(exclude_none=True)

        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=self._auth_header(),
            timeout=self._config.timeout,
        )

        # On 403, force-refresh the key and retry once
        if resp.status_code == 403:
            self._api_key = None
            resp = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._auth_header(),
                timeout=self._config.timeout,
            )

        self._raise_for_status(resp)
        return ChatResponse.model_validate(resp.json())

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        """Raise a typed exception if the response indicates an error."""
        if resp.ok:
            return

        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text

        exc_cls = _STATUS_EXCEPTION_MAP.get(resp.status_code, APIError)
        raise exc_cls(resp.status_code, detail)
