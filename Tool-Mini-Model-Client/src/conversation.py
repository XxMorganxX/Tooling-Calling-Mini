"""Stateful conversation manager built on top of InferenceClient."""

from __future__ import annotations

from typing import Optional

from .client import InferenceClient
from .config import ClientConfig
from .models import ChatMessage, ChatResponse, GenerationParams


class ConversationManager:
    """Maintains conversation history and handles truncation.

    The inference server is stateless, so the full message history must be sent
    with every request.  This class accumulates turns and automatically drops
    the oldest messages when the history exceeds ``max_history_messages`` to
    stay within the model's limited context window (~1 435 tokens for
    conversation + response).

    Parameters
    ----------
    client:
        An :class:`InferenceClient` instance.
    config:
        The :class:`ClientConfig` used to build *client* (supplies
        ``max_history_messages`` and ``enable_thinking``).
    """

    def __init__(self, client: InferenceClient, config: ClientConfig) -> None:
        self._client = client
        self._config = config
        self._history: list[ChatMessage] = []
        self._enable_thinking: bool = config.enable_thinking

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def history(self) -> list[ChatMessage]:
        """Return the current message history (read-only copy)."""
        return list(self._history)

    @property
    def enable_thinking(self) -> bool:
        return self._enable_thinking

    @enable_thinking.setter
    def enable_thinking(self, value: bool) -> None:
        self._enable_thinking = value

    # ── Public API ────────────────────────────────────────────────────────

    def send(
        self,
        user_message: str,
        *,
        generation: Optional[GenerationParams] = None,
    ) -> ChatResponse:
        """Send a user message and return the model's response.

        The user message is appended to the history, the full (possibly
        truncated) history is sent to the API, and the assistant reply is
        appended before returning.

        Parameters
        ----------
        user_message:
            Plain-text message from the user.
        generation:
            Optional per-call generation overrides.
        """
        self._history.append(ChatMessage(role="user", content=user_message))
        self._truncate()

        response = self._client.chat(
            messages=self._history,
            enable_thinking=self._enable_thinking,
            generation=generation,
        )

        self._history.append(
            ChatMessage(role="assistant", content=response.content)
        )

        return response

    def clear(self) -> None:
        """Reset the conversation history."""
        self._history.clear()

    # ── Internals ─────────────────────────────────────────────────────────

    def _truncate(self) -> None:
        """Drop the oldest turns if history exceeds the configured limit.

        Truncation always removes *pairs* of messages (user + assistant) so
        the history never starts with a dangling assistant reply.
        """
        max_msgs = self._config.max_history_messages
        if max_msgs <= 0 or len(self._history) <= max_msgs:
            return

        overflow = len(self._history) - max_msgs

        # Round up to the nearest even number so we drop complete turns
        if overflow % 2 != 0:
            overflow += 1

        self._history = self._history[overflow:]
