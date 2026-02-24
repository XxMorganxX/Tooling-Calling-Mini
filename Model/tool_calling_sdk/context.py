"""
ContextManager -- stateful conversation builder for tool-calling clients.

Manages the message list, provides approximate token budgeting,
and serialises payloads for the inference server's /v1/chat/completions
endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

CHARS_PER_TOKEN = 4  # rough estimate for Qwen3 tokenizer


@dataclass
class Message:
    role: Literal["user", "assistant", "tool", "system"]
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @property
    def approx_tokens(self) -> int:
        return max(1, len(self.content) // CHARS_PER_TOKEN)


@dataclass
class ContextManager:
    """Conversation context builder for the inference server.

    Parameters
    ----------
    system_prompt : str
        The system prompt text (without tool definitions -- those live
        on the server side).
    max_context_tokens : int
        Total context budget (should match model ``max_seq_length``).
    reserve_for_response : int
        Tokens reserved for the model's reply.  The remaining budget is
        split between the system prompt overhead and message history.
    """

    system_prompt: str = (
        "You are a tool-calling assistant. Analyze user requests and determine "
        "the correct tool call(s) to fulfill them. If no tool is needed, respond "
        "normally without tool calls."
    )
    max_context_tokens: int = 4096
    reserve_for_response: int = 512
    tool_schema_token_estimate: int = 2700

    messages: list[Message] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._truncate_if_needed()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Append a tool result message (role=tool) with serialised JSON."""
        if isinstance(result, str):
            content = result
        else:
            content = json.dumps(result, default=str)
        self.messages.append(
            Message(role="tool", content=f"[{tool_name}] {content}")
        )

    def clear(self) -> None:
        self.messages.clear()

    def get_messages(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    def build_payload(
        self,
        *,
        enable_thinking: bool = True,
        execute_tools: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> dict:
        """Return the JSON-serialisable request body for ``/v1/chat/completions``."""
        payload: dict[str, Any] = {
            "messages": self.get_messages(),
            "enable_thinking": enable_thinking,
            "execute_tools": execute_tools,
        }

        gen_overrides: dict[str, Any] = {}
        if max_tokens is not None:
            gen_overrides["max_tokens"] = max_tokens
        if temperature is not None:
            gen_overrides["temperature"] = temperature
        if top_p is not None:
            gen_overrides["top_p"] = top_p
        if top_k is not None:
            gen_overrides["top_k"] = top_k
        if min_p is not None:
            gen_overrides["min_p"] = min_p
        if repeat_penalty is not None:
            gen_overrides["repeat_penalty"] = repeat_penalty
        if gen_overrides:
            payload["generation"] = gen_overrides

        return payload

    # ------------------------------------------------------------------
    # Token budgeting
    # ------------------------------------------------------------------

    @property
    def system_overhead_tokens(self) -> int:
        """Approximate tokens consumed by system prompt + tool schemas."""
        prompt_tokens = max(1, len(self.system_prompt) // CHARS_PER_TOKEN)
        return prompt_tokens + self.tool_schema_token_estimate

    @property
    def message_budget(self) -> int:
        """Tokens available for conversation messages."""
        return max(
            0,
            self.max_context_tokens
            - self.system_overhead_tokens
            - self.reserve_for_response,
        )

    @property
    def message_tokens_used(self) -> int:
        return sum(m.approx_tokens for m in self.messages)

    def _truncate_if_needed(self) -> None:
        """Drop oldest messages (preserving the latest user turn) until
        the conversation fits within the token budget."""
        budget = self.message_budget
        while self.message_tokens_used > budget and len(self.messages) > 1:
            self.messages.pop(0)

    # ------------------------------------------------------------------
    # Debug / introspection
    # ------------------------------------------------------------------

    def debug_summary(self) -> dict:
        return {
            "total_messages": len(self.messages),
            "message_tokens_used": self.message_tokens_used,
            "message_budget": self.message_budget,
            "system_overhead_tokens": self.system_overhead_tokens,
            "max_context_tokens": self.max_context_tokens,
            "reserve_for_response": self.reserve_for_response,
        }
