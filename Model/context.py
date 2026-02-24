"""
Server-side context manager for the inference pipeline.

Wraps prompt construction with token-aware message management,
tool-result injection, and smart truncation to stay within the
model's context window.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from inference.inference import build_prompt

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # rough estimate for Qwen3 tokenizer


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class ContextManager:
    """Token-aware context builder for the inference server.

    Parameters
    ----------
    system_prompt : str
        System prompt text (tool definitions are added separately).
    tools : list[dict]
        Tool schemas in OpenAI-compatible format.
    max_context_tokens : int
        Model's total context size (n_ctx).
    reserve_for_response : int
        Tokens reserved for the model's reply generation.
    """

    system_prompt: str
    tools: list[dict]
    max_context_tokens: int = 4096
    reserve_for_response: int = 512

    _system_overhead: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._system_overhead = self._compute_system_overhead()

    def _compute_system_overhead(self) -> int:
        """Estimate tokens consumed by the system block (prompt + tool schemas).

        Mirrors the ``<tools>`` XML format from ``build_prompt()``, which
        matches the Qwen3 ``apply_chat_template`` output used in training.
        """
        parts = [
            "<|im_start|>system\n",
            self.system_prompt,
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            *[json.dumps(t) + "\n" for t in self.tools],
            "</tools>\n\n",
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n",
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>',
            "<|im_end|>\n",
        ]
        return _estimate_tokens("".join(parts))

    @property
    def message_budget(self) -> int:
        """Tokens available for conversation messages."""
        return max(0, self.max_context_tokens - self._system_overhead - self.reserve_for_response)

    def truncate_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop oldest messages to fit within the token budget.

        Preserves the most recent user turn at minimum.  Drops from
        the front of the list to maintain conversation coherence.
        """
        budget = self.message_budget
        total = sum(_estimate_tokens(m["content"]) for m in messages)

        if total <= budget:
            return list(messages)

        result = list(messages)
        while sum(_estimate_tokens(m["content"]) for m in result) > budget and len(result) > 1:
            result.pop(0)

        logger.debug(
            "Truncated %d messages -> %d to fit budget of %d tokens",
            len(messages),
            len(result),
            budget,
        )
        return result

    def inject_tool_results(
        self,
        messages: list[dict[str, str]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Append tool results as a 'tool' role message to the conversation.

        Each tool result is serialised as ``[tool_name] <json>`` on its
        own line inside a single tool message.
        """
        if not tool_results:
            return messages

        lines = []
        for tr in tool_results:
            name = tr.get("tool_name", "unknown")
            if tr.get("success"):
                payload = json.dumps(tr.get("result"), default=str)
                lines.append(f"[{name}] {payload}")
            else:
                lines.append(f"[{name}] ERROR: {tr.get('error', 'unknown error')}")

        return [*messages, {"role": "tool", "content": "\n".join(lines)}]

    def build_prompt(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = True,
        auto_truncate: bool = True,
    ) -> str:
        """Build a complete model prompt, optionally truncating to fit budget."""
        if auto_truncate:
            messages = self.truncate_messages(messages)
        return build_prompt(self.system_prompt, self.tools, messages, enable_thinking)

    def debug_summary(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "system_overhead_tokens": self._system_overhead,
            "message_budget": self.message_budget,
            "message_tokens_used": sum(_estimate_tokens(m["content"]) for m in messages),
            "message_count": len(messages),
            "max_context_tokens": self.max_context_tokens,
            "reserve_for_response": self.reserve_for_response,
            "tool_count": len(self.tools),
        }
