"""
AgentLoop -- iterative tool-call / result cycle.

Drives the model through multiple turns:
  1. Send messages to llama-server and get a completion.
  2. Parse any tool calls from the response.
  3. Validate and execute those tool calls.
  4. Inject tool results back into the conversation.
  5. Repeat until the model produces a final answer or limits are hit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests as http_requests

from context import ContextManager
from inference.inference import (
    clean_tool_artifacts,
    parse_tool_calls,
    strip_thinking,
)
from tools.executor import ToolExecutor, ToolResult
from tools.models import (
    ToolValidationError,
    ValidatedToolCall,
    validate_tool_call,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """Record of a single agent iteration."""

    iteration: int
    thinking: str | None
    content: str
    tool_calls: list[dict[str, Any]] | None
    tool_results: list[dict[str, Any]] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0


@dataclass
class AgentResult:
    """Final result of a complete agent loop run."""

    final_content: str
    final_thinking: str | None
    turns: list[AgentTurn]
    total_tool_calls: int
    total_iterations: int
    elapsed_ms: float
    stopped_reason: str


@dataclass
class AgentLoop:
    """Iterative agent that re-prompts the model with tool results.

    Parameters
    ----------
    context : ContextManager
        Manages prompt construction and token budgeting.
    executor : ToolExecutor
        Dispatches validated tool calls to handler functions.
    llama_host : str
        Host of the llama-server.
    llama_port : int
        Port of the llama-server.
    gen_params : dict
        Default generation parameters for llama-server.
    max_iterations : int
        Maximum number of model turns before force-stopping.
    enable_thinking : bool
        Whether to enable <think> tags in the prompt.
    """

    context: ContextManager
    executor: ToolExecutor
    llama_host: str
    llama_port: int
    gen_params: dict = field(default_factory=dict)
    max_iterations: int = 5
    enable_thinking: bool = True

    async def run(
        self,
        messages: list[dict[str, str]],
        gen_overrides: dict | None = None,
    ) -> AgentResult:
        """Execute the full agent loop.

        Returns an AgentResult with the final answer and full turn history.
        """
        start = time.monotonic()
        gen = {**self.gen_params, **(gen_overrides or {})}
        conversation = list(messages)
        turns: list[AgentTurn] = []
        total_tool_calls = 0
        stopped_reason = "max_iterations"

        for iteration in range(1, self.max_iterations + 1):
            prompt = self.context.build_prompt(
                conversation, enable_thinking=self.enable_thinking
            )
            raw_content, timings = self._call_llama(prompt, gen)

            thinking, response_text = strip_thinking(raw_content)
            tool_calls_raw = parse_tool_calls(response_text)
            content = clean_tool_artifacts(response_text)

            conversation.append({"role": "assistant", "content": content})

            turn = AgentTurn(
                iteration=iteration,
                thinking=thinking or None,
                content=content,
                tool_calls=tool_calls_raw or None,
                tool_results=None,
                prompt_tokens=timings.get("prompt_n", 0),
                completion_tokens=timings.get("predicted_n", 0),
                tokens_per_second=timings.get("predicted_per_second", 0.0),
            )

            if not tool_calls_raw:
                turn.tool_results = None
                turns.append(turn)
                stopped_reason = "natural"
                break

            total_tool_calls += len(tool_calls_raw)
            tool_results = await self._execute_tools(tool_calls_raw)
            turn.tool_results = [self._result_to_dict(r) for r in tool_results]
            turns.append(turn)

            conversation = self.context.inject_tool_results(
                conversation,
                turn.tool_results,
            )

        elapsed = (time.monotonic() - start) * 1000

        final_turn = turns[-1] if turns else None
        return AgentResult(
            final_content=final_turn.content if final_turn else "",
            final_thinking=final_turn.thinking if final_turn else None,
            turns=turns,
            total_tool_calls=total_tool_calls,
            total_iterations=len(turns),
            elapsed_ms=elapsed,
            stopped_reason=stopped_reason,
        )

    def _call_llama(self, prompt: str, gen: dict) -> tuple[str, dict]:
        """Synchronous call to llama-server."""
        url = f"http://{self.llama_host}:{self.llama_port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": gen.get("max_tokens", 512),
            "temperature": gen.get("temperature", 0.6),
            "top_p": gen.get("top_p", 0.95),
            "top_k": gen.get("top_k", 20),
            "min_p": gen.get("min_p", 0.0),
            "repeat_penalty": gen.get("repeat_penalty", 1.0),
            "stop": ["<|im_end|>", "<|im_start|>"],
            "stream": False,
        }
        resp = http_requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        return result.get("content", ""), result.get("timings", {})

    async def _execute_tools(
        self, tool_calls_raw: list[dict]
    ) -> list[ToolResult | _ValidationErrorResult]:
        """Validate and execute a batch of tool calls."""
        validated: list[ValidatedToolCall] = []
        errors: list[_ValidationErrorResult] = []

        for tc in tool_calls_raw:
            vr = validate_tool_call(tc.get("name", ""), tc.get("arguments", {}))
            if isinstance(vr, ToolValidationError):
                errors.append(
                    _ValidationErrorResult(
                        tool_name=vr.tool_name,
                        error=f"Validation failed: {vr.error}",
                    )
                )
            else:
                validated.append(vr)

        exec_results = await self.executor.execute_batch(validated) if validated else []
        return [*errors, *exec_results]

    @staticmethod
    def _result_to_dict(r: Any) -> dict[str, Any]:
        if isinstance(r, ToolResult):
            return {
                "tool_name": r.tool_name,
                "success": r.success,
                "result": r.result,
                "error": r.error,
                "duration_ms": r.duration_ms,
            }
        return {
            "tool_name": r.tool_name,
            "success": False,
            "result": None,
            "error": r.error,
            "duration_ms": 0.0,
        }


@dataclass
class _ValidationErrorResult:
    """Lightweight stand-in for a failed validation (not a real ToolResult)."""

    tool_name: str
    error: str
