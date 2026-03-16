"""
LocalToolExecutor -- client-side tool execution for SDK consumers.

Executes the client_side_tools listed in config.yaml using the existing
execute() implementations in Model/tools/. Any client (Discord bot, CLI,
web app) that imports this SDK gets access to all tool implementations
without duplicating code.

Usage:
    from tool_calling_sdk import LocalToolExecutor

    executor = LocalToolExecutor()  # registers client_side_tools from config

    # After receiving a response from the inference API:
    already_done = {r["tool_name"] for r in response.get("tool_results") or []}
    local_results = await executor.execute_pending(response["tool_calls"], already_done)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure Model/ (parent of this SDK package) is on sys.path so that
# sibling packages (tools/, inference/) are importable from any project.
_model_dir = Path(__file__).parent.parent
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from tools.executor import ToolExecutor, ToolResult  # noqa: E402
from tools.models import validate_tool_call, ValidatedToolCall, ToolValidationError  # noqa: E402
from inference.inference import load_config, DEFAULT_CONFIG  # noqa: E402


class LocalToolExecutor:
    """Executes client-side tools for SDK consumers.

    By default registers the tools listed under ``agent.client_side_tools``
    in config.yaml. Pass ``tools`` to override with a custom list.

    Args:
        timeout: Per-tool execution timeout in seconds.
        tools: Tool names to register. When None, uses client_side_tools
               from config.yaml.
    """

    def __init__(self, timeout: float = 30.0, tools: list[str] | None = None):
        cfg = load_config(DEFAULT_CONFIG)
        client_tools = tools if tools is not None else cfg.get("agent", {}).get("client_side_tools", [])
        self._executor = ToolExecutor(timeout=timeout)
        self._executor.register_all_tools(client_tools)
        self._registered: set[str] = set(client_tools)

    @property
    def registered_tools(self) -> set[str]:
        """Names of tools registered for local execution."""
        return frozenset(self._registered)

    async def execute_pending(
        self,
        tool_calls: list[dict[str, Any]],
        already_executed: set[str] | None = None,
    ) -> list[ToolResult]:
        """Execute tool calls not yet handled by the server.

        Filters ``tool_calls`` to those whose names are not in
        ``already_executed``, validates each, and runs them concurrently.

        Args:
            tool_calls: The ``tool_calls`` list from the inference API response.
                        Each entry is ``{"name": str, "arguments": dict}``.
            already_executed: Set of tool names already present in
                              ``tool_results`` from the server response.
                              Defaults to empty set.

        Returns:
            List of ToolResult (one per pending call, including validation errors).
        """
        skip = already_executed or set()
        pending = [tc for tc in tool_calls if tc["name"] not in skip]

        validated: list[ValidatedToolCall] = []
        errors: list[ToolResult] = []

        for tc in pending:
            result = validate_tool_call(tc["name"], tc.get("arguments", {}))
            if isinstance(result, ToolValidationError):
                errors.append(ToolResult(
                    tool_name=result.tool_name,
                    success=False,
                    error=f"Validation failed: {result.error}",
                    duration_ms=0.0,
                ))
            else:
                validated.append(result)

        exec_results = await self._executor.execute_batch(validated)
        return errors + exec_results
