"""
Tool executor -- dispatches validated tool calls to their execute functions.

Provides timeout handling, structured ToolResult responses, and batch execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel

from tools.models import ValidatedToolCall, ToolValidationError

logger = logging.getLogger(__name__)


_ERROR_DETAIL_KEYS = ("hint", "suggestion", "details", "action_required", "validation_errors")


def _build_error_string(result: dict) -> str | None:
    """Build a comprehensive error string from a tool result dict.

    Merges the primary 'error' field with any supplementary detail keys
    so the caller gets the full picture in a single string.
    """
    primary = result.get("error")
    if not primary:
        return None

    parts = [str(primary)]
    for key in _ERROR_DETAIL_KEYS:
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            parts.append(f"{key}: {'; '.join(str(v) for v in val)}")
        elif isinstance(val, dict):
            import json
            parts.append(f"{key}: {json.dumps(val, default=str)}")
        else:
            parts.append(str(val))

    return " | ".join(parts) if len(parts) > 1 else parts[0]


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float


class ToolExecutor:
    """Dispatches ValidatedToolCalls to their registered execute functions."""

    def __init__(self, timeout: float = 30.0):
        self._handlers: dict[str, Callable] = {}
        self._timeout = timeout

    def register(self, name: str, fn: Callable) -> None:
        self._handlers[name] = fn

    def register_all_tools(self, tools: list[str] | None = None) -> None:
        """Auto-import and register tool modules.

        Args:
            tools: Optional list of tool names to register. When provided, only
                   those tools are imported. When None, all tools are registered.
        """
        tool_modules = {
            "weather": "tools.weather",
            "spotify_playback": "tools.spotify",
            "kasa_lighting": "tools.kasa_lighting",
            "calendar_data": "tools.calendar_tool",
            "stickies": "tools.stickies",
            "send_sms": "tools.sms",
            "google_search": "tools.google_search",
            "read_clipboard": "tools.clipboard",
            "briefing": "tools.briefing",
            "get_notifications": "tools.notifications",
            "system_info": "tools.system_info",
            "cursor_composer": "tools.cursor",
        }

        subset = set(tools) if tools is not None else set(tool_modules)

        for tool_name, module_path in tool_modules.items():
            if tool_name not in subset:
                continue
            try:
                import importlib

                mod = importlib.import_module(module_path)
                fn = getattr(mod, "execute", None)
                if fn is None:
                    logger.warning("Module %s has no execute() function", module_path)
                    continue
                self.register(tool_name, fn)
                logger.debug("Registered tool: %s", tool_name)
            except Exception as e:
                logger.warning("Failed to import tool %s (%s): %s", tool_name, module_path, e)

        logger.info("Registered %d/%d tools", len(self._handlers), len(subset))

    async def execute(self, call: ValidatedToolCall) -> ToolResult:
        """Execute a single validated tool call with timeout."""
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=f"No handler registered for tool '{call.name}'",
                duration_ms=0.0,
            )

        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, handler, call.arguments),
                timeout=self._timeout,
            )
            elapsed = (time.perf_counter() - start) * 1000

            success = result.get("success", True) if isinstance(result, dict) else True
            error = _build_error_string(result) if isinstance(result, dict) and not success else None

            return ToolResult(
                tool_name=call.name,
                success=success,
                result=result,
                error=error,
                duration_ms=round(elapsed, 2),
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=f"Tool execution timed out after {self._timeout}s",
                duration_ms=round(elapsed, 2),
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception("Tool %s raised an exception", call.name)
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=str(e),
                duration_ms=round(elapsed, 2),
            )

    async def execute_batch(self, calls: list[ValidatedToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls. Runs them concurrently."""
        if not calls:
            return []
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))

    def execute_from_validation(
        self, validation_result: ValidatedToolCall | ToolValidationError
    ) -> ToolResult | None:
        """Synchronous convenience: returns a ToolResult for a validation error,
        or None if the call is valid (caller should then use execute/execute_batch).
        """
        if isinstance(validation_result, ToolValidationError):
            return ToolResult(
                tool_name=validation_result.tool_name,
                success=False,
                error=f"Validation failed: {validation_result.error}",
                result={"raw_arguments": validation_result.raw_arguments},
                duration_ms=0.0,
            )
        return None
