"""
Shared SDK for tool-calling clients.

Provides:
- ContextManager  -- stateful conversation builder and token budget tracker
- LocalToolExecutor -- executes client-side tools locally using Model/tools/
"""

from tool_calling_sdk.context import ContextManager
from tool_calling_sdk.executor import LocalToolExecutor

__all__ = ["ContextManager", "LocalToolExecutor"]
