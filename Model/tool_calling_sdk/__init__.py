"""
Shared SDK for building tool-calling conversation context.

Any client (web frontend, CLI, API consumer) imports ContextManager
to manage messages, track token budget, and build API request payloads.
"""

from tool_calling_sdk.context import ContextManager

__all__ = ["ContextManager"]
