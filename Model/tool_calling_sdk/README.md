# Tool Calling SDK

Shared SDK for building tool-calling conversation context. Importable by any client (CLI, web, external consumers) that needs to manage a conversation with the inference server.

## What It Provides

`ContextManager` — a stateful conversation builder that:

- Tracks message history (`user`, `assistant`, `tool` messages)
- Provides approximate token budgeting (4096 context window minus system overhead minus response reserve)
- Automatically truncates oldest messages when the conversation exceeds the token budget
- Serialises API request payloads for the `/v1/chat/completions` endpoint
- Injects tool results as `role=tool` messages

## Usage

```python
from tool_calling_sdk import ContextManager

ctx = ContextManager(
    system_prompt="You are a tool-calling assistant...",
    max_context_tokens=4096,
    reserve_for_response=512,
    tool_schema_token_estimate=2700,
)

ctx.add_user_message("What's the weather?")

payload = ctx.build_payload(
    enable_thinking=True,
    execute_tools=True,
    temperature=0.6,
)
# → Send payload to POST /v1/chat/completions

ctx.add_assistant_message("I'll check the weather for you.")
ctx.add_tool_result("weather", {"temp": 72, "condition": "sunny"})

# Inspect token usage
print(ctx.debug_summary())
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Exports `ContextManager` |
| `context.py` | `ContextManager` and `Message` dataclasses |

## Token Budget Model

The 4096-token context is split as:

```
system_prompt tokens + tool_schema_tokens (~2700) + message_history + response_reserve (512)
```

When `message_history` exceeds its budget, the oldest messages are dropped (preserving the latest user turn).

## Key Types

- `Message(role, content)` — a single conversation message with `approx_tokens` property
- `ContextManager.build_payload(...)` — returns the complete JSON body for the chat completions API, including optional generation parameter overrides
