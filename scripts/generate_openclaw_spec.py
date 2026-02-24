#!/usr/bin/env python3
"""
Regenerate OPENCLAW_API.md from the canonical tool definitions and API docs.

Run this whenever you add, remove, or modify tools in tool_calling_config.json.

Usage:
    python scripts/generate_openclaw_spec.py
"""

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_CONFIG = REPO_ROOT / "Model" / "model_qwen4_finetuning" / "tool_calling_config.json"
OUTPUT = REPO_ROOT / "OPENCLAW_API.md"

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/XxMorganxX/Tooling-Calling-Mini/master"
)


def _format_param(name: str, spec: dict, indent: int = 2) -> str:
    """Format a single parameter into readable markdown."""
    prefix = " " * indent
    typ = spec.get("type", "any")
    parts = [f"{prefix}- `{name}` ({typ})"]

    qualifiers = []
    if "enum" in spec:
        qualifiers.append("enum: " + ", ".join(f'`"{v}"`' for v in spec["enum"]))
    if "minimum" in spec or "maximum" in spec:
        lo = spec.get("minimum", "")
        hi = spec.get("maximum", "")
        qualifiers.append(f"range: {lo}–{hi}")
    if "default" in spec:
        qualifiers.append(f"default: `{json.dumps(spec['default'])}`")

    if qualifiers:
        parts[0] += f" [{', '.join(qualifiers)}]"

    desc = spec.get("description", "")
    if desc:
        parts[0] += f" — {desc}"

    if typ == "array" and "items" in spec:
        items = spec["items"]
        if items.get("type") == "object" and "properties" in items:
            parts.append(f"{prefix}  Object fields:")
            req = set(items.get("required", []))
            for k, v in items["properties"].items():
                req_tag = " **(required)**" if k in req else ""
                parts.append(f"{_format_param(k, v, indent + 4)}{req_tag}")

    return "\n".join(parts)


def _render_tool(tool: dict) -> str:
    """Render a single tool into a markdown section."""
    name = tool["name"]
    desc = tool["description"]
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    required = set()

    for constraint in params.get("required", []):
        if isinstance(constraint, str):
            required.add(constraint)

    one_of = params.get("oneOf", [])

    lines = [f"### `{name}`", "", desc, ""]

    if props:
        lines.append("| Parameter | Type | Required | Description |")
        lines.append("|---|---|---|---|")
        for pname, pspec in props.items():
            typ = pspec.get("type", "any")
            is_req = "yes" if pname in required else "no"
            pdesc = pspec.get("description", "")

            enum_vals = pspec.get("enum")
            if enum_vals:
                pdesc += " Values: " + ", ".join(f'`{v}`' for v in enum_vals) + "."

            rng_parts = []
            if "minimum" in pspec:
                rng_parts.append(f"min={pspec['minimum']}")
            if "maximum" in pspec:
                rng_parts.append(f"max={pspec['maximum']}")
            if rng_parts:
                pdesc += f" ({', '.join(rng_parts)})"

            if "default" in pspec:
                pdesc += f" Default: `{json.dumps(pspec['default'])}`."

            if typ == "array" and "items" in pspec:
                items = pspec["items"]
                if items.get("type") == "object" and "properties" in items:
                    typ = "array of objects"

            lines.append(f"| `{pname}` | {typ} | {is_req} | {pdesc.strip()} |")

        lines.append("")

    if one_of:
        constraints = []
        for entry in one_of:
            fields = entry.get("required", [])
            constraints.append(" or ".join(f"`{f}`" for f in fields))
        lines.append(f"**Constraint**: Provide one of: {', '.join(constraints)}.")
        lines.append("")

    for pname, pspec in props.items():
        if pspec.get("type") == "array" and "items" in pspec:
            items = pspec["items"]
            if items.get("type") == "object" and "properties" in items:
                lines.append(f"**`{pname}` object fields:**")
                lines.append("")
                lines.append("| Field | Type | Description |")
                lines.append("|---|---|---|")
                sub_req = set(items.get("required", []))
                for sk, sv in items["properties"].items():
                    styp = sv.get("type", "any")
                    sdesc = sv.get("description", "")
                    enum_vals = sv.get("enum")
                    if enum_vals:
                        sdesc += " Values: " + ", ".join(f'`{v}`' for v in enum_vals) + "."
                    req_tag = " **(required)**" if sk in sub_req else ""
                    lines.append(f"| `{sk}` | {styp} | {sdesc.strip()}{req_tag} |")
                lines.append("")

    return "\n".join(lines)


def generate():
    with open(TOOL_CONFIG) as f:
        config = json.load(f)

    tools = config["tools"]
    tool_names = [t["name"] for t in tools]
    generated_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tool_table_rows = []
    for t in tools:
        tool_table_rows.append(f"| `{t['name']}` | {t['description']} |")

    tool_sections = []
    for t in tools:
        tool_sections.append(_render_tool(t))

    doc = textwrap.dedent(f"""\
    # Tool-Calling Mini — External Integration Spec

    > **Auto-generated** from [`tool_calling_config.json`]({GITHUB_RAW_BASE}/Model/model_qwen4_finetuning/tool_calling_config.json)
    > on {generated_ts}. Do not edit by hand — run `python scripts/generate_openclaw_spec.py` to regenerate.

    This document gives an external agent everything it needs to delegate tool calls
    to the Tool-Calling Mini inference API.

    ---

    ## 1. API Endpoint

    | Item | Value |
    |---|---|
    | Base URL | `https://inference.stuart-labs.com` |
    | Chat endpoint | `POST /v1/chat/completions` |
    | Stream endpoint | `POST /v1/chat/completions/stream` |
    | Health check | `GET /health` (no auth) |
    | Interactive docs | `https://inference.stuart-labs.com/docs` (Swagger UI) |

    Served via Cloudflare Tunnel. The server listens on `localhost:8000` internally.

    ---

    ## 2. Authentication

    The API uses **rotating 5-minute API keys** derived from a permanent refresh token.

    ### Flow

    1. **Exchange refresh token for API key:**

    ```http
    POST /auth/token
    Content-Type: application/json

    {{"refresh_token": "<REFRESH_TOKEN>"}}
    ```

    Response:
    ```json
    {{
      "api_key": "a3f8c1...64-hex-chars",
      "expires_at": "2026-02-16T20:35:00+00:00"
    }}
    ```

    2. **Use API key in all subsequent requests:**

    ```
    X-API-Key: <api_key>
    ```

    3. **Refresh on expiry or 403.** The server accepts keys from the current and previous
       5-minute window as a grace period.

    ---

    ## 3. Request Format

    ```http
    POST /v1/chat/completions
    Content-Type: application/json
    X-API-Key: <api_key>
    ```

    ```json
    {{
      "messages": [
        {{"role": "user", "content": "What's the weather tomorrow?"}}
      ],
      "enable_thinking": true,
      "generation": {{
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0
      }}
    }}
    ```

    | Field | Type | Required | Default | Description |
    |---|---|---|---|---|
    | `messages` | array | **yes** | — | Conversation history. Each: `{{"role": "user"|"assistant", "content": "..."}}` |
    | `enable_thinking` | bool | no | `true` | Model produces a reasoning trace before responding. |
    | `generation` | object | no | server defaults | Override sampling params (all sub-fields optional). |

    The server is **stateless** — send the full conversation history each request.

    **Context window**: 4096 tokens. Tool schemas consume ~2661 tokens, leaving ~1435 for
    conversation + response. Keep history short.

    ---

    ## 4. Response Format

    ```json
    {{
      "content": "I'll check the weather for you.",
      "thinking": "The user wants tomorrow's weather...",
      "tool_calls": [
        {{
          "name": "weather",
          "arguments": {{"specific_date": "tomorrow"}}
        }}
      ],
      "usage": {{
        "prompt_tokens": 1842,
        "completion_tokens": 67,
        "tokens_per_second": 48.3
      }}
    }}
    ```

    | Field | Type | Nullable | Description |
    |---|---|---|---|
    | `content` | string | no | Cleaned response text (tool-call artifacts stripped). |
    | `thinking` | string | yes | Reasoning trace. Null if disabled or not produced. |
    | `tool_calls` | array | yes | Parsed tool calls. **Null when no tools needed.** |
    | `tool_results` | array | yes | Present only if server-side tool execution is enabled. |
    | `usage` | object | yes | Token counts and generation speed. |

    Each `tool_calls` entry:

    | Field | Type |
    |---|---|
    | `name` | Tool name (string) |
    | `arguments` | JSON object matching the tool's schema |

    ### Error codes

    | Status | Meaning |
    |---|---|
    | 403 | Expired/invalid API key or refresh token |
    | 422 | Malformed request body |
    | 500 | Server misconfiguration |
    | 502 | llama-server unreachable |
    | 504 | llama-server timeout (>300s) |

    ---

    ## 5. When to Defer

    Delegate to this API when the user's request matches one of the **{len(tools)} registered tools**:

    | Tool | Description |
    |---|---|
    """).rstrip()

    doc += "\n"
    for row in tool_table_rows:
        doc += f"{row}\n"

    doc += textwrap.dedent("""
    **Decision rule**: If the user's intent involves any of the above capabilities
    (home automation, media control, calendar, weather, web search, notifications,
    notes, SMS, briefings, clipboard, system info, or coding tasks), send the message
    to this API. The model handles tool selection and argument extraction internally —
    just forward the natural-language request.

    If `tool_calls` is null in the response, the model determined no tool was needed
    and answered conversationally — use `content` directly.

    ---

    ## 6. Integration Pattern

    ```
    User message arrives
        │
        ├─ Does it involve one of the 12 tool domains? ──► NO ──► Handle normally
        │
        ▼ YES
    POST /v1/chat/completions  (with message history)
        │
        ▼
    Check response.tool_calls
        │
        ├─ null ──► Use response.content as conversational answer
        │
        ▼ non-null
    Execute each tool call locally (you own the tool implementations)
        │
        ▼
    Optionally feed results back as a follow-up message
        │
        ▼
    Synthesize final response for the user
    ```

    ### Python integration example

    ```python
    import requests
    from datetime import datetime, timezone

    BASE_URL = "https://inference.stuart-labs.com"
    REFRESH_TOKEN = "<your-refresh-token>"

    api_key = None
    expires_at = None


    def get_api_key():
        global api_key, expires_at
        now = datetime.now(timezone.utc)
        if api_key and expires_at and now < expires_at:
            return api_key
        resp = requests.post(f"{BASE_URL}/auth/token", json={"refresh_token": REFRESH_TOKEN})
        resp.raise_for_status()
        data = resp.json()
        api_key = data["api_key"]
        expires_at = datetime.fromisoformat(data["expires_at"])
        return api_key


    def delegate(messages: list[dict]) -> dict:
        headers = {"Content-Type": "application/json", "X-API-Key": get_api_key()}
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"messages": messages},
            headers=headers,
            timeout=120,
        )
        if resp.status_code == 403:
            api_key = None
            headers["X-API-Key"] = get_api_key()
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={"messages": messages},
                headers=headers,
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()
    ```

    ---

    ## 7. Full Tool Schemas

    Canonical definitions from
    """ + f"[`tool_calling_config.json`]({GITHUB_RAW_BASE}/Model/model_qwen4_finetuning/tool_calling_config.json).\n\n")

    for section in tool_sections:
        doc += section + "\n---\n\n"

    doc += textwrap.dedent(f"""\
    ## 8. Keeping This Spec Up to Date

    This file is generated from the canonical tool definitions at:

    ```
    Model/model_qwen4_finetuning/tool_calling_config.json
    ```

    When tools are added or modified:

    1. Edit `tool_calling_config.json`
    2. Run `python scripts/generate_openclaw_spec.py`
    3. Commit and push

    The raw-file URL for periodic polling:

    ```
    {GITHUB_RAW_BASE}/OPENCLAW_API.md
    ```

    The raw tool config JSON (if you prefer to parse schemas directly):

    ```
    {GITHUB_RAW_BASE}/Model/model_qwen4_finetuning/tool_calling_config.json
    ```

    ### Automation options

    - **Git pre-commit hook**: Add `python scripts/generate_openclaw_spec.py` to
      `.git/hooks/pre-commit` so the spec regenerates on every commit that touches
      `tool_calling_config.json`.
    - **GitHub Action**: Trigger on pushes to `Model/model_qwen4_finetuning/tool_calling_config.json`
      and auto-commit the regenerated `OPENCLAW_API.md`.
    - **Polling**: Have openclaw fetch the raw URL on a schedule (e.g. every 15 minutes)
      and diff against its cached copy.

    ---

    *Generated {generated_ts} — {len(tools)} tools registered*
    """)

    OUTPUT.write_text(doc)
    print(f"Wrote {OUTPUT} ({len(tools)} tools, {len(doc)} chars)")


if __name__ == "__main__":
    generate()
