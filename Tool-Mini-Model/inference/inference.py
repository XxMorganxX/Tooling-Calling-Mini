"""
inference.py -- Interactive inference via llama.cpp GPU server.

Launches llama-server from your local llama.cpp installation with full
GPU offload, then provides an interactive chat loop with tool-calling
support using the fine-tuned GGUF model.

Usage:
    python inference.py                          # from inference/ directory
    python inference.py --config ../config.yaml
    python inference.py --gguf path/to/model.gguf
    python inference.py --server-running         # connect to existing server
    python inference.py --port 9090              # custom port
    python inference.py --no-think               # disable thinking/reasoning

Commands inside the chat:
    /reset   -- clear conversation history
    /system  -- show current system prompt + tool count
    /quit    -- exit
"""

import argparse
import json
import os
import re
import subprocess
import signal
import sys
import time
from pathlib import Path

import requests
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "model_qwen4_finetuning", "config.yaml")


def load_config(config_path: str = None) -> dict:
    config_path = config_path or DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: str, relative_path: str) -> str:
    """Resolve a path from config relative to the config file's directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path or DEFAULT_CONFIG))
    return os.path.normpath(os.path.join(config_dir, relative_path))


def load_tools(tool_config_path: str) -> list[dict]:
    """Load tools in OpenAI-compatible format for the prompt."""
    with open(tool_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in config["tools"]
    ]


def find_gguf(gguf_dir: str) -> str | None:
    """Find the most recent GGUF file in the output directory."""
    gguf_files = sorted(Path(gguf_dir).glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(gguf_files[0]) if gguf_files else None


def build_prompt(
    system_prompt: str,
    tools: list[dict],
    messages: list[dict],
    enable_thinking: bool = True,
) -> str:
    """Build a Qwen3 chat-formatted prompt with tool definitions.

    Follows the chat_template.jinja from the model exactly:
    - System message with tool schemas in <tools> XML block
    - User/assistant turns with <|im_start|>/<|im_end|> markers
    - Tool calls wrapped in <tool_call> XML tags
    """
    prompt = "<|im_start|>system\n"
    prompt += system_prompt + "\n\n"
    prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    prompt += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    for tool in tools:
        prompt += json.dumps(tool) + "\n"
    prompt += "</tools>\n\n"
    prompt += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    prompt += '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    prompt += "<|im_end|>\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"

    if enable_thinking:
        prompt += "<think>\n"

    return prompt


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output.

    Handles multiple formats the model may produce:
      1. <tool_call>{...}</tool_call>     (standard)
      2. {..."name":...}...</tool_call>   (missing opening tag)
      3. Bare JSON with "name" + "arguments" keys
    """
    calls = []

    full_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.findall(full_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    partial_pattern = r'(\{"name"\s*:.*?\})\s*</tool_call>'
    for match in re.findall(partial_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    bare_pattern = r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\})'
    for match in re.findall(bare_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            continue

    return calls


def clean_tool_artifacts(text: str) -> str:
    """Remove tool_call XML tags and leftover artifacts from display text."""
    text = re.sub(r"</?tool_call>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_thinking(text: str) -> tuple[str, str]:
    """Separate thinking from response content. Returns (thinking, response)."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, response


def stream_response(prompt: str, host: str, port: int, gen_params: dict, enable_thinking: bool) -> tuple[str, dict]:
    """Stream tokens from llama-server, displaying thinking and response live.

    Returns (full_content, timings_dict).

    Uses a unified buffer to reliably detect XML tags (</think>,
    <tool_call>, </tool_call>) even when they are split across token
    boundaries, and suppresses all tool-call content from the live
    display (tool calls are parsed and pretty-printed separately).
    """
    url = f"http://{host}:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": gen_params["max_tokens"],
        "temperature": gen_params["temperature"],
        "top_p": gen_params["top_p"],
        "top_k": gen_params["top_k"],
        "min_p": gen_params["min_p"],
        "repeat_penalty": gen_params["repeat_penalty"],
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": True,
    }

    resp = requests.post(url, json=payload, timeout=120, stream=True)
    resp.raise_for_status()

    full_text = ""
    timings = {}

    in_thinking = enable_thinking
    thinking_done = False
    response_started = False
    in_tool_call = False
    buf = ""

    TAG_MARGIN = len("</tool_call>")

    def _flush_response(text: str):
        """Print text as assistant response content."""
        nonlocal response_started
        if not text:
            return
        if not response_started:
            print("\nAssistant: ", end="", flush=True)
            response_started = True
        print(text, end="", flush=True)

    if in_thinking:
        print("\n  [Thinking] ", end="", flush=True)

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("timings"):
            timings = chunk["timings"]

        if chunk.get("stop"):
            break

        token = chunk.get("content", "")
        if not token:
            continue

        full_text += token
        buf += token

        # ── Thinking phase ────────────────────────────────────────────
        if in_thinking and not thinking_done:

            # 1) Normal end of thinking
            if "</think>" in buf:
                before, _, after = buf.partition("</think>")
                clean = before.strip()
                if clean:
                    print(clean, end="", flush=True)
                print()

                thinking_done = True
                in_thinking = False
                buf = after

                if "<tool_call>" in buf:
                    _, _, buf = buf.partition("<tool_call>")
                    in_tool_call = True
                elif buf.strip():
                    _flush_response(buf.lstrip("\n"))
                    buf = ""
                else:
                    buf = ""
                continue

            # 2) <tool_call> appeared without </think> (model skipped it)
            if "<tool_call>" in buf:
                before, _, after = buf.partition("<tool_call>")
                clean = before.strip()
                if clean:
                    print(clean, end="", flush=True)
                print()

                thinking_done = True
                in_thinking = False
                in_tool_call = True
                buf = after
                continue

            # 3) Stray </tool_call> without any opening tag
            if "</tool_call>" in buf:
                print()
                thinking_done = True
                in_thinking = False
                buf = ""
                continue

            # 4) Safety valve: buffer contains a full tool-call JSON
            #    pattern but no XML tags arrived — model emitted bare JSON
            if len(buf) > 150 and re.search(
                r'"name"\s*:.*"arguments"\s*:', buf, re.DOTALL,
            ):
                print()
                thinking_done = True
                in_thinking = False
                in_tool_call = True
                buf = ""
                continue

            # 5) Safe incremental display of thinking text
            safe_len = len(buf) - TAG_MARGIN
            if safe_len > 0:
                json_match = re.search(r'\{\s*"name"\s*:', buf)
                if json_match:
                    print_len = min(safe_len, json_match.start())
                    if print_len > 0:
                        print(buf[:print_len], end="", flush=True)
                        buf = buf[print_len:]
                else:
                    print(buf[:safe_len], end="", flush=True)
                    buf = buf[safe_len:]
            continue

        # ── Inside a tool-call block: suppress display ────────────────
        if in_tool_call:
            if "</tool_call>" in buf:
                _, _, after = buf.partition("</tool_call>")
                in_tool_call = False
                buf = after
                if not buf.strip():
                    buf = ""
                    continue
                # fall through to display any remaining content
            else:
                if len(buf) > TAG_MARGIN * 3:
                    buf = buf[-TAG_MARGIN:]
                continue

        # ── Normal response display ───────────────────────────────────

        # Opening <tool_call> tag
        if "<tool_call>" in buf:
            before, _, after = buf.partition("<tool_call>")
            _flush_response(before.lstrip("\n"))
            in_tool_call = True
            buf = after
            if "</tool_call>" in buf:
                _, _, buf = buf.partition("</tool_call>")
                in_tool_call = False
            if not buf.strip():
                buf = ""
            continue

        # Stray </tool_call> without opening tag
        if "</tool_call>" in buf:
            _, _, after = buf.partition("</tool_call>")
            buf = after
            continue

        # Buffered incremental display
        safe_len = len(buf) - TAG_MARGIN
        if safe_len > 0:
            display = buf[:safe_len]
            buf = buf[safe_len:]
            clean = display.lstrip("\n") if not response_started else display
            _flush_response(clean)

    # ── Flush remaining buffer ────────────────────────────────────────
    if buf:
        clean = re.sub(r"</?tool_call>", "", buf)
        clean = re.sub(
            r'\{\s*"name"\s*:.*?"arguments"\s*:\s*\{.*?\}\s*\}',
            "", clean, flags=re.DOTALL,
        )
        clean = clean.strip()

        if in_thinking and not thinking_done:
            if clean:
                print(clean, end="", flush=True)
            print()
        elif not in_tool_call and clean:
            _flush_response(clean)

    if response_started:
        print()

    return full_text, timings


def start_server(
    llamacpp_path: str,
    gguf_path: str,
    host: str,
    port: int,
    gpu_layers: int,
    n_ctx: int,
    flash_attn: bool,
) -> subprocess.Popen:
    """Start llama-server and wait for it to be ready."""
    server_bin = os.path.join(llamacpp_path, "llama-server.exe")
    if not os.path.exists(server_bin):
        server_bin = os.path.join(llamacpp_path, "llama-server")

    if not os.path.exists(server_bin):
        print(f"ERROR: llama-server not found at {llamacpp_path}")
        sys.exit(1)

    cmd = [
        server_bin,
        "-m", gguf_path,
        "--host", host,
        "--port", str(port),
        "-ngl", str(gpu_layers),
        "-c", str(n_ctx),
    ]
    if flash_attn:
        cmd.extend(["--flash-attn", "on"])

    print(f"Starting llama-server...")
    print(f"  Binary:  {server_bin}")
    print(f"  Model:   {gguf_path}")
    print(f"  GPU:     {gpu_layers} layers offloaded")
    print(f"  Context: {n_ctx}")
    print(f"  Address: http://{host}:{port}")
    print()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("  Waiting for server to be ready...", end="", flush=True)
    for i in range(60):
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=2)
            if resp.status_code == 200:
                print(" ready!")
                return proc
        except requests.exceptions.ConnectionError:
            pass

        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            print(f"\n  ERROR: Server exited with code {proc.returncode}")
            if stderr:
                print(f"  stderr: {stderr[:500]}")
            sys.exit(1)

        print(".", end="", flush=True)
        time.sleep(2)

    print("\n  ERROR: Server did not start within 120 seconds.")
    proc.terminate()
    sys.exit(1)


def format_tool_calls(calls: list[dict]) -> str:
    """Pretty-format parsed tool calls for display."""
    lines = []
    for call in calls:
        name = call.get("name", "unknown")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        args_str = json.dumps(args, indent=2)
        lines.append(f"  {name}({args_str})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Interactive inference with fine-tuned GGUF model")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--gguf", default=None, help="Path to GGUF file (auto-detects from output/gguf/)")
    parser.add_argument("--server-running", action="store_true", help="Skip server startup, connect to existing")
    parser.add_argument("--host", default=None, help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens per response")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking/reasoning tags")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    export_cfg = cfg["export"]

    host = args.host or infer_cfg["server_host"]
    port = args.port or infer_cfg["server_port"]
    system_prompt = data_cfg["system_prompt"]
    enable_thinking = infer_cfg.get("enable_thinking", True) and not args.no_think

    gen_params = {
        "max_tokens": args.max_tokens or infer_cfg.get("max_tokens", 512),
        "temperature": args.temperature if args.temperature is not None else infer_cfg.get("temperature", 0.6),
        "top_p": infer_cfg.get("top_p", 0.95),
        "top_k": infer_cfg.get("top_k", 20),
        "min_p": infer_cfg.get("min_p", 0.0),
        "repeat_penalty": infer_cfg.get("repeat_penalty", 1.0),
    }

    tool_config_path = resolve_path(args.config, data_cfg["tool_config_path"])
    tools = load_tools(tool_config_path)
    print(f"Loaded {len(tools)} tool schemas from {tool_config_path}")

    gguf_path = args.gguf
    if not gguf_path and not args.server_running:
        gguf_dir = resolve_path(args.config, export_cfg["gguf_output_dir"])
        gguf_path = find_gguf(gguf_dir)
        if not gguf_path:
            print(f"ERROR: No GGUF file found in {gguf_dir}")
            print("Run inference/export_gguf.py first, or specify --gguf path/to/model.gguf")
            sys.exit(1)

    server_proc = None
    if not args.server_running:
        server_proc = start_server(
            llamacpp_path=export_cfg["llamacpp_path"],
            gguf_path=gguf_path,
            host=host,
            port=port,
            gpu_layers=infer_cfg["gpu_layers"],
            n_ctx=infer_cfg["n_ctx"],
            flash_attn=infer_cfg.get("flash_attn", False),
        )
    else:
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Connected to existing server at http://{host}:{port}")
            else:
                print(f"WARNING: Server responded with status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Cannot connect to server at http://{host}:{port}")
            print("Start the server first or remove --server-running")
            sys.exit(1)

    messages: list[dict] = []

    print()
    print("=" * 60)
    print("  Interactive Inference")
    print("  Type a message, or use a command:")
    print("    /reset   - clear conversation")
    print("    /system  - show system prompt info")
    print("    /quit    - exit")
    print("=" * 60)
    print()

    def cleanup():
        if server_proc:
            print("\nStopping llama-server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("Server stopped.")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/reset":
                messages.clear()
                print("  Conversation cleared.\n")
                continue
            elif user_input.lower() == "/system":
                print(f"  System prompt: {system_prompt[:80]}...")
                print(f"  Tools loaded: {len(tools)}")
                tool_names = [t["function"]["name"] for t in tools]
                print(f"  Tool names: {', '.join(tool_names)}")
                print(f"  Thinking: {'enabled' if enable_thinking else 'disabled'}")
                print(f"  History: {len(messages)} messages")
                print(f"  Generation params:")
                for k, v in gen_params.items():
                    print(f"    {k}: {v}")
                print()
                continue

            messages.append({"role": "user", "content": user_input})

            prompt = build_prompt(system_prompt, tools, messages, enable_thinking)

            try:
                raw_content, timings = stream_response(
                    prompt, host, port, gen_params, enable_thinking,
                )
            except requests.exceptions.ConnectionError:
                print("  ERROR: Lost connection to server.")
                messages.pop()
                continue
            except requests.exceptions.Timeout:
                print("  ERROR: Request timed out.")
                messages.pop()
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                messages.pop()
                continue

            _, content = strip_thinking(raw_content)
            tool_calls = parse_tool_calls(content)

            messages.append({"role": "assistant", "content": content})

            if tool_calls:
                print(f"  [Tool Call{'s' if len(tool_calls) > 1 else ''}]")
                print(format_tool_calls(tool_calls))

            if timings:
                prompt_tokens = timings.get("prompt_n", 0)
                gen_tokens = timings.get("predicted_n", 0)
                gen_speed = timings.get("predicted_per_second", 0)
                prompt_speed = timings.get("prompt_per_second", 0)
                print(f"  [{gen_tokens} tokens | {gen_speed:.1f} t/s | prompt: {prompt_tokens} tokens @ {prompt_speed:.1f} t/s]")

            print()

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
