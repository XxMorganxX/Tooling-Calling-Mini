"""
validate.py -- Test tool-calling accuracy of the fine-tuned GGUF model.

Runs a suite of test prompts against the model via llama.cpp's /completion
endpoint, parses <tool_call> blocks, and compares against expected output.

Usage:
    python validate.py                             # from inference/ directory
    python validate.py --config ../config.yaml
    python validate.py --server-running            # if server is already up
    python validate.py --gguf path/to/model.gguf   # specific GGUF file
"""

import argparse
import json
import os
import re
import subprocess
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


# ── Test cases ──────────────────────────────────────────────────────────
# Each test: (user_input, expected_tool_calls)
# expected_tool_calls is a list of {"name": str, "arguments": dict}
# Use None for arguments to skip argument validation (check name only)

TEST_CASES = [
    # --- Single tool calls ---
    (
        "What's the weather tomorrow?",
        [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}],
    ),
    (
        "Weather for the next 3 days",
        [{"name": "weather", "arguments": {"days": 3}}],
    ),
    (
        "Play some Drake",
        [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Drake", "search_type": "artist"}}],
    ),
    (
        "Pause the music",
        [{"name": "spotify_playback", "arguments": {"action": "pause"}}],
    ),
    (
        "Set volume to 50",
        [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 50}}],
    ),
    (
        "Turn on Light 1",
        [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}],
    ),
    (
        "Set the bedroom to movie mode",
        [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "bedroom"}}],
    ),
    (
        "What's on my calendar today?",
        [{"name": "calendar_data", "arguments": None}],
    ),
    (
        "Search for best coffee shops",
        [{"name": "google_search", "arguments": None}],
    ),
    (
        "Read my sticky notes",
        [{"name": "stickies", "arguments": {"action": "read"}}],
    ),
    (
        "Text me to take out the trash",
        [{"name": "send_sms", "arguments": None}],
    ),
    (
        "Any new emails?",
        [{"name": "get_notifications", "arguments": None}],
    ),
    # --- Multi-tool calls ---
    (
        "Turn on the living room lights and play jazz",
        [
            {"name": "kasa_lighting", "arguments": None},
            {"name": "spotify_playback", "arguments": None},
        ],
    ),
    (
        "What's the weather and what's on my calendar?",
        [
            {"name": "weather", "arguments": None},
            {"name": "calendar_data", "arguments": None},
        ],
    ),
    # --- No-tool cases ---
    (
        "Thanks!",
        [],
    ),
    (
        "Hello, how are you?",
        [],
    ),
    (
        "Got it",
        [],
    ),
    (
        "Okay",
        [],
    ),
    (
        "Never mind",
        [],
    ),
]


def parse_tool_calls_from_response(text: str) -> list[dict]:
    """Extract tool calls from model output containing <tool_call> blocks."""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            calls.append(parsed)
        except json.JSONDecodeError:
            continue

    return calls


def build_prompt(
    tokenizer_or_none,
    tools: list[dict],
    system_prompt: str,
    user_content: str,
) -> str:
    """Build a Qwen3 chat-formatted prompt for the /completion endpoint."""
    prompt = "<|im_start|>system\n"
    prompt += system_prompt + "\n\n"
    prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    prompt += "You are provided with function signatures within XML tags:\n<tools>\n"
    for tool in tools:
        prompt += json.dumps(tool) + "\n"
    prompt += "</tools>\n\n"
    prompt += "For each function call, return a json object with function name and arguments within XML tags:\n"
    prompt += '<tool_call>\n{"name": <function_name>, "arguments": <args>}\n</tool_call>'
    prompt += "<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    return prompt


def query_model(
    prompt: str,
    host: str,
    port: int,
    infer_cfg: dict,
) -> str:
    """Send prompt to llama.cpp /completion endpoint."""
    url = f"http://{host}:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": infer_cfg.get("max_tokens", 256),
        "temperature": 0.0,
        "top_p": infer_cfg.get("top_p", 0.95),
        "top_k": infer_cfg.get("top_k", 20),
        "min_p": infer_cfg.get("min_p", 0.0),
        "repeat_penalty": infer_cfg.get("repeat_penalty", 1.0),
        "stop": ["<|im_end|>", "<|im_start|>"],
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("content", "")
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to llama-server"
    except Exception as e:
        return f"ERROR: {e}"


def compare_tool_calls(
    actual: list[dict],
    expected: list[dict],
) -> tuple[bool, str]:
    """Compare actual vs expected tool calls. Returns (pass, detail)."""
    if not expected:
        if not actual:
            return True, "Correct: no tools called"
        else:
            names = [c.get("name", "?") for c in actual]
            return False, f"Expected no tools, got: {names}"

    if len(actual) != len(expected):
        a_names = [c.get("name", "?") for c in actual]
        e_names = [c.get("name", "?") for c in expected]
        return False, f"Count mismatch: expected {e_names}, got {a_names}"

    actual_by_name = {c.get("name"): c for c in actual}
    details = []

    for exp in expected:
        exp_name = exp["name"]
        if exp_name not in actual_by_name:
            return False, f"Missing tool: {exp_name}"

        act = actual_by_name[exp_name]

        if exp["arguments"] is None:
            details.append(f"{exp_name}: name OK (args unchecked)")
            continue

        act_args = act.get("arguments", {})
        if isinstance(act_args, str):
            try:
                act_args = json.loads(act_args)
            except json.JSONDecodeError:
                return False, f"{exp_name}: cannot parse arguments"

        if act_args == exp["arguments"]:
            details.append(f"{exp_name}: exact match")
        else:
            mismatches = []
            for k, v in exp["arguments"].items():
                if k not in act_args:
                    mismatches.append(f"missing '{k}'")
                elif act_args[k] != v:
                    mismatches.append(f"'{k}': expected {v}, got {act_args[k]}")

            if mismatches:
                return False, f"{exp_name}: {'; '.join(mismatches)}"
            else:
                details.append(f"{exp_name}: match (extra args OK)")

    return True, " | ".join(details)


def load_tools_for_prompt(tool_config_path: str) -> list[dict]:
    """Load tools in OpenAI-compatible format for prompt building."""
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


def main():
    parser = argparse.ArgumentParser(description="Validate tool-calling accuracy")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--gguf", default=None, help="Path to specific GGUF file")
    parser.add_argument("--server-running", action="store_true",
                        help="Skip server startup (assume already running)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    export_cfg = cfg["export"]

    host = infer_cfg["server_host"]
    port = infer_cfg["server_port"]
    system_prompt = data_cfg["system_prompt"]

    tool_config_path = resolve_path(args.config, data_cfg["tool_config_path"])
    tools = load_tools_for_prompt(tool_config_path)

    gguf_path = args.gguf
    if not gguf_path:
        gguf_dir = resolve_path(args.config, export_cfg["gguf_output_dir"])
        gguf_files = list(Path(gguf_dir).glob("*.gguf"))
        if gguf_files:
            gguf_path = str(gguf_files[0])
        else:
            print("ERROR: No GGUF file found. Run inference/export_gguf.py first.")
            sys.exit(1)

    print(f"Using GGUF: {gguf_path}")

    server_proc = None
    if not args.server_running:
        llamacpp_path = export_cfg["llamacpp_path"]
        server_bin = os.path.join(llamacpp_path, "llama-server.exe")
        if not os.path.exists(server_bin):
            server_bin = os.path.join(llamacpp_path, "llama-server")

        if not os.path.exists(server_bin):
            print(f"ERROR: llama-server not found at {llamacpp_path}")
            print("Use --server-running if server is already started.")
            sys.exit(1)

        print(f"Starting llama-server on {host}:{port}...")
        cmd = [
            server_bin,
            "-m", gguf_path,
            "--host", host,
            "--port", str(port),
            "-ngl", str(infer_cfg["gpu_layers"]),
            "-c", str(infer_cfg["n_ctx"]),
        ]
        if infer_cfg.get("flash_attn", False):
            cmd.extend(["--flash-attn", "on"])

        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print("  Waiting for server to start...")
        for i in range(30):
            try:
                resp = requests.get(f"http://{host}:{port}/health", timeout=2)
                if resp.status_code == 200:
                    print("  Server ready.")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        else:
            print("  ERROR: Server did not start within 60 seconds.")
            server_proc.terminate()
            sys.exit(1)

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    print(f"{'#':>3}  {'RESULT':>6}  {'INPUT':<45}  {'DETAIL'}")
    print("-" * 100)

    passed = 0
    failed = 0
    errors = 0
    per_tool_stats = {}

    for i, (user_input, expected) in enumerate(TEST_CASES, 1):
        prompt = build_prompt(None, tools, system_prompt, user_input)
        response = query_model(prompt, host, port, infer_cfg)

        if response.startswith("ERROR"):
            print(f"{i:>3}  {'ERROR':>6}  {user_input:<45}  {response}")
            errors += 1
            continue

        actual = parse_tool_calls_from_response(response)
        ok, detail = compare_tool_calls(actual, expected)

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"{i:>3}  {status:>6}  {user_input:<45}  {detail}")

        for exp in expected:
            name = exp["name"]
            if name not in per_tool_stats:
                per_tool_stats[name] = {"total": 0, "correct": 0}
            per_tool_stats[name]["total"] += 1
            if ok:
                per_tool_stats[name]["correct"] += 1

    total = passed + failed + errors
    print("\n" + "=" * 100)
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.1f}%) | {failed} failed | {errors} errors")

    if per_tool_stats:
        print("\nPer-tool accuracy:")
        for name, stats in sorted(per_tool_stats.items()):
            pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {name:<25} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    if server_proc:
        print("\nStopping llama-server...")
        server_proc.terminate()
        server_proc.wait(timeout=10)
        print("  Server stopped.")


if __name__ == "__main__":
    main()
