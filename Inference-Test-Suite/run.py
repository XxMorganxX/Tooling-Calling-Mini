"""
Inference Test Suite -- Validate tool-calling accuracy against training data.

Randomly samples examples from the training dataset, queries the model via
llama.cpp's /completion endpoint, and compares predicted tool calls against
the ground-truth labels from training.

Reports three accuracy tiers:
  - Strict:  correct tool names AND matching arguments in proper JSON format
  - Routing: correct tool names selected (regardless of format or arg values)
  - Format:  response used valid {"name":...,"arguments":...} JSON structure

Usage:
    python run.py                                    # defaults: 25 samples, localhost:8080
    python run.py --samples 50                       # more samples
    python run.py --host 192.168.1.75 --port 8080    # custom server
    python run.py --seed 42                          # reproducible run
    python run.py --verbose                          # show thinking + raw output
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DATA = SCRIPT_DIR / "training_data.jsonl"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SAMPLES = 25


# ── Data loading ────────────────────────────────────────────────────────────

def load_training_data(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_known_tool_names(examples: list[dict]) -> set[str]:
    """Pull tool names from the AVAILABLE TOOLS block in the system prompt."""
    system = next(
        (m["content"] for ex in examples for m in ex["messages"] if m["role"] == "system"),
        "",
    )
    match = re.search(r"AVAILABLE TOOLS:\s*\n(\[[\s\S]*?\n\])", system)
    if match:
        try:
            tools = json.loads(match.group(1))
            return {t["name"] for t in tools if "name" in t}
        except json.JSONDecodeError:
            pass
    return set()


def parse_expected_calls(assistant_content: str) -> list[dict]:
    """Extract expected tool calls from a training example's assistant message.

    Training format: <think>...</think> followed by a JSON array of tool call
    objects, or an empty array for no-tool responses.
    """
    stripped = re.sub(r"<think>.*?</think>", "", assistant_content, flags=re.DOTALL).strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        return []


def sample_examples(
    examples: list[dict],
    n: int,
    seed: int | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    n = min(n, len(examples))
    return rng.sample(examples, n)


# ── Prompt building ─────────────────────────────────────────────────────────

def build_prompt(system_content: str, user_content: str, enable_thinking: bool = True) -> str:
    """Build a Qwen3 chat prompt from the training example's system message.

    The system message in the training data already contains tool definitions
    in the AVAILABLE TOOLS section, so we use it verbatim.
    """
    prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    if enable_thinking:
        prompt += "<think>\n"
    return prompt


# ── Model querying ──────────────────────────────────────────────────────────

def query_model(
    prompt: str,
    host: str,
    port: int,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Send a prompt to llama-server and return (content, timings)."""
    url = f"http://{host}:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("content", ""), data.get("timings", {})


# ── Response parsing ────────────────────────────────────────────────────────

def strip_thinking(text: str) -> tuple[str, str]:
    """Separate <think> blocks from response content. Returns (thinking, rest)."""
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        thinking = re.sub(r"</?think>", "", before).strip()
        return thinking, after.strip()
    if "<think>" in text:
        _, _, after = text.partition("<think>")
        return after.strip(), ""
    return "", text.strip()


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a balanced JSON object from text starting at a '{' character."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class ParseResult:
    """Holds parsed tool calls and whether the output used proper format."""

    def __init__(self, calls: list[dict], proper_format: bool):
        self.calls = calls
        self.proper_format = proper_format


def parse_tool_calls(text: str, known_tools: set[str]) -> ParseResult:
    """Extract tool calls from model output.

    Tries structured formats first (proper JSON), then falls back to
    detecting the model's "bare tool name + args" malformed output.
    Returns a ParseResult with the calls and whether format was proper.
    """
    calls: list[dict] = []

    # Strategy 1: <tool_call>...</tool_call> tags
    for block in re.findall(r"<tool_call>([\s\S]*?)</tool_call>", text):
        stripped = block.strip()
        if not stripped:
            continue
        brace = stripped.find("{")
        if brace == -1:
            continue
        obj_str = _extract_json_object(stripped, brace)
        if obj_str:
            try:
                parsed = json.loads(obj_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    calls.append(parsed)
            except json.JSONDecodeError:
                pass
    if calls:
        return ParseResult(calls, proper_format=True)

    # Strategy 2: JSON array of {"name":...,"arguments":...} objects
    bracket = text.find("[")
    if bracket != -1:
        depth = 0
        for i in range(bracket, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[bracket : i + 1])
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "name" in item:
                                    calls.append(item)
                            if calls:
                                return ParseResult(calls, proper_format=True)
                    except json.JSONDecodeError:
                        pass
                    break

    # Strategy 3: bare JSON objects with name + arguments keys
    search_start = 0
    while search_start < len(text):
        brace = text.find("{", search_start)
        if brace == -1:
            break
        obj_str = _extract_json_object(text, brace)
        if obj_str is None:
            search_start = brace + 1
            continue
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
        search_start = brace + len(obj_str) if obj_str else brace + 1
    if calls:
        return ParseResult(calls, proper_format=True)

    # Strategy 4 (fallback): model emitted bare tool name + raw args.
    # Pattern: "tool_name\n{...}" or "tool_name\nkey: value" or just "tool_name"
    # on separate lines. This is a malformed output but we can recover tool selection.
    if known_tools:
        lines = text.strip().splitlines()
        found_names: list[str] = []
        remaining_text = text

        for tool_name in known_tools:
            pattern = re.compile(
                rf"(?:^|\n)\s*-?\s*\"?{re.escape(tool_name)}\"?\s*[:]?\s*(?:\n|$|[{{])",
                re.MULTILINE,
            )
            if pattern.search(remaining_text):
                brace_pos = remaining_text.find("{", remaining_text.find(tool_name))
                args = {}
                if brace_pos != -1:
                    obj_str = _extract_json_object(remaining_text, brace_pos)
                    if obj_str:
                        try:
                            args = json.loads(obj_str)
                        except json.JSONDecodeError:
                            pass
                calls.append({"name": tool_name, "arguments": args})

        if calls:
            return ParseResult(calls, proper_format=False)

        # Simplest fallback: just check if any known tool name appears as a word
        for tool_name in known_tools:
            if re.search(rf"\b{re.escape(tool_name)}\b", text):
                calls.append({"name": tool_name, "arguments": {}})
        if calls:
            return ParseResult(calls, proper_format=False)

    return ParseResult([], proper_format=True)


# ── Comparison logic ────────────────────────────────────────────────────────

def compare_tool_calls(
    actual: list[dict],
    expected: list[dict],
) -> tuple[str, str]:
    """Compare actual vs expected tool calls.

    Returns (grade, detail_string).
    Grade is one of: "exact", "args_diff", "routing_ok", "wrong", "no_tool_ok".
    """
    if not expected:
        if not actual:
            return "no_tool_ok", "correct: no tools called"
        names = [c.get("name", "?") for c in actual]
        return "wrong", f"expected no tools, got: {names}"

    if not actual:
        e_names = [c.get("name", "?") for c in expected]
        return "wrong", f"expected {e_names}, got nothing"

    e_names = sorted(c.get("name", "?") for c in expected)
    a_names = sorted(c.get("name", "?") for c in actual)

    if e_names != a_names:
        return "wrong", f"names: expected {e_names}, got {a_names}"

    # Names match -- now check arguments
    actual_by_name: dict[str, list[dict]] = {}
    for c in actual:
        actual_by_name.setdefault(c.get("name"), []).append(c)

    details = []
    all_exact = True
    for exp in expected:
        exp_name = exp["name"]
        exp_args = exp.get("arguments", {})
        candidates = actual_by_name.get(exp_name, [])
        if not candidates:
            return "wrong", f"missing tool: {exp_name}"

        act = candidates.pop(0)
        act_args = act.get("arguments", {})
        if isinstance(act_args, str):
            try:
                act_args = json.loads(act_args)
            except json.JSONDecodeError:
                details.append(f"{exp_name}: unparseable args")
                all_exact = False
                continue

        if act_args == exp_args:
            details.append(f"{exp_name}: exact")
        else:
            all_exact = False
            mismatches = []
            for k, v in exp_args.items():
                if k not in act_args:
                    mismatches.append(f"missing '{k}'")
                elif act_args[k] != v:
                    mismatches.append(f"'{k}': {v!r} -> {act_args[k]!r}")
            if mismatches:
                details.append(f"{exp_name}: {'; '.join(mismatches)}")
            else:
                details.append(f"{exp_name}: extra args ok")

    detail_str = " | ".join(details)
    if all_exact:
        return "exact", detail_str
    return "args_diff", detail_str


# ── Display ─────────────────────────────────────────────────────────────────

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def grade_color(grade: str) -> str:
    if grade in ("exact", "no_tool_ok"):
        return GREEN
    if grade == "args_diff":
        return YELLOW
    if grade == "routing_ok":
        return CYAN
    return RED


def grade_label(grade: str) -> str:
    return {
        "exact": "EXACT",
        "no_tool_ok": "EXACT",
        "args_diff": "ARGS",
        "routing_ok": "ROUTE",
        "wrong": "FAIL",
    }.get(grade, "FAIL")


def print_header(samples: int, total: int, host: str, port: int, seed: int | None):
    print()
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  Inference Test Suite{RESET}")
    print(f"  {samples} random samples from {total} training examples")
    print(f"  Server: {host}:{port}")
    if seed is not None:
        print(f"  Seed: {seed} (reproducible)")
    print()
    print(f"  {GREEN}EXACT{RESET} = correct tool + args   "
          f"{YELLOW}ARGS{RESET} = right tool, wrong args   "
          f"{CYAN}ROUTE{RESET} = right tool, bad format   "
          f"{RED}FAIL{RESET} = wrong")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print()


def print_result_line(idx: int, total: int, grade: str, fmt_ok: bool, user_input: str, detail: str, timings: dict | None = None):
    color = grade_color(grade)
    label = grade_label(grade)
    fmt_flag = "" if fmt_ok else f" {MAGENTA}!fmt{RESET}"
    truncated = user_input[:48] + ("..." if len(user_input) > 48 else "")
    speed = ""
    if timings and timings.get("predicted_per_second"):
        speed = f" {DIM}{timings['predicted_per_second']:.1f} t/s{RESET}"
    print(f"  {idx:>3}/{total}  [{color}{label:>5}{RESET}]{fmt_flag}  {truncated:<53} {DIM}{detail}{RESET}{speed}")


def print_summary(
    results: list[dict],
    per_tool: dict,
    elapsed: float,
):
    total = len(results)
    n_exact = sum(1 for r in results if r["grade"] in ("exact", "no_tool_ok"))
    n_args = sum(1 for r in results if r["grade"] == "args_diff")
    n_route = sum(1 for r in results if r["grade"] == "routing_ok")
    n_wrong = sum(1 for r in results if r["grade"] == "wrong")
    n_error = sum(1 for r in results if r["grade"] == "error")
    n_fmt_bad = sum(1 for r in results if not r["fmt_ok"])

    routing_correct = n_exact + n_args + n_route
    strict_correct = n_exact

    print()
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"  {BOLD}Results ({total} samples, {elapsed:.1f}s){RESET}")
    print()

    routing_pct = 100 * routing_correct / total if total else 0
    strict_pct = 100 * strict_correct / total if total else 0
    rc = GREEN if routing_pct >= 90 else YELLOW if routing_pct >= 70 else RED
    sc = GREEN if strict_pct >= 90 else YELLOW if strict_pct >= 70 else RED

    print(f"  Tool routing accuracy:  {rc}{BOLD}{routing_correct}/{total} ({routing_pct:.0f}%){RESET}"
          f"  (picked the right tool)")
    print(f"  Strict accuracy:        {sc}{BOLD}{strict_correct}/{total} ({strict_pct:.0f}%){RESET}"
          f"  (right tool + exact args)")
    print()
    print(f"    {GREEN}EXACT:{RESET} {n_exact:>3}   "
          f"{YELLOW}ARGS:{RESET} {n_args:>3}   "
          f"{CYAN}ROUTE:{RESET} {n_route:>3}   "
          f"{RED}FAIL:{RESET} {n_wrong:>3}   "
          f"errors: {n_error}")
    if n_fmt_bad:
        print(f"    {MAGENTA}Format issues:{RESET} {n_fmt_bad}/{total} responses used malformed output (bare tool name, no JSON wrapping)")

    # ── Throughput stats ──
    timings_list = [r["timings"] for r in results if r.get("timings")]
    if timings_list:
        gen_speeds = [t["predicted_per_second"] for t in timings_list if t.get("predicted_per_second")]
        prompt_speeds = [t["prompt_per_second"] for t in timings_list if t.get("prompt_per_second")]
        total_prompt_tokens = sum(t.get("prompt_n", 0) for t in timings_list)
        total_gen_tokens = sum(t.get("predicted_n", 0) for t in timings_list)

        print()
        print(f"  {BOLD}Throughput:{RESET}")
        print(f"    Tokens generated:  {total_gen_tokens:,}  ({total_prompt_tokens:,} prompt)")
        if gen_speeds:
            avg_gen = statistics.mean(gen_speeds)
            med_gen = statistics.median(gen_speeds)
            min_gen = min(gen_speeds)
            max_gen = max(gen_speeds)
            print(f"    Generation speed:  {avg_gen:.1f} t/s avg  |  {med_gen:.1f} median  |  {min_gen:.1f}–{max_gen:.1f} range")
        if prompt_speeds:
            avg_prompt = statistics.mean(prompt_speeds)
            print(f"    Prompt eval:       {avg_prompt:.1f} t/s avg")
        if total_gen_tokens and elapsed > 0:
            print(f"    End-to-end:        {total_gen_tokens / elapsed:.1f} t/s effective ({elapsed:.1f}s wall)")

    if per_tool:
        print()
        print(f"  {BOLD}Per-tool breakdown:{RESET}")
        print(f"    {'tool':<25} {'routing':>10}  {'strict':>10}")
        print(f"    {'─' * 25} {'─' * 10}  {'─' * 10}")
        for name in sorted(per_tool.keys()):
            if name == "__no_tool__":
                continue
            s = per_tool[name]
            r_pct = 100 * s["routed"] / s["total"] if s["total"] else 0
            s_pct = 100 * s["strict"] / s["total"] if s["total"] else 0
            rc = GREEN if r_pct >= 90 else YELLOW if r_pct >= 70 else RED
            sc = GREEN if s_pct >= 90 else YELLOW if s_pct >= 70 else RED
            print(f"    {name:<25} {rc}{s['routed']:>3}/{s['total']:<3} ({r_pct:>3.0f}%){RESET}  "
                  f"{sc}{s['strict']:>3}/{s['total']:<3} ({s_pct:>3.0f}%){RESET}")

        no_tool = per_tool.get("__no_tool__")
        if no_tool:
            pct = 100 * no_tool["strict"] / no_tool["total"] if no_tool["total"] else 0
            c = GREEN if pct >= 90 else RED
            print(f"    {'(no tool)':<25} {c}{no_tool['strict']:>3}/{no_tool['total']:<3} ({pct:>3.0f}%){RESET}")

    print(f"{BOLD}{'=' * 80}{RESET}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test tool-calling inference against training data")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Number of random samples to test (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"llama-server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"llama-server port (default: {DEFAULT_PORT})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking tags in prompt")
    parser.add_argument("--verbose", action="store_true", help="Show thinking and raw model output")
    parser.add_argument("--data", type=str, default=str(TRAINING_DATA), help="Path to training_data.jsonl")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        sys.exit(1)

    try:
        requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to llama-server at {args.host}:{args.port}")
        print("Start the server first, then re-run this script.")
        sys.exit(1)

    all_examples = load_training_data(data_path)
    known_tools = extract_known_tool_names(all_examples)
    samples = sample_examples(all_examples, args.samples, args.seed)
    enable_thinking = not args.no_think

    print_header(len(samples), len(all_examples), args.host, args.port, args.seed)

    results: list[dict] = []
    per_tool: dict[str, dict] = {}
    start_time = time.time()

    for i, example in enumerate(samples, 1):
        messages = example["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")

        expected = parse_expected_calls(assistant_msg)
        prompt = build_prompt(system_msg, user_msg, enable_thinking)

        try:
            raw_response, timings = query_model(
                prompt, args.host, args.port, args.max_tokens, args.temperature,
            )
        except Exception as e:
            print_result_line(i, len(samples), "error", True, user_msg, f"ERROR: {e}")
            results.append({"grade": "error", "fmt_ok": True})
            continue

        if args.verbose:
            thinking, _ = strip_thinking(raw_response)
            if thinking:
                print(f"\n    {DIM}[Think] {thinking[:200]}{'...' if len(thinking) > 200 else ''}{RESET}")
            print(f"    {DIM}[Raw]   {raw_response[:300]}{'...' if len(raw_response) > 300 else ''}{RESET}")

        _, content = strip_thinking(raw_response)
        parsed = parse_tool_calls(content, known_tools)
        actual = parsed.calls
        fmt_ok = parsed.proper_format

        grade, detail = compare_tool_calls(actual, expected)

        if not fmt_ok and grade in ("exact", "args_diff"):
            grade = "routing_ok"
            detail = f"[bad format] {detail}"

        results.append({"grade": grade, "fmt_ok": fmt_ok, "timings": timings})
        print_result_line(i, len(samples), grade, fmt_ok, user_msg, detail, timings)

        # Per-tool stats
        if expected:
            e_names_set = {c["name"] for c in expected}
            a_names_set = {c.get("name") for c in actual}
            for exp in expected:
                name = exp["name"]
                if name not in per_tool:
                    per_tool[name] = {"total": 0, "routed": 0, "strict": 0}
                per_tool[name]["total"] += 1
                if name in a_names_set:
                    per_tool[name]["routed"] += 1
                if grade == "exact":
                    per_tool[name]["strict"] += 1
        else:
            key = "__no_tool__"
            if key not in per_tool:
                per_tool[key] = {"total": 0, "routed": 0, "strict": 0}
            per_tool[key]["total"] += 1
            if grade == "no_tool_ok":
                per_tool[key]["routed"] += 1
                per_tool[key]["strict"] += 1

    elapsed = time.time() - start_time
    print_summary(results, per_tool, elapsed)


if __name__ == "__main__":
    main()
