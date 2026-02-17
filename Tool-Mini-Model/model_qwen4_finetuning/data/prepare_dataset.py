"""
prepare_dataset.py -- Convert raw training JSONL into Qwen3 native tool-calling format.

Reads training data and tool schemas, extracts compact user/assistant pairs,
wraps them in Qwen3's native chat template with <tool_call> tokens, and
caches the result as a parquet file.

Usage:
    python prepare_dataset.py                        # from data/ directory
    python prepare_dataset.py --config ../config.yaml
    python prepare_dataset.py --force                # rebuild even if cache exists
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(config_path: str = None) -> dict:
    config_path = config_path or DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: str, relative_path: str) -> str:
    """Resolve a path from config relative to the config file's directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path or DEFAULT_CONFIG))
    return os.path.normpath(os.path.join(config_dir, relative_path))


def load_tools(tool_config_path: str) -> list[dict]:
    """Load tools from config and wrap in OpenAI-compatible function format."""
    with open(tool_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tools = []
    for tool in config["tools"]:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return tools


def parse_training_example(line: str) -> dict | None:
    """Extract user query and assistant response from a raw JSONL line."""
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    messages = data.get("messages", [])
    user_content = None
    assistant_content = None

    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    if user_content is None or assistant_content is None:
        return None

    return {"user": user_content, "assistant": assistant_content}


def extract_thinking(assistant_raw: str) -> tuple[str, str]:
    """Separate <think>...</think> block from the rest of the assistant response.

    Returns (thinking_text, remainder) where thinking_text includes the
    <think></think> tags (empty string if no thinking), and remainder is
    the JSON tool-call array.
    """
    match = re.search(r"(<think>.*?</think>)", assistant_raw, re.DOTALL)
    if match:
        thinking = match.group(1)
        remainder = assistant_raw[:match.start()] + assistant_raw[match.end():]
        return thinking, remainder.strip()
    return "", assistant_raw.strip()


def parse_tool_calls(json_str: str) -> list[dict]:
    """Parse a JSON array string into structured tool calls.

    Expects the raw JSON array (thinking already stripped):
        '[]'                                          -> no tools
        '[{"name": "weather", "arguments": {...}}]'   -> one tool
        '[{"name": "a", ...}, {"name": "b", ...}]'   -> multiple tools
    """
    try:
        calls = json.loads(json_str.strip())
    except json.JSONDecodeError:
        return []

    if not isinstance(calls, list):
        return []

    tool_calls = []
    for call in calls:
        if "name" not in call:
            continue
        tool_calls.append({
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call.get("arguments", {}),
            },
        })
    return tool_calls


def build_messages(
    system_prompt: str,
    user_content: str,
    thinking: str,
    tool_calls: list[dict],
) -> list[dict]:
    """Build a Qwen3-compatible message list for apply_chat_template.

    When thinking is present, it's placed in the assistant content field
    so the model trains on generating <think>...</think> before tool calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if tool_calls:
        messages.append({
            "role": "assistant",
            "content": thinking,
            "tool_calls": tool_calls,
        })
    else:
        messages.append({
            "role": "assistant",
            "content": thinking,
        })

    return messages


IGNORE_INDEX = -100


def tokenize_example(
    tokenizer,
    tools: list[dict],
    system_prompt: str,
    user_content: str,
    assistant_raw: str,
    max_seq_length: int,
) -> dict | None:
    """Tokenize one example and build labels with completion-only masking.

    Handles assistant responses with optional <think>...</think> blocks
    followed by a JSON tool-call array.

    Returns dict with input_ids, attention_mask, labels (all as lists),
    plus text for inspection.
    """
    thinking, json_remainder = extract_thinking(assistant_raw)
    tool_calls = parse_tool_calls(json_remainder)
    messages = build_messages(system_prompt, user_content, thinking, tool_calls)

    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"  Warning: chat template failed for '{user_content[:50]}...': {e}")
        return None

    prompt_messages = messages[:-1]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        print(f"  Warning: prompt template failed for '{user_content[:50]}...': {e}")
        return None

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]

    prompt_len = len(prompt_ids)
    labels = [IGNORE_INDEX] * min(prompt_len, len(full_ids))
    labels += full_ids[prompt_len:]

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": full_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen3 tool-calling dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    training_data_path = resolve_path(args.config, data_cfg["training_data_path"])
    tool_config_path = resolve_path(args.config, data_cfg["tool_config_path"])
    output_path = resolve_path(args.config, data_cfg["processed_dataset_path"])
    system_prompt = data_cfg["system_prompt"]

    cache_file = output_path + ".parquet"
    if os.path.exists(cache_file) and not args.force:
        print(f"Cache exists at {cache_file}. Use --force to rebuild.")
        print("Loading cached dataset...")
        ds = Dataset.from_parquet(cache_file)
        print(f"  {len(ds)} examples loaded.")
        return

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    print(f"Loading tools from {tool_config_path}...")
    tools = load_tools(tool_config_path)
    print(f"  {len(tools)} tools loaded: {[t['function']['name'] for t in tools]}")

    print(f"Loading tokenizer for {model_cfg['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=True,
    )
    print("  Tokenizer loaded.")

    print(f"Processing {training_data_path}...")
    examples = []
    skipped = 0
    tool_call_count = 0
    no_tool_count = 0
    thinking_count = 0

    with open(training_data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            parsed = parse_training_example(line)
            if parsed is None:
                skipped += 1
                continue

            result = tokenize_example(
                tokenizer=tokenizer,
                tools=tools,
                system_prompt=system_prompt,
                user_content=parsed["user"],
                assistant_raw=parsed["assistant"],
                max_seq_length=model_cfg["max_seq_length"],
            )

            if result is None:
                skipped += 1
                continue

            thinking, json_part = extract_thinking(parsed["assistant"])
            calls = parse_tool_calls(json_part)
            if calls:
                tool_call_count += 1
            else:
                no_tool_count += 1
            if thinking:
                thinking_count += 1

            examples.append(result)

    print(f"  Processed: {len(examples)} examples ({skipped} skipped)")
    print(f"  Tool calls: {tool_call_count} | No-tool: {no_tool_count}")
    print(f"  With thinking: {thinking_count} | Without: {len(examples) - thinking_count}")

    if not examples:
        print("ERROR: No examples processed. Check training data format.")
        sys.exit(1)

    print("Computing token statistics...")
    token_lengths = []
    for ex in examples:
        token_lengths.append(len(ex["input_ids"]))

    max_len = max(token_lengths)
    min_len = min(token_lengths)
    avg_len = sum(token_lengths) / len(token_lengths)
    over_limit = sum(1 for l in token_lengths if l > model_cfg["max_seq_length"])

    print(f"  Token lengths: min={min_len}, max={max_len}, avg={avg_len:.0f}")
    print(f"  Max seq length: {model_cfg['max_seq_length']}")
    if over_limit > 0:
        print(f"  WARNING: {over_limit} examples exceed max_seq_length and will be truncated!")
    else:
        print(f"  All examples fit within max_seq_length.")

    ds = Dataset.from_list(examples)
    ds.to_parquet(cache_file)
    print(f"  Dataset saved to {cache_file}")

    print("\n" + "=" * 60)
    print("SAMPLE (first tool-call example with thinking):")
    print("=" * 60)
    for ex in examples:
        if "<tool_call>" in ex["text"] and "<think>" in ex["text"]:
            print(ex["text"])
            break
    else:
        print("  (none found -- showing first tool-call example)")
        for ex in examples:
            if "<tool_call>" in ex["text"]:
                print(ex["text"])
                break

    print("\n" + "=" * 60)
    print("SAMPLE (first no-tool example):")
    print("=" * 60)
    for ex in examples:
        if "<tool_call>" not in ex["text"]:
            print(ex["text"])
            break


if __name__ == "__main__":
    main()
