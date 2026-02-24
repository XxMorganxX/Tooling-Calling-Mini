"""
run.py -- Single-file pipeline runner for Qwen3 tool-calling fine-tune.

Executes the full pipeline in order:
    1. prepare_dataset.py  (preprocess JSONL -> Qwen3 native format)
    2. train.py            (Unsloth QLoRA fine-tuning)
    3. export_gguf.py      (merge adapter + convert to GGUF)
    4. validate.py         (tool-calling accuracy test)

Usage:
    python run.py                        # full pipeline
    python run.py --start train          # skip data prep, start at training
    python run.py --stop export          # stop after export (skip validation)
    python run.py --start train --stop export
    python run.py --force                # force rebuild of cached dataset
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = ["prepare", "train", "export", "validate"]

STAGE_COMMANDS = {
    "prepare":  [sys.executable, os.path.join(SCRIPT_DIR, "data", "prepare_dataset.py")],
    "train":    [sys.executable, os.path.join(SCRIPT_DIR, "training", "train.py")],
    "export":   [sys.executable, os.path.join(SCRIPT_DIR, "inference", "export_gguf.py")],
    "validate": [sys.executable, os.path.join(SCRIPT_DIR, "inference", "validate.py")],
}

STAGE_LABELS = {
    "prepare":  "Data Preprocessing",
    "train":    "QLoRA Training",
    "export":   "GGUF Export",
    "validate": "Validation",
}


def run_stage(name: str, extra_args: list[str] = None):
    cmd = STAGE_COMMANDS[name][:]
    if extra_args:
        cmd.extend(extra_args)

    label = STAGE_LABELS[name]
    print(f"\n{'='*70}")
    print(f"  STAGE: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n{'!'*70}")
        print(f"  FAILED: {label} (exit code {result.returncode})")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'!'*70}")
        sys.exit(result.returncode)

    print(f"\n  Completed: {label} in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full fine-tuning pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--start", choices=STAGES, default="prepare",
                        help="Stage to start from (default: prepare)")
    parser.add_argument("--stop", choices=STAGES, default="validate",
                        help="Stage to stop after (default: validate)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild of cached dataset")
    parser.add_argument("--quant", default=None,
                        help="Override GGUF quantization type")
    args = parser.parse_args()

    start_idx = STAGES.index(args.start)
    stop_idx = STAGES.index(args.stop)

    if start_idx > stop_idx:
        print(f"ERROR: --start ({args.start}) is after --stop ({args.stop})")
        sys.exit(1)

    active_stages = STAGES[start_idx:stop_idx + 1]

    print(f"Pipeline: {' -> '.join(active_stages)}")
    print(f"Config: {args.config}")

    timings = {}
    total_start = time.time()

    for stage in active_stages:
        extra = ["--config", args.config]

        if stage == "prepare" and args.force:
            extra.append("--force")
        if stage == "export":
            if args.quant:
                extra.extend(["--quant", args.quant])

        timings[stage] = run_stage(stage, extra)

    total = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    for stage in active_stages:
        print(f"  {STAGE_LABELS[stage]:<25} {timings[stage]:>8.1f}s")
    print(f"  {'─'*35}")
    print(f"  {'Total':<25} {total:>8.1f}s")
    print()


if __name__ == "__main__":
    main()
