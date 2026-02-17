"""
export_gguf.py -- Merge LoRA adapter into base model and convert to GGUF.

Uses the local llama.cpp installation for conversion and quantization:
  1. Merge LoRA adapter into base model (Unsloth)
  2. convert_hf_to_gguf.py -> f16 GGUF
  3. llama-quantize -> quantized GGUF

Usage:
    python export_gguf.py                         # from inference/ directory
    python export_gguf.py --config ../config.yaml
    python export_gguf.py --quant q8_0            # override quantization type
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml
from unsloth import FastLanguageModel

CONVERT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
)


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


def find_convert_script(llamacpp_path: str, project_root: str) -> str:
    """Locate convert_hf_to_gguf.py, downloading from GitHub if needed.

    Pre-built llama.cpp releases only ship binaries, not the Python
    convert script.  We check the install dir first, then fall back
    to a cached copy in the inference/ directory, and finally download it.
    """
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        path = os.path.join(llamacpp_path, name)
        if os.path.exists(path):
            return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "convert_hf_to_gguf.py")
    if os.path.exists(local_path):
        print(f"  Using cached convert script: {local_path}")
        return local_path

    project_root_path = os.path.join(project_root, "convert_hf_to_gguf.py")
    if os.path.exists(project_root_path):
        print(f"  Using convert script from project root: {project_root_path}")
        return project_root_path

    print(f"  convert_hf_to_gguf.py not found in {llamacpp_path}")
    print(f"  Downloading from llama.cpp GitHub...")
    try:
        urllib.request.urlretrieve(CONVERT_SCRIPT_URL, local_path)
        print(f"  Saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ERROR: Failed to download convert script: {e}")
        print(f"  URL: {CONVERT_SCRIPT_URL}")
        sys.exit(1)


def ensure_gguf_package():
    """Ensure the 'gguf' Python package is installed (required by convert script)."""
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("  Installing required 'gguf' Python package...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gguf"],
            stdout=subprocess.DEVNULL,
        )
        print("  Installed 'gguf' package.")


def find_quantize_bin(llamacpp_path: str) -> str:
    """Locate llama-quantize binary."""
    for name in ("llama-quantize.exe", "llama-quantize"):
        path = os.path.join(llamacpp_path, name)
        if os.path.exists(path):
            return path

    print(f"ERROR: llama-quantize not found in {llamacpp_path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapter to GGUF")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--quant", default=None, help="Override quantization type (e.g., q4_k_m, q8_0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    export_cfg = cfg["export"]

    adapter_dir = resolve_path(args.config, export_cfg["lora_adapter_dir"])
    merged_dir = resolve_path(args.config, export_cfg["merged_model_dir"])
    gguf_dir = resolve_path(args.config, export_cfg["gguf_output_dir"])
    quant_type = args.quant or export_cfg["gguf_quant_type"]
    llamacpp_path = export_cfg["llamacpp_path"]
    project_root = resolve_path(args.config, ".")

    safe_name = model_cfg["model_name"].replace("/", "_")
    f16_path = os.path.join(gguf_dir, f"{safe_name}-f16.gguf")
    quant_path = os.path.join(gguf_dir, f"{safe_name}-tool-calling-{quant_type}.gguf")

    if not os.path.exists(adapter_dir):
        print(f"ERROR: LoRA adapter not found at {adapter_dir}")
        print("Run training/train.py first.")
        sys.exit(1)

    if not os.path.isdir(llamacpp_path):
        print(f"ERROR: llama.cpp not found at {llamacpp_path}")
        print("Set export.llamacpp_path in config.yaml to your llama.cpp directory.")
        sys.exit(1)

    quantize_bin = find_quantize_bin(llamacpp_path)
    convert_script = find_convert_script(llamacpp_path, project_root)
    ensure_gguf_package()

    print(f"llama.cpp:  {llamacpp_path}")
    print(f"Quantize:   {quantize_bin}")
    print(f"Convert:    {convert_script}")
    print(f"Quant type: {quant_type}")
    print()

    merged_weights = list(Path(merged_dir).glob("model*.safetensors"))
    if merged_weights:
        print(f"Step 1: Merged model exists at {merged_dir} ({len(merged_weights)} shard(s)), skipping merge.")
    else:
        print(f"Step 1: Merging LoRA adapter into base model...")
        print(f"  Loading {adapter_dir}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir,
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"],
            dtype=model_cfg.get("dtype"),
        )
        os.makedirs(merged_dir, exist_ok=True)
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"  Saved to {merged_dir}")
        del model, tokenizer

    os.makedirs(gguf_dir, exist_ok=True)

    print(f"\nStep 2: Converting HF -> f16 GGUF...")
    cmd_convert = [
        sys.executable, convert_script,
        merged_dir,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)
    print(f"  f16 GGUF: {f16_path}")

    print(f"\nStep 3: Quantizing f16 -> {quant_type}...")
    cmd_quant = [quantize_bin, f16_path, quant_path, quant_type.upper()]
    print(f"  Running: {' '.join(cmd_quant)}")
    result = subprocess.run(cmd_quant)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)

    size_mb = Path(quant_path).stat().st_size / (1024 * 1024)
    print(f"\n  Quantized GGUF: {quant_path}")
    print(f"  Size: {size_mb:.1f} MB")

    if os.path.exists(f16_path):
        os.remove(f16_path)
        print(f"  Cleaned up intermediate f16 file.")

    print(f"\nExport complete: {quant_path}")
    print("Next step: python inference/inference.py")


if __name__ == "__main__":
    main()
