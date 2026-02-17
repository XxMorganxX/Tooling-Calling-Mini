# Training

QLoRA fine-tuning for Qwen3 tool-calling using Unsloth.

## What It Does

1. Loads the pre-tokenized dataset produced by `data/prepare_dataset.py`
2. Loads the base model (e.g., Qwen3-4B-Instruct) with 4-bit quantization via Unsloth
3. Applies LoRA adapters to attention and MLP projection layers
4. Trains with completion-only masking (loss only on assistant response tokens)
5. Saves the LoRA adapter to `output/lora_adapter/`

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main script -- Unsloth QLoRA training with HuggingFace Trainer |

## Usage

```bash
# From the training/ directory
python train.py

# Use a specific config file
python train.py --config ../config.yaml
```

Or run as part of the full pipeline from the project root:

```bash
python run.py --start train --stop train
```

## Prerequisites

- The processed dataset must exist (run `data/prepare_dataset.py` first)
- NVIDIA GPU with CUDA (tested on RTX 2070+ / 16GB+ VRAM)
- Unsloth installed (`pip install -r requirements.txt`)

## Training Details

| Parameter | Default | Notes |
|-----------|---------|-------|
| Optimizer | adamw_8bit | Memory-efficient for QLoRA |
| Learning rate | 2e-4 | With cosine decay |
| Epochs | 5 | Multiple passes for small datasets |
| Batch size | 2 x 8 = 16 effective | Per-device x gradient accumulation |
| LoRA rank | 24 | ~2% trainable parameters on 4B model |
| LoRA alpha | 48 | Convention: 2x rank |
| Precision | Auto-detected | bf16 on Ampere+, fp16 on Turing |

## Output

The LoRA adapter is saved to `output/lora_adapter/` containing:
- `adapter_model.safetensors` -- LoRA weights
- `adapter_config.json` -- LoRA configuration
- Tokenizer files
- Training checkpoints (configurable retention)

## Dependencies

```bash
pip install -r requirements.txt
```

Requires: `unsloth`, `trl`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `pyyaml`

## Config

Reads from the shared `config.yaml` at the project root:

- `model.*` -- base model, sequence length, quantization
- `lora.*` -- rank, alpha, target modules, RSLoRA
- `training.*` -- optimizer, LR, epochs, batch size, precision
- `data.processed_dataset_path` -- where to find the tokenized dataset
- `export.lora_adapter_dir` -- where to save the trained adapter
