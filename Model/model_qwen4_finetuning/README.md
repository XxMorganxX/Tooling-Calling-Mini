# Training Pipeline (`model_qwen4_finetuning/`)

End-to-end pipeline for fine-tuning Qwen3 on tool-calling data using QLoRA, then exporting a quantized GGUF model for llama.cpp inference.

## Pipeline Stages

```
prepare → train → export → validate
```

| Stage | Script | What It Does |
|-------|--------|-------------|
| `prepare` | `data/prepare_dataset.py` | JSONL → tokenized parquet with completion-only masking |
| `train` | `training/train.py` | QLoRA fine-tuning via Unsloth + HuggingFace Trainer |
| `export` | `inference/export_gguf.py` | Merge LoRA adapter into base model, convert to GGUF, quantize |
| `validate` | `inference/validate.py` | Run 19 test cases to verify tool-calling accuracy |

## Running

```bash
cd Model/model_qwen4_finetuning

# Full pipeline
python run.py

# Specific stages
python run.py --start train          # skip data prep
python run.py --stop export          # stop before validation
python run.py --start train --stop export
python run.py --force                # force rebuild of cached dataset
```

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | **Single source of truth** for all config (model registry, LoRA, training, export, inference, agent) |
| `run.py` | Pipeline runner — orchestrates all four stages |
| `tool_calling_config.json` | 12 tool schemas in OpenAI-compatible format — used by training prompts and inference |

## Subfolders

| Folder | Purpose | Has README |
|--------|---------|------------|
| `data/` | Dataset preparation (JSONL → parquet) and raw training data | Yes |
| `training/` | QLoRA training script | Yes |
| `output/` | Generated artifacts (LoRA adapter, merged model, GGUF files) | — |
| `llama.cpp/` | Git submodule of llama.cpp (for conversion tools) | — |

## Configuration (`config.yaml`)

| Section | Controls |
|---------|----------|
| `models` / `active_model` | Model registry and which model profile is active |
| `lora` | LoRA rank (24), alpha (48), target modules, RSLoRA |
| `training` | Optimizer, LR, epochs, batch size, precision, warmup |
| `data` | Paths to training data, tool schemas, system prompt |
| `export` | GGUF quantization type, llama.cpp path, output dirs |
| `inference` | llama-server settings, generation parameters |
| `agent` | Tool execution settings (enable, timeout) |

All paths in config are relative to this directory. Sub-projects resolve them via `resolve_path()`.

## Tool Schemas (`tool_calling_config.json`)

Contains the canonical JSON schemas for all 12 tools. These are:
- Injected into training prompts via `apply_chat_template(tools=...)`
- Loaded at inference time to build the system prompt
- Source of truth is the HomeAssist submodule at `../homeassist-ref/`

## Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| LoRA adapter | `output/lora_adapter/` | Trained adapter weights + tokenizer + checkpoints |
| Merged model | `output/merged_model/` | Full-precision merged base + LoRA weights |
| GGUF model | `output/gguf/` | Quantized model file for llama.cpp inference |
| Processed dataset | `data/output/processed_dataset.parquet` | Cached tokenized training data |
