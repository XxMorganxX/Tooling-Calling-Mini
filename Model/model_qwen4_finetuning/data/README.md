# Data Preparation

Converts raw training JSONL into Qwen3's native tool-calling format with pre-tokenized tensors ready for training.

## What It Does

1. Reads `training_data.jsonl` -- raw user/assistant pairs where assistant responses are JSON arrays of tool calls
2. Loads tool schemas from the shared `tool_calling_config.json` at the project root
3. Wraps each example in Qwen3's chat template with `<tool_call>` XML tokens
4. Tokenizes with completion-only masking (prompt tokens get label `-100`, only assistant tokens contribute to loss)
5. Caches the result as a parquet file for fast loading during training

## Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Main script -- processes JSONL into tokenized parquet |
| `training_data.jsonl` | Raw training data (user queries + expected tool calls) |
| `output/` | Created at runtime -- holds `processed_dataset.parquet` |

## Usage

```bash
# From the data/ directory
python prepare_dataset.py

# Force rebuild even if cache exists
python prepare_dataset.py --force

# Use a specific config file
python prepare_dataset.py --config ../config.yaml
```

Or run as part of the full pipeline from the project root:

```bash
python run.py --start prepare --stop prepare
```

## Training Data Format

Each line in `training_data.jsonl` is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather tomorrow?"},
    {"role": "assistant", "content": "[{\"name\": \"weather\", \"arguments\": {\"specific_date\": \"tomorrow\"}}]"}
  ]
}
```

- Assistant content is a JSON array of tool calls (or `[]` for no-tool responses)
- Each tool call has `name` and `arguments` fields

## Output

The processed dataset at `output/processed_dataset.parquet` contains:

| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | list[int] | Full tokenized sequence |
| `attention_mask` | list[int] | 1 for real tokens, 0 for padding |
| `labels` | list[int] | `-100` for prompt tokens, token IDs for assistant tokens |
| `text` | str | Human-readable text for inspection |

## Dependencies

```bash
pip install -r requirements.txt
```

Requires: `transformers`, `datasets`, `sentencepiece`, `pyyaml`

## Config

Reads from the shared `config.yaml` at the project root:

- `model.model_name` -- which tokenizer to load
- `model.max_seq_length` -- truncation limit
- `data.training_data_path` -- path to raw JSONL
- `data.tool_config_path` -- path to tool schemas
- `data.processed_dataset_path` -- where to save output
- `data.system_prompt` -- system message injected into each example
