# Inference Test Suite

Validates tool-calling accuracy by sampling from the training dataset, querying the model via llama.cpp, and comparing predicted tool calls against ground-truth labels.

## What It Measures

Three accuracy tiers:

| Tier | What It Checks |
|------|---------------|
| **Strict** | Correct tool names AND matching argument values in proper JSON format |
| **Routing** | Correct tool names selected (regardless of argument values or format) |
| **Format** | Response used valid `{"name": ..., "arguments": ...}` JSON structure |

## Running

```bash
cd Inference-Test-Suite
pip install -r requirements.txt

# Default: 25 random samples against localhost:8080
python run.py

# Custom options
python run.py --samples 50                    # more samples
python run.py --host 10.0.0.50 --port 8080   # remote server
python run.py --seed 42                       # reproducible run
python run.py --verbose                       # show thinking + raw output
```

Requires a running llama-server instance (the test suite talks directly to llama.cpp's `/completion` endpoint, not the FastAPI server).

## Files

| File | Purpose |
|------|---------|
| `run.py` | Main test runner — sampling, inference, comparison, reporting |
| `training_data.jsonl` | Copy of training data used as ground truth |
| `test_examples.jsonl` | Curated test examples |
| `generate_test_data.py` | Generates test data from training set |
| `requirements.txt` | Python dependencies |

## How It Works

1. Loads all examples from `training_data.jsonl`
2. Randomly samples N examples (default 25)
3. For each example, builds the full prompt (system + tools + user message)
4. Sends to llama-server's `/completion` endpoint
5. Parses tool calls from the response
6. Compares against the expected tool calls from training data
7. Reports per-example results and aggregate accuracy across all three tiers

## Output

The runner prints a per-example breakdown (pass/fail at each tier) and a summary table with aggregate percentages. Use `--verbose` to see the model's raw output including thinking traces.

## Relationship to `inference/validate.py`

The `Model/inference/validate.py` script is a different validation tool that runs 19 hardcoded test cases. This test suite instead samples randomly from the full training dataset for statistical coverage.
