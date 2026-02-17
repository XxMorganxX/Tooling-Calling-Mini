"""
train.py -- Unsloth QLoRA fine-tuning for Qwen3 tool calling.

Loads the preprocessed dataset, applies LoRA adapters to the model,
and trains with completion-only masking (loss on assistant tokens only).

Usage:
    python train.py                          # from training/ directory
    python train.py --config ../config.yaml  # explicit config path
"""

import argparse
import os
import sys
from pathlib import Path

import unsloth  # must be imported before trl/transformers for Unsloth optimizations

import torch
import yaml
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel


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


IGNORE_INDEX = -100


class PaddingCollator:
    """Right-pads pre-tokenized examples (input_ids, attention_mask, labels) to batch max length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in examples)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ex in examples:
            ids = list(ex["input_ids"])
            mask = list(ex["attention_mask"])
            labels = list(ex["labels"])
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(labels + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def detect_precision(cfg: dict) -> dict:
    """Resolve fp16/bf16 flags based on GPU capability."""
    if cfg["training"].get("auto_detect_precision", False):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return {"fp16": False, "bf16": True}
        else:
            return {"fp16": True, "bf16": False}
    return {
        "fp16": cfg["training"].get("fp16", False),
        "bf16": cfg["training"].get("bf16", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3 tool-calling QLoRA training")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    export_cfg = cfg["export"]

    dataset_path = resolve_path(args.config, data_cfg["processed_dataset_path"] + ".parquet")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Processed dataset not found at {dataset_path}")
        print("Run data/prepare_dataset.py first.")
        sys.exit(1)

    print(f"Loading {model_cfg['model_name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg.get("dtype"),
    )

    if tokenizer.eos_token != "<|im_end|>":
        tokenizer.eos_token = "<|im_end|>"
        print(f"  Set eos_token to <|im_end|>")

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_rslora=lora_cfg["use_rslora"],
        use_gradient_checkpointing="unsloth",
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_params / total_params
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    print(f"Loading dataset from {dataset_path}...")
    full_dataset = Dataset.from_parquet(dataset_path)
    print(f"  {len(full_dataset)} total examples")

    eval_ratio = train_cfg.get("eval_split_ratio", 0.05)
    split = full_dataset.train_test_split(
        test_size=eval_ratio,
        seed=train_cfg["seed"],
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = PaddingCollator(pad_token_id=pad_id)

    precision = detect_precision(cfg)
    print(f"  Precision: fp16={precision['fp16']}, bf16={precision['bf16']}")

    output_dir = resolve_path(args.config, export_cfg["lora_adapter_dir"])
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        eval_accumulation_steps=train_cfg.get("eval_accumulation_steps", 4),
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optimizer"],
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
    )

    print("Starting training...")
    print(f"  Epochs: {train_cfg['num_train_epochs']}")
    eff_batch = train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']
    print(f"  Batch: {train_cfg['per_device_train_batch_size']} x {train_cfg['gradient_accumulation_steps']} = {eff_batch} effective")
    print(f"  Eval batch: {train_cfg.get('per_device_eval_batch_size', 1)}")
    print(f"  LR: {train_cfg['learning_rate']} ({train_cfg['lr_scheduler_type']})")
    print()

    train_result = trainer.train()

    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    print("\nTraining complete!")
    print(f"  Total steps: {metrics.get('total_flos', 'N/A')}")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Train runtime: {metrics.get('train_runtime', 0):.1f}s")

    print("\nRunning evaluation...")
    eval_metrics = trainer.evaluate()
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    print(f"\nAdapter saved to: {output_dir}")
    print("Next step: python inference/export_gguf.py")


if __name__ == "__main__":
    main()
