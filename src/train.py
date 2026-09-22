"""
Training entrypoint for fine-tuning Indic-Speak on Marathi speech data with LoRA.
Loads model/tokenizer, attaches LoRA adapters, and trains using Hugging Face Trainer.
"""

import argparse
from pathlib import Path
import sys
from typing import Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import yaml

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import MarathiSpeechDataset, load_marathi_splits
from src.utils import PAD_TOKEN_ID


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_linear_attention_modules(model: torch.nn.Module) -> list[str]:
    """Finds all unique linear attention module target names in the model."""
    attention_target_candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf_name = name.split(".")[-1]
            if leaf_name in attention_target_candidates:
                found_modules.add(leaf_name)

    return sorted(list(found_modules))


def data_collator(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """Pads input_ids, labels, and attention_mask to the maximum length in the batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        padded_input_ids.append(x["input_ids"] + [PAD_TOKEN_ID] * pad_len)
        padded_labels.append(x["labels"] + [-100] * pad_len)
        padded_attention_mask.append(x["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
    }


def train(config_path: str = "configs/marathi_lora.yaml"):
    cfg = load_config(config_path)

    model_id = cfg.get("model_name_or_path", "bodhan-ai/indic-speak")
    print(f"Loading tokenizer and model from: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|pad|>"
        tokenizer.pad_token_id = PAD_TOKEN_ID

    # Use fp16 precision for Kaggle T4 compatibility (no native bf16 support on T4)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    print("\n--- Model Named Modules ---")
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # print leaf modules
            print(f"  {name}: {mod.__class__.__name__}")

    # Discover and configure LoRA attention target modules
    target_modules = find_linear_attention_modules(model)
    print(f"\nIdentified linear attention target modules: {target_modules}")

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters count and percentage
    trainable_params, all_params = model.get_nb_trainable_parameters()
    pct = 100.0 * trainable_params / all_params
    print(f"\nTrainable parameters: {trainable_params:,} / {all_params:,} ({pct:.4f}%)")

    # Load dataset
    dataset_path = cfg.get("dataset_path", "data_stage_1.parquet")
    print(f"\nLoading dataset from: {dataset_path}")
    train_records, val_records = load_marathi_splits(
        parquet_path=dataset_path,
        train_size=cfg.get("train_size", 1200),
        val_size=cfg.get("val_size", 100),
        seed=cfg.get("seed", 42),
    )
    print(f"Dataset split: {len(train_records)} train, {len(val_records)} validation samples")

    max_length = cfg.get("max_sequence_length", cfg.get("max_length", 1400))
    train_dataset = MarathiSpeechDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = MarathiSpeechDataset(val_records, tokenizer, max_length=max_length)

    # Configure training arguments with periodic checkpoints, gradient checkpointing, and fp16
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "outputs/lora_marathi"),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        fp16=cfg.get("fp16", True),
        bf16=False,
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 50),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 50),
        logging_steps=cfg.get("logging_steps", 10),
        save_total_limit=cfg.get("save_total_limit", 3),
        seed=cfg.get("seed", 42),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    final_output_dir = Path(training_args.output_dir) / "final_adapter"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"Training completed. Final adapter saved to {final_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Indic-Speak LoRA on Marathi dataset")
    parser.add_argument("--config", default="configs/marathi_lora.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
