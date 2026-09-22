"""
Smoke test suite for Indic-Speak Marathi LoRA pipeline.
Tests single-sample and batched forward/backward passes and finite gradient checks.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_marathi_splits, parse_snac_codes
from src.tokenize_format import build_training_sequence
from src.train import data_collator, find_linear_attention_modules, load_config
from src.utils import PAD_TOKEN_ID


def check_finite_gradients(model: torch.nn.Module) -> tuple[bool, str]:
    """Checks if all trainable parameters have finite gradients (no NaN or Inf)."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                return False, f"Gradient is None for {name}"
            if torch.isnan(param.grad).any():
                return False, f"NaN gradient detected in {name}"
            if torch.isinf(param.grad).any():
                return False, f"Inf gradient detected in {name}"
    return True, "All gradients finite"


def build_sample_batch(tokenizer, n_samples: int = 4) -> dict[str, torch.Tensor]:
    """Loads n_samples from dataset or generates synthetic samples if dataset is absent."""
    candidate_paths = [
        "data_stage_1.parquet",
        os.path.join(os.path.expanduser("~"), "indic-dataset-cache", "data_stage_1.parquet"),
    ]
    dataset_file = None
    for p in candidate_paths:
        if os.path.exists(p):
            dataset_file = p
            break

    if dataset_file is not None:
        train_recs, _ = load_marathi_splits(dataset_file, train_size=n_samples, val_size=1)
        samples = [build_training_sequence(tokenizer, rec, max_length=512) for rec in train_recs[:n_samples]]
    else:
        samples = []
        for i in range(n_samples):
            mock_row = {
                "utterance": f"मॅडम, काही मदत हवी आहे का? {i}",
                "speaker": "Anagha",
                "snac_codes": [[3981, 2068, 295], [426, 2426, 1736, 100, 200, 300], [3909, 3977, 1568, 10, 20, 30, 40, 50, 60, 70, 80, 90]],
            }
            samples.append(build_training_sequence(tokenizer, mock_row, max_length=512))

    return data_collator(samples)


def run_smoke_test(config_path: str = "configs/marathi_lora.yaml"):
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    model_id = cfg.get("model_name_or_path", "bodhan-ai/indic-speak")

    # If running locally and local repo exists, use local snapshot
    local_repo = os.path.join(os.path.expanduser("~"), "indic-speak-repo")
    if os.path.exists(local_repo) and not os.path.exists(model_id):
        model_load_path = local_repo
    else:
        model_load_path = model_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float16 on GPU (as on T4) or float32 fallback on CPU for compatibility
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"--- Environment ---")
    print(f"Device: {device}")
    print(f"Precision: {model_dtype}")
    print(f"Loading model/tokenizer from: {model_load_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|pad|>"
        tokenizer.pad_token_id = PAD_TOKEN_ID

    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
    ).to(device)

    # Attach LoRA
    target_modules = find_linear_attention_modules(model)
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)
    all_stages_passed = True

    # --- Stage 1: Single example forward pass ---
    print("\n[Stage 1] Single Example Forward Pass:")
    batch_1 = build_sample_batch(tokenizer, n_samples=1)
    batch_1 = {k: v.to(device) for k, v in batch_1.items()}

    outputs = model(**batch_1)
    loss = outputs.loss
    loss_val = loss.item()

    if math.isfinite(loss_val):
        print(f"  PASS: Loss is finite float ({loss_val:.4f})")
    else:
        print(f"  FAIL: Loss is non-finite ({loss_val})")
        all_stages_passed = False

    # --- Stage 2: Single example forward + backward + optimizer step ---
    print("\n[Stage 2] Single Example Forward + Backward + Optimizer Step:")
    optimizer.zero_grad()
    outputs = model(**batch_1)
    loss = outputs.loss
    loss.backward()

    finite_grads, grad_msg = check_finite_gradients(model)
    if finite_grads:
        optimizer.step()
        print(f"  PASS: Gradients are finite; optimizer step completed successfully")
    else:
        print(f"  FAIL: Gradient check failed ({grad_msg})")
        all_stages_passed = False

    # --- Stage 3: Batch of 4 forward pass ---
    print("\n[Stage 3] Batch of 4 Forward Pass:")
    batch_4 = build_sample_batch(tokenizer, n_samples=4)
    batch_4 = {k: v.to(device) for k, v in batch_4.items()}

    outputs_4 = model(**batch_4)
    loss_4 = outputs_4.loss
    loss_4_val = loss_4.item()

    if math.isfinite(loss_4_val):
        print(f"  PASS: Batch loss is finite float ({loss_4_val:.4f})")
    else:
        print(f"  FAIL: Batch loss is non-finite ({loss_4_val})")
        all_stages_passed = False

    # --- Stage 4: Batch of 4 forward + backward + optimizer step ---
    print("\n[Stage 4] Batch of 4 Forward + Backward + Optimizer Step:")
    optimizer.zero_grad()
    outputs_4 = model(**batch_4)
    loss_4 = outputs_4.loss
    loss_4.backward()

    finite_grads_4, grad_msg_4 = check_finite_gradients(model)
    if finite_grads_4:
        optimizer.step()
        print(f"  PASS: Batch gradients are finite; optimizer step completed successfully")
    else:
        print(f"  FAIL: Batch gradient check failed ({grad_msg_4})")
        all_stages_passed = False

    print("\n==========================================")
    if all_stages_passed:
        print("ALL SMOKE TEST STAGES PASSED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED AT ONE OR MORE STAGES")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Indic-Speak smoke test")
    parser.add_argument("--config", default="configs/marathi_lora.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_smoke_test(config_path=args.config)
