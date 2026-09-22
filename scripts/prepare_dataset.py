"""
Downloads, inspects, and filters Marathi speech dataset with pre-encoded SNAC tokens.
Filters utterances exceeding 1400 total sequence tokens and computes distribution stats.
"""

import json
import os
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tokenize_format import build_training_sequence

MAX_SEQUENCE_LENGTH = 1400


def prepare_and_inspect_marathi_data():
    cache_dir = os.path.join(os.path.expanduser("~"), "indic-dataset-cache")
    local_path = os.path.join(cache_dir, "data_stage_1.parquet")

    if not os.path.exists(local_path):
        repo_id = "snorbyte/indic-tts-sample-snac-encoded"
        filename = "data_stage_1.parquet"
        token = os.environ.get("HF_TOKEN")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=cache_dir,
            token=token,
        )

    # Ensure a copy exists in the local working directory as well
    if not os.path.exists("data_stage_1.parquet") and os.path.exists(local_path):
        import shutil
        shutil.copyfile(local_path, "data_stage_1.parquet")

    print(f"Dataset path: {local_path}")
    df = pd.read_parquet(local_path)
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    marathi_df = df[df["language"].str.lower() == "marathi"].dropna(subset=["utterance", "snac_codes"]).copy()
    initial_count = len(marathi_df)
    print(f"\n--- Marathi Subset Statistics ---")
    print(f"Initial Marathi row count: {initial_count}")

    # Load tokenizer for sequence construction
    local_repo = os.path.join(os.path.expanduser("~"), "indic-speak-repo")
    model_id = local_repo if os.path.isdir(local_repo) else "bodhan-ai/indic-speak"
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))

    # Compute exact total sequence lengths via build_training_sequence and filter
    valid_indices = []
    dropped_indices = []
    all_sequence_lengths = []

    for idx, row in marathi_df.iterrows():
        gender_val = str(row.get("gender", "")).lower()
        speaker_name = "Anagha" if gender_val in ["woman", "female"] else "Chinmay"
        row_dict = {
            "utterance": str(row["utterance"]).strip(),
            "speaker": speaker_name,
            "snac_codes": row["snac_codes"],
        }
        seq = build_training_sequence(tok, row_dict)
        seq_len = len(seq["input_ids"])
        all_sequence_lengths.append(seq_len)

        if seq_len <= MAX_SEQUENCE_LENGTH:
            valid_indices.append(idx)
        else:
            dropped_indices.append((idx, seq_len))

    filtered_marathi_df = marathi_df.loc[valid_indices].copy()
    num_dropped = len(dropped_indices)
    num_remaining = len(filtered_marathi_df)

    print(f"\n--- Sequence Length Filtering (Threshold: {MAX_SEQUENCE_LENGTH} tokens) ---")
    print(f"Rows dropped (seq_len > {MAX_SEQUENCE_LENGTH}): {num_dropped} ({100.0 * num_dropped / initial_count:.2f}%)")
    print(f"Rows remaining (seq_len <= {MAX_SEQUENCE_LENGTH}): {num_remaining} ({100.0 * num_remaining / initial_count:.2f}%)")

    # Speaker distribution in filtered dataset
    speakers = filtered_marathi_df["user"].unique()
    print(f"\nFiltered Marathi Speaker Count: {len(speakers)}")
    print(f"Filtered Utterances per speaker:\n{filtered_marathi_df['user'].value_counts()}")

    # Distribution statistics on filtered sequence lengths
    filtered_lengths = [l for l in all_sequence_lengths if l <= MAX_SEQUENCE_LENGTH]
    lengths_arr = np.array(filtered_lengths)
    print(f"\nFiltered Sequence Length Distribution:")
    print(f"  Count:  {len(lengths_arr)}")
    print(f"  Min:    {lengths_arr.min()}")
    print(f"  Max:    {lengths_arr.max()}")
    print(f"  Mean:   {lengths_arr.mean():.1f}")
    print(f"  Median: {np.median(lengths_arr):.1f}")
    print(f"  p50:    {np.percentile(lengths_arr, 50):.1f}")
    print(f"  p75:    {np.percentile(lengths_arr, 75):.1f}")
    print(f"  p90:    {np.percentile(lengths_arr, 90):.1f}")
    print(f"  p95:    {np.percentile(lengths_arr, 95):.1f}")
    print(f"  p99:    {np.percentile(lengths_arr, 99):.1f}")


if __name__ == "__main__":
    prepare_and_inspect_marathi_data()
