"""
Downloads and inspects Marathi speech dataset with pre-encoded SNAC tokens.
Computes dataset statistics, speaker distributions, and token length quantiles.
"""

import json
import os
import sys
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


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

    print(f"Dataset path: {local_path}")
    df = pd.read_parquet(local_path)
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    marathi_df = df[df["language"].str.lower() == "marathi"].copy()
    print(f"\n--- Marathi Subset Statistics ---")
    print(f"Row count: {len(marathi_df)}")

    # Speaker count & unique speakers
    speakers = marathi_df["user"].unique()
    print(f"Speaker count (unique 'user' IDs): {len(speakers)}")
    print(f"Speakers: {list(speakers)}")
    print(f"Utterances per speaker:\n{marathi_df['user'].value_counts()}")

    # Sample transcripts
    print(f"\nSample transcripts (first 3):")
    for idx, text in enumerate(marathi_df["utterance"].head(3)):
        print(f"  {idx + 1}. {text}")

    # Inspect SNAC code structure for one row
    raw_sample = marathi_df["snac_codes"].iloc[0]
    sample_codes = json.loads(raw_sample) if isinstance(raw_sample, str) else raw_sample
    print(f"\nSNAC structure for first row:")
    print(f"  Container: list of {len(sample_codes)} codebook levels")
    print(f"  Codebook lengths (c0, c1, c2): {[len(c) for c in sample_codes]}")
    all_sample_codes = [code for c in sample_codes for code in c]
    sample_arr = np.array(all_sample_codes, dtype=np.int32)
    print(f"  Dtype: {sample_arr.dtype}")
    print(f"  Min code value: {sample_arr.min()}, Max code value: {sample_arr.max()}")
    print(f"  Interleaved token count (frames * 7): {len(sample_codes[0]) * 7}")

    # Compute sequence lengths across Marathi subset
    lengths = []
    for val in marathi_df["snac_codes"]:
        codes = json.loads(val) if isinstance(val, str) else val
        # Total tokens = sum of c0, c1, c2 elements = 7 * len(c0)
        total_tokens = sum(len(c) for c in codes)
        lengths.append(total_tokens)

    lengths_arr = np.array(lengths)
    print(f"\nToken Sequence Length Distribution (SNAC tokens per utterance):")
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
