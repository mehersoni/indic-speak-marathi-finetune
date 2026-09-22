"""
Dataset loading, parsing, and splitting for Marathi speech fine-tuning.
Extracts Marathi utterances, decodes SNAC tokens, and creates train/val splits.
"""

import json
from typing import Any
import pandas as pd
from torch.utils.data import Dataset

from src.tokenize_format import build_training_sequence

# Map dataset gender to official Indic-Speak Marathi voice names.
# Indic-Speak requires a closed set of known voice names from its library (Anagha/Chinmay for Marathi);
# unseen speaker strings yield unconditioned/degraded voice priors. We explicitly collapse the 9 raw
# dataset user IDs into 2 voice labels by gender so fine-tuning reinforces the model's existing Marathi voice priors.
GENDER_TO_SPEAKER = {
    "woman": "Anagha",
    "female": "Anagha",
    "man": "Chinmay",
    "male": "Chinmay",
}


def parse_snac_codes(raw_codes: str | list) -> list[list[int]]:
    """Parses JSON-encoded or nested list SNAC codes into 3 codebook lists [c0, c1, c2]."""
    if isinstance(raw_codes, str):
        return json.loads(raw_codes)
    return raw_codes


class MarathiSpeechDataset(Dataset):
    """PyTorch Dataset yielding formatted tokenized training sequences."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        return build_training_sequence(
            tokenizer=self.tokenizer,
            row=item,
            max_length=self.max_length,
        )


def load_marathi_splits(
    parquet_path: str,
    train_size: int = 1200,
    val_size: int = 100,
    max_sequence_length: int | None = 1400,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loads parquet data, filters Marathi rows, and returns (train_records, val_records)."""
    df = pd.read_parquet(parquet_path)
    marathi_df = df[df["language"].str.lower() == "marathi"].copy()

    # Drop any rows missing utterance or snac_codes
    marathi_df = marathi_df.dropna(subset=["utterance", "snac_codes"])

    if max_sequence_length is not None:
        # Filter rows whose SNAC audio tokens + text prompt would exceed max_sequence_length
        # Fast pre-filter on SNAC code lengths: 7 * len(c0) + estimated prompt <= max_sequence_length
        valid_rows = []
        for idx, row in marathi_df.iterrows():
            codes = parse_snac_codes(row["snac_codes"])
            audio_tokens_count = len(codes[0]) * 7 if isinstance(codes, list) and len(codes) > 0 else 0
            # Allow 100 tokens margin for prompt text
            if audio_tokens_count + 100 <= max_sequence_length:
                valid_rows.append(idx)
        marathi_df = marathi_df.loc[valid_rows]

    # Shuffle deterministically
    shuffled_df = marathi_df.sample(n=len(marathi_df), random_state=seed).reset_index(drop=True)

    total_requested = train_size + val_size
    if len(shuffled_df) < total_requested:
        raise ValueError(f"Dataset has {len(shuffled_df)} rows, fewer than requested {total_requested}")

    train_df = shuffled_df.iloc[:train_size]
    val_df = shuffled_df.iloc[train_size : train_size + val_size]

    def _to_records(subset_df: pd.DataFrame) -> list[dict[str, Any]]:
        records = []
        for _, row in subset_df.iterrows():
            gender_val = str(row.get("gender", "")).lower()
            speaker = GENDER_TO_SPEAKER.get(gender_val, "Anagha")

            records.append({
                "utterance": str(row["utterance"]).strip(),
                "speaker": speaker,
                "snac_codes": parse_snac_codes(row["snac_codes"]),
                "user_id": int(row.get("user", 0)) if pd.notna(row.get("user")) else 0,
                "gender": gender_val,
            })
        return records

    return _to_records(train_df), _to_records(val_df)
