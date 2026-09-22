"""
Tokenization and sequence formatting for Marathi speech fine-tuning.
Builds training sequences matching indic-speak's prompt and SNAC audio layout.
"""

import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import torch

# Ensure parent directory is in path when running as standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import (
    AUDIO_TOKEN_BASE,
    AUDIO_TOKEN_END,
    BOS_TOKEN_ID,
    CODEBOOK_SIZE,
    END_OF_AI_ID,
    END_OF_HUMAN_ID,
    END_OF_SPEECH_ID,
    EOT_TOKEN_ID,
    NUM_CODEBOOKS,
    SPEAKER_END_ID,
    SPEAKER_START_ID,
    START_OF_AI_ID,
    START_OF_HUMAN_ID,
    START_OF_SPEECH_ID,
)

# Style, environment, and non-verbal tokens are supported by the base model but omitted here as our dataset only contains text, speaker, and audio.


def codes_to_audio_tokens(codes: list[list[int]] | list[torch.Tensor] | list[np.ndarray]) -> list[int]:
    """Flattens 3 SNAC hierarchical codebooks (1:2:4 ratio) into 7-token interleaved frame IDs."""
    c0 = np.asarray(codes[0]).squeeze()
    c1 = np.asarray(codes[1]).squeeze()
    c2 = np.asarray(codes[2]).squeeze()

    n_frames = len(c0)
    audio_tokens: list[int] = []

    for i in range(n_frames):
        frame = [
            c0[i],
            c1[2 * i],
            c2[4 * i],
            c2[4 * i + 1],
            c1[2 * i + 1],
            c2[4 * i + 2],
            c2[4 * i + 3],
        ]
        for pos, code in enumerate(frame):
            token_id = AUDIO_TOKEN_BASE + pos * CODEBOOK_SIZE + int(code)
            audio_tokens.append(token_id)

    return audio_tokens


def audio_tokens_to_codes(audio_tokens: list[int], device: str = "cpu") -> list[torch.Tensor]:
    """Converts flat 7-token interleaved audio token IDs back to 3 SNAC codebook tensors."""
    audio = []
    for t in audio_tokens:
        if not (AUDIO_TOKEN_BASE <= t <= AUDIO_TOKEN_END):
            break
        audio.append(t)

    n_frames = len(audio) // NUM_CODEBOOKS
    if n_frames == 0:
        raise ValueError("Sequence contains no complete SNAC frame")

    a = np.array(audio[: n_frames * NUM_CODEBOOKS], dtype=np.int32).reshape(n_frames, NUM_CODEBOOKS)
    offsets = AUDIO_TOKEN_BASE + np.arange(NUM_CODEBOOKS, dtype=np.int32) * CODEBOOK_SIZE
    a = a - offsets

    c0 = torch.from_numpy(a[:, 0].copy()).long().unsqueeze(0).to(device)

    c1_arr = np.empty(n_frames * 2, dtype=np.int32)
    c1_arr[0::2] = a[:, 1]
    c1_arr[1::2] = a[:, 4]
    c1 = torch.from_numpy(c1_arr).long().unsqueeze(0).to(device)

    c2_arr = np.empty(n_frames * 4, dtype=np.int32)
    c2_arr[0::4] = a[:, 2]
    c2_arr[1::4] = a[:, 3]
    c2_arr[2::4] = a[:, 5]
    c2_arr[3::4] = a[:, 6]
    c2 = torch.from_numpy(c2_arr).long().unsqueeze(0).to(device)

    return [c0, c1, c2]


def build_training_sequence(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int | None = None,
) -> dict[str, list[int]]:
    """Builds tokenized training sequence for one dataset row with prompt masked out in labels."""
    text = str(row.get("utterance") or row.get("text", "")).strip()
    speaker = str(row.get("speaker") or row.get("user", "")).strip()

    raw_codes = row.get("snac_codes", [])
    if isinstance(raw_codes, str):
        raw_codes = json.loads(raw_codes)

    audio_tokens = codes_to_audio_tokens(raw_codes)

    nl = tokenizer.encode("\n", add_special_tokens=False)
    enc_text = tokenizer.encode(text, add_special_tokens=False)

    meta: list[int] = []
    if speaker:
        enc_speaker = tokenizer.encode(speaker, add_special_tokens=False)
        meta = [SPEAKER_START_ID] + enc_speaker + [SPEAKER_END_ID] + nl

    # Prompt sequence stops at start_of_ai
    prompt_tokens = [START_OF_HUMAN_ID, BOS_TOKEN_ID] + meta + enc_text + [EOT_TOKEN_ID, END_OF_HUMAN_ID, START_OF_AI_ID]
    # Prediction target starts with start_of_speech and ends with end_of_speech + end_of_ai
    target_tokens = [START_OF_SPEECH_ID] + audio_tokens + [END_OF_SPEECH_ID, END_OF_AI_ID]

    input_ids = prompt_tokens + target_tokens
    labels = [-100] * len(prompt_tokens) + target_tokens
    attention_mask = [1] * len(input_ids)

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    local_repo = os.path.join(os.path.expanduser("~"), "indic-speak-repo")
    tok = AutoTokenizer.from_pretrained(local_repo)

    # Example row matching Marathi dataset schema
    sample_row = {
        "utterance": "मॅडम, काही मदत हवी आहे का?",
        "speaker": "Anagha",
        "snac_codes": [
            [3981, 2068, 3981, 295, 2933],
            [426, 2426, 1736, 1736, 2609, 100, 200, 300, 400, 500],
            [3909, 3977, 1568, 3422, 3422, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        ],
    }

    seq = build_training_sequence(tok, sample_row)
    seq_len = len(seq["input_ids"])

    print(f"Sample Input IDs (first 25): {seq['input_ids'][:25]}")
    print(f"Sample Labels (first 25):    {seq['labels'][:25]}")
    print(f"Sample Labels (last 10):     {seq['labels'][-10:]}")
    print(f"Sequence Length: {seq_len}")

    # Flag check against Marathi length distribution (p99 is ~1295 audio tokens, max is 2170 audio tokens)
    TYPICAL_P99_LENGTH = 1400
    if seq_len > TYPICAL_P99_LENGTH:
        print(f"WARNING: Sequence length ({seq_len}) exceeds typical p99 threshold ({TYPICAL_P99_LENGTH}).")
    else:
        print(f"Length check: OK (within typical range).")
