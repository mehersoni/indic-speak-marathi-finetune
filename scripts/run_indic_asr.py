"""
Run Marathi Automatic Speech Recognition (ASR) on synthesized audio.
Transcribes 24 kHz WAV files using Wav2Vec2 and computes empirical CER/WER against references.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

DEFAULT_GROUND_TRUTHS = {
    "00": "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.",
    "01": "मॅडम, काही मदत हवी आहे का?",
    "02": "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.",
    "03": "शिक्षण हे मानवी जीवनाचा पाया आहे.",
}


def levenshtein_distance(s1: list, s2: list) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_metrics(ref: str, hyp: str) -> tuple[float, float]:
    ref_c = list(re.sub(r"[^\w]", "", ref))
    hyp_c = list(re.sub(r"[^\w]", "", hyp))
    cer = levenshtein_distance(ref_c, hyp_c) / max(1, len(ref_c))

    ref_w = re.sub(r"[^\w\s]", "", ref).split()
    hyp_w = re.sub(r"[^\w\s]", "", hyp).split()
    wer = levenshtein_distance(ref_w, hyp_w) / max(1, len(ref_w))
    return cer, wer


def transcribe_and_evaluate(audio_dir: str, model_id: str = "sumedh/wav2vec2-large-xlsr-marathi"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ASR model '{model_id}' on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device)

    path = Path(audio_dir)
    wav_files = sorted([f for f in path.glob("*.wav") if not f.name.startswith("finetuned_")])
    if not wav_files:
        wav_files = sorted(path.glob("*.wav"))

    print(f"\n--- Running Indic ASR Evaluation: {audio_dir} ---")
    print(f"{'File':<26} | {'CER':<6} | {'WER':<6} | {'Reference vs ASR Hypothesis'}")
    print("-" * 80)

    total_cer, total_wer = [], []
    for wf in wav_files:
        wav, sr = sf.read(str(wf))
        if sr != 16000:
            target_len = int(len(wav) * 16000 / sr)
            wav = scipy.signal.resample(wav, target_len)
            sr = 16000

        inputs = processor(wav, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        hyp = processor.batch_decode(predicted_ids)[0].strip()

        # Find matching key
        idx = None
        for k in DEFAULT_GROUND_TRUTHS:
            if f"_{k}" in wf.name:
                idx = k
                break

        if idx and idx in DEFAULT_GROUND_TRUTHS:
            ref = DEFAULT_GROUND_TRUTHS[idx]
            cer, wer = compute_metrics(ref, hyp)
            total_cer.append(cer)
            total_wer.append(wer)
            print(f"{wf.name:<26} | {cer*100:>4.1f}% | {wer*100:>4.1f}% | Ref: {ref}")
            print(f"{'':<26} | {'':<6} | {'':<6} | Hyp: {hyp}\n")

    if total_cer:
        print(f"Overall Mean CER: {np.mean(total_cer)*100:.2f}% | Mean WER: {np.mean(total_wer)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Marathi TTS audio using an Indic ASR model")
    parser.add_argument("--audio-dir", default="audio/model_3/finetune_normalised", help="Directory of WAV files")
    parser.add_argument("--model", default="sumedh/wav2vec2-large-xlsr-marathi", help="Hugging Face Marathi ASR model ID")
    args = parser.parse_args()
    transcribe_and_evaluate(args.audio_dir, args.model)


if __name__ == "__main__":
    main()
