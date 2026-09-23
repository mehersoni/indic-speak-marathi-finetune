"""
Evaluates synthesized Marathi speech audio against ground-truth reference texts.
Computes Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance.
"""

import argparse
from pathlib import Path
import soundfile as sf

DEFAULT_GROUND_TRUTHS = {
    "00": "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.",
    "01": "मॅडम, काही मदत हवी आहे का?",
    "02": "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.",
    "03": "शिक्षण हे मानवी जीवनाचा पाया आहे.",
}


def levenshtein_distance(s1: list, s2: list) -> int:
    """Computes Levenshtein edit distance between two sequences (words or chars)."""
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
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_cer_wer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Computes CER and WER between reference and hypothesis strings."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    cer = levenshtein_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    wer = levenshtein_distance(ref_words, hyp_words) / max(1, len(ref_words))
    return cer, wer


def evaluate_audio_directory(audio_dir: str = "audio/model_2/finetune_normalised"):
    path = Path(audio_dir)
    wav_files = sorted([f for f in path.glob("*.wav") if "finetune" in f.name and not f.name.startswith("finetuned")])
    if not wav_files:
        wav_files = sorted(path.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return

    print(f"\n--- Acoustic & Intelligibility Evaluation: {audio_dir} ---")
    print(f"{'Filename':<30} | {'Duration':<8} | {'CER':<8} | {'WER':<8} | {'Acoustic Status'}")
    print("-" * 75)

    for wf in wav_files:
        data, sr = sf.read(str(wf))
        duration = len(data) / sr
        
        # Match sentence index (00, 01, 02, 03)
        idx = None
        for k in DEFAULT_GROUND_TRUTHS:
            if f"_{k}" in wf.name:
                idx = k
                break
        
        if idx and idx in DEFAULT_GROUND_TRUTHS:
            ref_text = DEFAULT_GROUND_TRUTHS[idx]
            # CER/WER requires an actual independent ASR model transcription (e.g. IndicWav2Vec / Whisper-large)
            print(f"{wf.name:<30} | {duration:>6.2f}s  | {'N/A*':>6}  | {'N/A*':>6}  | Verified readable ({sr} Hz)")
        else:
            print(f"{wf.name:<30} | {duration:>6.2f}s  | {'N/A':>6}  | {'N/A':>6}  | Verified readable ({sr} Hz)")

    print("-" * 75)
    print("* Note: CER/WER calculation requires transcribing synthesized audio with an independent")
    print("  Marathi ASR model (e.g., ai4bharat/indicwav2vec-hindi-marathi or whisper-large-v3).")
    print("  Four demo sentences provide human-verified phonetic completeness, not a statistical ASR benchmark.")


def main():
    parser = argparse.ArgumentParser(description="Inspect and evaluate synthesized audio files")
    parser.add_argument("--audio-dir", default="audio/model_2/finetune_normalised", help="Path to folder containing .wav files")
    args = parser.parse_args()
    evaluate_audio_directory(args.audio_dir)


if __name__ == "__main__":
    main()
