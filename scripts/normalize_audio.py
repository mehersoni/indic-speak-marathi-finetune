"""
Audio peak normalization script for generated TTS wav files.
Scales audio to target peak amplitude so all samples have consistent loudness.
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def normalize_file(wav_path: Path, target_peak: float = 0.90) -> None:
    data, sr = sf.read(str(wav_path))
    peak = np.max(np.abs(data))
    if peak < 1e-4:
        return
    normalized = (data / peak) * target_peak
    sf.write(str(wav_path), normalized, sr)
    rms_before = np.sqrt(np.mean(data**2))
    rms_after = np.sqrt(np.mean(normalized**2))
    print(f"Normalized {wav_path.name}: peak {peak:.3f} -> {target_peak:.2f}, RMS {rms_before:.4f} -> {rms_after:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Normalize WAV files to consistent peak level")
    parser.add_argument("path", help="WAV file or directory containing WAV files")
    parser.add_argument("--peak", type=float, default=0.90, help="Target peak amplitude (default: 0.90)")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file() and target.suffix.lower() == ".wav":
        normalize_file(target, args.peak)
    elif target.is_dir():
        for wav_file in sorted(target.rglob("*.wav")):
            normalize_file(wav_file, args.peak)


if __name__ == "__main__":
    main()
