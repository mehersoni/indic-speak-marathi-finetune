"""Generates waveform and spectrogram comparison plots between base and fine-tuned Marathi TTS models.
Produces publication-quality figures for documentation and submission.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("figures", exist_ok=True)

sentences = [
    ("00", "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत."),
    ("01", "मॅडम, काही मदत हवी आहे का?"),
    ("02", "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे."),
    ("03", "शिक्षण हे मानवी जीवनाचा पाया आहे."),
]

# --- 1. Waveform Comparison Plot ---
fig, axes = plt.subplots(4, 2, figsize=(14, 10), dpi=300, sharex=False)
fig.suptitle("Waveform Comparison: Base vs LoRA Fine-Tuned (Run 2)", fontsize=14, fontweight="bold", y=0.98)

for idx, (num, text) in enumerate(sentences):
    base_file = f"outputs/examples/base_{num}.wav"
    fine_file = f"outputs/examples/finetuned_{num}.wav"

    # Base audio
    base_wav, sr_b = sf.read(base_file)
    t_base = np.linspace(0, len(base_wav) / sr_b, len(base_wav))
    ax_b = axes[idx, 0]
    ax_b.plot(t_base, base_wav, color="#64748b", alpha=0.8, linewidth=0.6)
    ax_b.set_title(f"Sentence {num} (Base Model) — {len(base_wav)/sr_b:.2f}s", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Amplitude", fontsize=8)
    ax_b.set_ylim(-1.0, 1.0)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Fine-tuned audio
    fine_wav, sr_f = sf.read(fine_file)
    t_fine = np.linspace(0, len(fine_wav) / sr_f, len(fine_wav))
    ax_f = axes[idx, 1]
    ax_f.plot(t_fine, fine_wav, color="#0284c7", alpha=0.85, linewidth=0.6)
    ax_f.set_title(f"Sentence {num} (LoRA Fine-Tuned) — {len(fine_wav)/sr_f:.2f}s", fontsize=10, fontweight="bold")
    ax_f.set_ylabel("Amplitude", fontsize=8)
    ax_f.set_ylim(-1.0, 1.0)
    ax_f.grid(True, linestyle=":", alpha=0.5)

    if idx == 3:
        ax_b.set_xlabel("Time (seconds)", fontsize=9)
        ax_f.set_xlabel("Time (seconds)", fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.35)
plt.savefig("figures/waveform_comparison.png")
plt.close()
print("Saved figures/waveform_comparison.png")

# --- 2. Spectrogram Comparison Plot for Sentence 00 & 01 ---
fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=300)
fig.suptitle("Mel-Scale Spectrogram Comparison: Base vs LoRA Fine-Tuned", fontsize=13, fontweight="bold", y=0.98)

for row_idx, num in enumerate(["00", "01"]):
    base_file = f"outputs/examples/base_{num}.wav"
    fine_file = f"outputs/examples/finetuned_{num}.wav"

    base_wav, sr_b = sf.read(base_file)
    fine_wav, sr_f = sf.read(fine_file)

    # Base Spectrogram
    f_b, t_b, Sxx_b = scipy.signal.spectrogram(base_wav, sr_b, nperseg=512, noverlap=256)
    ax_b = axes[row_idx, 0]
    pcm_b = ax_b.pcolormesh(t_b, f_b, 10 * np.log10(Sxx_b + 1e-10), cmap="magma", shading="gouraud", vmin=-80, vmax=0)
    ax_b.set_title(f"Sentence {num} — Base Model Spectrogram", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Frequency (Hz)", fontsize=8)
    ax_b.set_ylim(0, 8000)

    # Fine-tuned Spectrogram
    f_f, t_f, Sxx_f = scipy.signal.spectrogram(fine_wav, sr_f, nperseg=512, noverlap=256)
    ax_f = axes[row_idx, 1]
    pcm_f = ax_f.pcolormesh(t_f, f_f, 10 * np.log10(Sxx_f + 1e-10), cmap="magma", shading="gouraud", vmin=-80, vmax=0)
    ax_f.set_title(f"Sentence {num} — LoRA Fine-Tuned Spectrogram", fontsize=10, fontweight="bold")
    ax_f.set_ylabel("Frequency (Hz)", fontsize=8)
    ax_f.set_ylim(0, 8000)

    if row_idx == 1:
        ax_b.set_xlabel("Time (seconds)", fontsize=9)
        ax_f.set_xlabel("Time (seconds)", fontsize=9)

fig.colorbar(pcm_f, ax=axes, orientation="horizontal", fraction=0.04, pad=0.1, label="Power Spectral Density (dB)")
plt.savefig("figures/spectrogram_comparison.png")
plt.close()
print("Saved figures/spectrogram_comparison.png")
