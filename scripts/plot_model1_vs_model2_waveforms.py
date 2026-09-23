"""Generates high-resolution waveform comparison plots between Model 1 (Run 1 Fixed) and Model 2 (Run 2 Production).
Contrasts time-domain speech amplitude, cadence, and energy profiles across all 4 benchmark Marathi sentences.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("figures", exist_ok=True)

sentences = [
    ("00", "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.", "Sentence 00 (Statement, Long)"),
    ("01", "मॅडम, काही मदत हवी आहे का?", "Sentence 01 (Question, Short)"),
    ("02", "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.", "Sentence 02 (Complex Compound)"),
    ("03", "शिक्षण हे मानवी जीवनाचा पाया आहे.", "Sentence 03 (Formal Declarative)"),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 10.5), dpi=300, sharex=False)
fig.suptitle(
    "Waveform Comparison: Model 1 (1.2k Samples, Attn-Only) vs Model 2 (3.5k Samples, Attn + MLP)",
    fontsize=13, fontweight="bold", y=0.98, color="#0f172a"
)

for idx, (num, text, title_desc) in enumerate(sentences):
    f1 = f"outputs/examples/Model1/finetuned_{num}.wav"
    f2 = f"outputs/examples/Model2/finetuned_{num}.wav"

    w1, sr1 = sf.read(f1)
    w2, sr2 = sf.read(f2)

    t1 = np.linspace(0, len(w1) / sr1, len(w1))
    t2 = np.linspace(0, len(w2) / sr2, len(w2))

    rms1 = float(np.sqrt(np.mean(w1**2)))
    peak1 = float(np.max(np.abs(w1)))
    dur1 = len(w1) / sr1

    rms2 = float(np.sqrt(np.mean(w2**2)))
    peak2 = float(np.max(np.abs(w2)))
    dur2 = len(w2) / sr2

    # Left Column: Model 1
    ax1 = axes[idx, 0]
    ax1.plot(t1, w1, color="#f59e0b", alpha=0.85, linewidth=0.6)
    ax1.set_title(
        f"{title_desc} — Model 1 | {dur1:.2f}s | RMS: {rms1:.4f} | Peak: {peak1:.3f}",
        fontsize=9.5, fontweight="bold", color="#b45309"
    )
    ax1.set_ylabel("Amplitude", fontsize=8)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right Column: Model 2
    ax2 = axes[idx, 1]
    ax2.plot(t2, w2, color="#0284c7", alpha=0.85, linewidth=0.6)
    ax2.set_title(
        f"{title_desc} — Model 2 | {dur2:.2f}s | RMS: {rms2:.4f} | Peak: {peak2:.3f}",
        fontsize=9.5, fontweight="bold", color="#0369a1"
    )
    ax2.set_ylabel("Amplitude", fontsize=8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)

    if idx == 3:
        ax1.set_xlabel("Time (seconds)", fontsize=9)
        ax2.set_xlabel("Time (seconds)", fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.38)

out_file = "figures/waveform_comparison_model1_vs_model2.png"
plt.savefig(out_file)
plt.close()
print(f"Successfully generated {out_file}")
