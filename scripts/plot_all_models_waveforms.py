"""
Plot waveform comparison across Model 1, Model 2, and Model 3 fine-tuned audio.
Generates a 4x3 comparison grid displaying duration, RMS, and peak amplitude per sentence.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sentences = [
    ("00", "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.", "Sentence 00 (Standard)"),
    ("01", "मॅडम, काही मदत हवी आहे का?", "Sentence 01 (Question)"),
    ("02", "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.", "Sentence 02 (Compound)"),
    ("03", "शिक्षण हे मानवी जीवनाचा पाया आहे.", "Sentence 03 (Formal)"),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 11), dpi=300, sharex=False)
fig.suptitle(
    "Waveform Comparison: Model 1 (1.2k Samples) vs Model 2 (3.5k Samples) vs Model 3 (6.8k Samples Full Corpus)",
    fontsize=13, fontweight="bold", y=0.98, color="#0f172a"
)

models = [
    ("Model 1 (1.2k Attn)", Path("audio/model_1/finetune"), "#f59e0b", "#b45309"),
    ("Model 2 (3.5k Attn+MLP)", Path("audio/model_2/finetune"), "#0284c7", "#0369a1"),
    ("Model 3 (6.8k Full)", Path("audio/model_3/finetune"), "#10b981", "#047857"),
]

for row_idx, (num, text, title_desc) in enumerate(sentences):
    for col_idx, (model_label, model_dir, line_color, text_color) in enumerate(models):
        wav_path = model_dir / f"finetune_{num}.wav"
        wav, sr = sf.read(str(wav_path))
        t = np.linspace(0, len(wav) / sr, len(wav))
        rms = float(np.sqrt(np.mean(wav**2)))
        peak = float(np.max(np.abs(wav)))
        dur = len(wav) / sr

        ax = axes[row_idx, col_idx]
        ax.plot(t, wav, color=line_color, alpha=0.85, linewidth=0.6)
        ax.set_title(
            f"{title_desc} — {model_label} | {dur:.2f}s | RMS: {rms:.3f} | Peak: {peak:.2f}",
            fontsize=8.5, fontweight="bold", color=text_color
        )
        if col_idx == 0:
            ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 3:
            ax.set_xlabel("Time (seconds)", fontsize=8)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
out_fig = Path("figures/waveform_comparison_all_models.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Successfully generated {out_fig}")
