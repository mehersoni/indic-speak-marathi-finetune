"""
Plot comprehensive multi-metric comparison between Base model and Fine-Tuned models.
Visualizes duration pacing, acoustic energy (RMS), and speech rate alignment.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sentences = ["Sent 00 (Standard)", "Sent 01 (Question)", "Sent 02 (Compound)", "Sent 03 (Formal)"]

# Durations (seconds)
dur_base = [3.75, 3.93, 5.55, 3.16]
dur_m1   = [5.97, 4.44, 8.28, 4.18]
dur_m2   = [6.14, 3.50, 7.34, 3.58]
dur_m3   = [5.55, 3.58, 6.31, 4.10]

# RMS Energy
rms_base = [0.0699, 0.0576, 0.0704, 0.0757]
rms_m1   = [0.1863, 0.0551, 0.0880, 0.0336]
rms_m2   = [0.1997, 0.0600, 0.0222, 0.0260]
rms_m3   = [0.2306, 0.1103, 0.1830, 0.2328]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
fig.suptitle(
    "Empirical Comparison: Base Model vs LoRA Fine-Tuned Models (Runs 1, 2, 3)",
    fontsize=13, fontweight="bold", y=0.98, color="#0f172a"
)

x = np.arange(len(sentences))
w = 0.20

# Panel 1: Duration & Pacing
ax1.bar(x - 1.5*w, dur_base, width=w, label="Base Model (Untuned)", color="#94a3b8", edgecolor="#64748b")
ax1.bar(x - 0.5*w, dur_m1,   width=w, label="Model 1 (1.2k, Attn)", color="#f59e0b", edgecolor="#d97706")
ax1.bar(x + 0.5*w, dur_m2,   width=w, label="Model 2 (3.5k, Attn+MLP)", color="#0284c7", edgecolor="#0369a1")
ax1.bar(x + 1.5*w, dur_m3,   width=w, label="Model 3 (6.8k, Full Corpus)", color="#10b981", edgecolor="#059669")

ax1.set_title("Synthesis Duration & Pacing (Seconds)", fontsize=11, fontweight="bold", color="#1e293b")
ax1.set_xticks(x)
ax1.set_xticklabels(sentences, fontsize=8.5)
ax1.set_ylabel("Duration (s)", fontsize=10)
ax1.set_ylim(0, 9.5)
ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)

# Panel 2: Acoustic Energy (RMS Power)
ax2.bar(x - 1.5*w, rms_base, width=w, label="Base Model (Untuned)", color="#94a3b8", edgecolor="#64748b")
ax2.bar(x - 0.5*w, rms_m1,   width=w, label="Model 1 (1.2k, Attn)", color="#f59e0b", edgecolor="#d97706")
ax2.bar(x + 0.5*w, rms_m2,   width=w, label="Model 2 (3.5k, Attn+MLP)", color="#0284c7", edgecolor="#0369a1")
ax2.bar(x + 1.5*w, rms_m3,   width=w, label="Model 3 (6.8k, Full Corpus)", color="#10b981", edgecolor="#059669")

ax2.set_title("Acoustic Energy: Root-Mean-Square (RMS Power)", fontsize=11, fontweight="bold", color="#1e293b")
ax2.set_xticks(x)
ax2.set_xticklabels(sentences, fontsize=8.5)
ax2.set_ylabel("RMS Amplitude", fontsize=10)
ax2.set_ylim(0, 0.28)
ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
out_path = Path("figures/base_vs_finetuned_comparison.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Successfully generated {out_path}")
