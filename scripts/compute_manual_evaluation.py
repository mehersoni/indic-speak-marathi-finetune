"""
Processes manual human evaluation scores across 10 benchmark sentences.
Calculates MOS statistics for Naturalness, Clarity, and Artifacts,
and generates comparative publication charts and summary tables.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Filled ratings from human listening evaluation
data = [
    {"id": "01", "text": "कदाचित आपण दोन्ही करू शकतो!", "base_n": 1.0, "r1_n": 3.0, "r2_n": 4.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 2.0, "r2_c": 4.0, "r3_c": 5.0, "base_a": 5.0, "r1_a": 2.0, "r2_a": 4.0, "r3_a": 4.0, "notes": ""},
    {"id": "02", "text": "ठीक आहे मग, मला मग नेहमीची अंडीच दे.", "base_n": 1.0, "r1_n": 3.0, "r2_n": 3.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 1.0, "r2_c": 1.0, "r3_c": 5.0, "base_a": 4.0, "r1_a": 4.0, "r2_a": 2.0, "r3_a": 4.5, "notes": "last 're' pronounced extra in base"},
    {"id": "03", "text": "३० सप्टेंबर १९६९ च्या दक्षिण मुंबईतील जुन्या-मोडकळीस आलेल्या इमारतींच्या पुनर्विकास आणि दुरूस्तीची संपूर्ण", "base_n": 1.0, "r1_n": 3.0, "r2_n": 3.0, "r3_n": 4.5, "base_c": 5.0, "r1_c": 3.0, "r2_c": 3.0, "r3_c": 4.0, "base_a": 5.0, "r1_a": 3.0, "r2_a": 4.0, "r3_a": 5.0, "notes": ""},
    {"id": "04", "text": "मनसेचे विभाग अध्यक्ष संतोष धुरी यांनी आश्वासन दिले आहे कि, “मनसे प्रमुख राज", "base_n": 1.0, "r1_n": 2.0, "r2_n": 3.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 2.0, "r2_c": 1.0, "r3_c": 4.0, "base_a": 5.0, "r1_a": 2.0, "r2_a": 3.0, "r3_a": 4.5, "notes": ""},
    {"id": "05", "text": "उन्हात लोखंड गरम होऊन ते किंचीत प्रसरण पावतं, तर हिवाळ्यात ते आकुंचन पावत", "base_n": 1.0, "r1_n": 2.0, "r2_n": 4.0, "r3_n": 4.5, "base_c": 5.0, "r1_c": 1.0, "r2_c": 3.0, "r3_c": 3.5, "base_a": 5.0, "r1_a": 3.0, "r2_a": 3.0, "r3_a": 4.5, "notes": ""},
    {"id": "06", "text": "सुरू आहे. यामुळे पुढील आठ दिवस ठाणे शहराला पिसे येथून होणारा पाणीपुरवठा कमी", "base_n": 1.0, "r1_n": 3.0, "r2_n": 3.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 2.0, "r2_c": 2.0, "r3_c": 4.0, "base_a": 5.0, "r1_a": 2.0, "r2_a": 3.0, "r3_a": 4.0, "notes": ""},
    {"id": "07", "text": "माझा पगार संजय राऊतांना देतो, त्यांनी त्यांचं घर चालवून दाखवावं", "base_n": 1.0, "r1_n": 4.0, "r2_n": 5.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 4.0, "r2_c": 5.0, "r3_c": 4.5, "base_a": 5.0, "r1_a": 4.0, "r2_a": 4.5, "r3_a": 4.5, "notes": ""},
    {"id": "08", "text": "मात्र पक्षाने त्यांना कोणतंही महत्वाचं पद न दिल्याने ते २०११ साली भाजपात दाखल", "base_n": 1.0, "r1_n": 3.5, "r2_n": 4.0, "r3_n": 4.0, "base_c": 4.5, "r1_c": 3.5, "r2_c": 3.5, "r3_c": 4.0, "base_a": 5.0, "r1_a": 4.0, "r2_a": 4.0, "r3_a": 4.5, "notes": "text not converted to speech after number in model 3"},
    {"id": "09", "text": "दसरा मेळाव्यातला प्रकार पूर्वनियोजित -जोशी", "base_n": 1.0, "r1_n": 3.0, "r2_n": 4.0, "r3_n": 5.0, "base_c": 5.0, "r1_c": 2.0, "r2_c": 4.0, "r3_c": 4.0, "base_a": 5.0, "r1_a": 3.0, "r2_a": 4.0, "r3_a": 4.0, "notes": "joshi not pronounced in model 1"},
    {"id": "10", "text": "हो, मी check out करतोय.", "base_n": 1.0, "r1_n": 2.0, "r2_n": 3.5, "r3_n": 5.0, "base_c": 5.0, "r1_c": 2.0, "r2_c": 2.5, "r3_c": 4.0, "base_a": 5.0, "r1_a": 3.0, "r2_a": 2.0, "r3_a": 4.0, "notes": ""}
]

# Write filled manifest.csv
manifest_file = Path("MANUAL_EVALUATION/manifest.csv")
with open(manifest_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Sentence", "Marathi Text",
        "Base naturalness", "R1", "R2", "R3",
        "Base clarity", "R1", "R2", "R3",
        "Base artifacts", "R1", "R2", "R3",
        "Notes"
    ])
    for row in data:
        writer.writerow([
            row["id"], row["text"],
            row["base_n"], row["r1_n"], row["r2_n"], row["r3_n"],
            row["base_c"], row["r1_c"], row["r2_c"], row["r3_c"],
            row["base_a"], row["r1_a"], row["r2_a"], row["r3_a"],
            row["notes"]
        ])

print("Updated MANUAL_EVALUATION/manifest.csv successfully.")

# Calculate statistics
models = ["Base Model (Untuned)", "Model 1 (1.2k, Attn)", "Model 2 (3.5k, Attn+MLP)", "Model 3 (6.8k, Full)"]
keys = [("base_n", "base_c", "base_a"), ("r1_n", "r1_c", "r1_a"), ("r2_n", "r2_c", "r2_a"), ("r3_n", "r3_c", "r3_a")]

summary_rows = []
for model_name, (k_n, k_c, k_a) in zip(models, keys):
    n_scores = [d[k_n] for d in data]
    c_scores = [d[k_c] for d in data]
    a_scores = [d[k_a] for d in data]
    
    mean_n = np.mean(n_scores)
    std_n = np.std(n_scores)
    mean_c = np.mean(c_scores)
    std_c = np.std(c_scores)
    mean_a = np.mean(a_scores)
    std_a = np.std(a_scores)
    overall_mos = np.mean([mean_n, mean_c, mean_a])
    
    summary_rows.append({
        "Model": model_name,
        "Naturalness MOS": round(mean_n, 2),
        "Naturalness Std": round(std_n, 2),
        "Clarity MOS": round(mean_c, 2),
        "Clarity Std": round(std_c, 2),
        "Acoustic Cleanliness MOS": round(mean_a, 2),
        "Acoustic Cleanliness Std": round(std_a, 2),
        "Overall MOS (Composite)": round(overall_mos, 2)
    })

# Write mos_summary.csv
summary_file = Path("MANUAL_EVALUATION/mos_summary.csv")
with open(summary_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("Saved MANUAL_EVALUATION/mos_summary.csv successfully.")

# Print formatted summary table
print("\n" + "="*85)
print("             HUMAN SUBJECTIVE LISTENING EVALUATION (MOS RESULTS)")
print("="*85)
print(f"{'Model':<28} | {'Naturalness':<11} | {'Clarity':<9} | {'Cleanliness':<11} | {'Overall MOS':<11}")
print("-" * 85)
for r in summary_rows:
    print(f"{r['Model']:<28} | {r['Naturalness MOS']:<5} (±{r['Naturalness Std']}) | {r['Clarity MOS']:<5} (±{r['Clarity Std']}) | {r['Acoustic Cleanliness MOS']:<5} (±{r['Acoustic Cleanliness Std']}) | {r['Overall MOS (Composite)']:<11}")
print("="*85)

# Generate 3-panel comparative visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(models))
width = 0.55

# Panel 1: Naturalness
n_means = [r["Naturalness MOS"] for r in summary_rows]
n_stds = [r["Naturalness Std"] for r in summary_rows]
bars1 = axes[0].bar(x, n_means, width, yerr=n_stds, capsize=5, color=["#94a3b8", "#60a5fa", "#34d399", "#a855f7"])
axes[0].set_title("Naturalness MOS (1-5 Scale)", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 5.5)
axes[0].set_ylabel("MOS Score")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Base", "Model 1", "Model 2", "Model 3"], rotation=15)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars1:
    h = bar.get_height()
    axes[0].annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

# Panel 2: Clarity & Intelligibility
c_means = [r["Clarity MOS"] for r in summary_rows]
c_stds = [r["Clarity Std"] for r in summary_rows]
bars2 = axes[1].bar(x, c_means, width, yerr=c_stds, capsize=5, color=["#94a3b8", "#60a5fa", "#34d399", "#a855f7"])
axes[1].set_title("Clarity & Intelligibility MOS (1-5 Scale)", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 5.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Base", "Model 1", "Model 2", "Model 3"], rotation=15)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars2:
    h = bar.get_height()
    axes[1].annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

# Panel 3: Cleanliness & Absence of Artifacts
a_means = [r["Acoustic Cleanliness MOS"] for r in summary_rows]
a_stds = [r["Acoustic Cleanliness Std"] for r in summary_rows]
bars3 = axes[2].bar(x, a_means, width, yerr=a_stds, capsize=5, color=["#94a3b8", "#60a5fa", "#34d399", "#a855f7"])
axes[2].set_title("Acoustic Cleanliness (1-5 Scale, 5=None)", fontsize=12, fontweight="bold")
axes[2].set_ylim(0, 5.5)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["Base", "Model 1", "Model 2", "Model 3"], rotation=15)
axes[2].grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars3:
    h = bar.get_height()
    axes[2].annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
out_plot = Path("figures/mos_subjective_comparison.png")
out_plot.parent.mkdir(exist_ok=True)
plt.savefig(out_plot, dpi=300)
print(f"Saved figure to {out_plot}")
