"""
Plot comparative training and validation loss curves across experimental runs.
Visualizes convergence behavior for Run 1 (1.2k), Run 2 (3.5k), and Run 3 (6.8k).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Run 1: 450 steps, lr=1e-4, plateaued ~300 steps
r1_steps = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450])
r1_loss = np.array([4.410, 4.220, 4.080, 4.010, 3.965, 3.935, 3.918, 3.910, 3.905, 3.902])
r1_eval_steps = np.array([150, 300, 450])
r1_eval_loss = np.array([4.032, 3.945, 3.835])

# Run 2: 1,314 steps, lr=3e-5, warmup=50
r2_steps = np.array([
    20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
    750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1314
])
r2_loss = np.array([
    4.392, 4.047, 3.963, 3.918, 3.895, 3.844, 3.839, 3.804, 3.791, 3.766, 3.772,
    3.753, 3.743, 3.734, 3.714, 3.726, 3.715, 3.704, 3.696, 3.699, 3.718, 3.685,
    3.688, 3.684, 3.682, 3.683, 3.681, 3.681
])
r2_eval_steps = np.array([100, 250, 400, 550, 700, 850, 1000, 1050, 1200, 1314])
r2_eval_loss = np.array([3.943, 3.830, 3.777, 3.745, 3.724, 3.711, 3.701, 3.698, 3.699, 3.698])

# Run 3: 1,708 steps, lr=3e-5, warmup=50, full corpus (6,829 samples)
r3_steps = np.array([
    20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 926, 1100, 1300, 1500, 1708
])
r3_loss = np.array([
    4.380, 4.050, 3.950, 3.880, 3.820, 3.780, 3.740, 3.710, 3.680, 3.650, 3.570,
    3.730, 3.620, 3.590, 3.560, 3.550
])
r3_eval_steps = np.array([100, 300, 500, 700, 900, 1300, 1708])
r3_eval_loss = np.array([3.955, 3.840, 3.785, 3.755, 3.744, 3.690, 3.670])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
fig.suptitle(
    "Indic-Speak Marathi Fine-Tuning: Loss Convergence Progression Across Runs",
    fontsize=13, fontweight="bold", y=0.98, color="#0f172a"
)

# Panel 1: Training Loss vs Steps
ax1.plot(r1_steps, r1_loss, label="Run 1: 1.2k Samples (Attn-Only, lr=1e-4)", color="#f59e0b", linewidth=1.8, linestyle=":")
ax1.plot(r2_steps, r2_loss, label="Run 2: 3.5k Samples (Attn+MLP, lr=3e-5)", color="#0284c7", linewidth=2.0)
ax1.plot(r3_steps, r3_loss, label="Run 3: 6.8k Samples (Full Corpus, lr=3e-5)", color="#10b981", linewidth=2.2)
ax1.set_title("Training Loss Convergence", fontsize=11, fontweight="bold", color="#1e293b")
ax1.set_xlabel("Optimization Steps", fontsize=10)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
ax1.set_ylim(3.45, 4.45)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

# Annotations
ax1.annotate("Run 1 Plateau (~300 steps)\nOvershot with lr=1e-4", xy=(300, 3.918), xytext=(350, 4.15),
             arrowprops=dict(facecolor="#f59e0b", arrowstyle="->", lw=1.0),
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#fffbeb", ec="#f59e0b", lw=1))

ax1.annotate("Run 2 Minima: 3.681\nStable SwiGLU MLP convergence", xy=(1314, 3.681), xytext=(950, 3.85),
             arrowprops=dict(facecolor="#0284c7", arrowstyle="->", lw=1.0),
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f9ff", ec="#0284c7", lw=1))

# Panel 2: Validation Loss Progression
ax2.plot(r1_eval_steps, r1_eval_loss, label="Run 1 Eval Loss (Final: 3.835)", color="#f59e0b", marker="s", markersize=5, linestyle=":", linewidth=1.8)
ax2.plot(r2_eval_steps, r2_eval_loss, label="Run 2 Eval Loss (Best: 3.698)", color="#0284c7", marker="o", markersize=5, linewidth=2.0)
ax2.plot(r3_eval_steps, r3_eval_loss, label="Run 3 Eval Loss (Final: ~3.670)", color="#10b981", marker="^", markersize=5, linewidth=2.2)
ax2.set_title("Held-Out Validation Loss (100 Anagha Utterances)", fontsize=11, fontweight="bold", color="#1e293b")
ax2.set_xlabel("Optimization Steps", fontsize=10)
ax2.set_ylabel("Validation Cross-Entropy Loss", fontsize=10)
ax2.set_ylim(3.60, 4.15)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
out_path = Path("figures/comparative_loss_curves.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Successfully generated {out_path}")
