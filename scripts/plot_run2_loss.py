"""Plot training and validation loss progression for Run 2.
Generates publication-quality curve for reports and documentation.
"""

import os
import matplotlib.pyplot as plt

train_data = [
    (10, 4.376), (20, 4.392), (30, 4.334), (40, 4.213), (50, 4.047),
    (60, 4.044), (70, 3.998), (80, 3.978), (90, 3.987), (100, 3.963),
    (110, 3.972), (120, 3.918), (130, 3.921), (140, 3.915), (150, 3.918),
    (160, 3.882), (170, 3.896), (180, 3.886), (190, 3.884), (200, 3.895),
    (210, 3.874), (220, 3.842), (230, 3.880), (240, 3.869), (250, 3.844),
    (260, 3.839), (270, 3.840), (280, 3.846), (290, 3.827), (300, 3.839),
    (310, 3.815), (320, 3.824), (330, 3.831), (340, 3.803), (350, 3.804),
    (360, 3.799), (370, 3.800), (380, 3.799), (390, 3.799), (400, 3.791),
    (410, 3.798), (420, 3.782), (430, 3.770), (440, 3.776), (450, 3.766),
    (460, 3.774), (470, 3.760), (480, 3.774), (490, 3.772), (500, 3.772),
    (510, 3.759), (520, 3.762), (530, 3.743), (540, 3.762), (550, 3.753),
    (560, 3.743), (570, 3.748), (580, 3.738), (590, 3.737), (600, 3.743),
    (610, 3.737), (620, 3.746), (630, 3.733), (640, 3.743), (650, 3.734),
    (660, 3.729), (670, 3.730), (680, 3.735), (690, 3.728), (700, 3.714),
    (710, 3.726), (720, 3.727), (730, 3.725), (740, 3.735), (750, 3.726),
    (760, 3.716), (770, 3.722), (780, 3.707), (790, 3.719), (800, 3.715),
    (810, 3.706), (820, 3.707), (830, 3.704), (840, 3.702), (850, 3.704),
    (860, 3.703), (870, 3.704), (880, 3.709), (890, 3.702), (900, 3.696),
    (910, 3.689), (920, 3.704), (930, 3.700), (940, 3.693), (950, 3.699),
    (960, 3.687), (970, 3.724), (980, 3.681), (990, 3.768), (1000, 3.718),
    (1050, 3.685)
]

eval_data = [
    (50, 4.0594), (100, 3.9431), (150, 3.8920), (200, 3.8562), (250, 3.8299),
    (300, 3.8099), (350, 3.7904), (400, 3.7770), (450, 3.7656), (500, 3.7533),
    (550, 3.7445), (600, 3.7363), (650, 3.7291), (700, 3.7242), (750, 3.7196),
    (800, 3.7149), (850, 3.7107), (900, 3.7071), (950, 3.7038), (1000, 3.7005),
    (1050, 3.6980)
]

train_steps, train_loss = zip(*train_data)
eval_steps, eval_loss = zip(*eval_data)

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(9, 4.8), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

plt.plot(train_steps, train_loss, label='Training Loss (Logged every 10 steps)', color='#0284c7', alpha=0.85, linewidth=1.8)
plt.plot(eval_steps, eval_loss, label='Validation Loss (Evaluated every 50 steps)', color='#e11d48', marker='o', markersize=4.5, linewidth=2.0, linestyle='--')

plt.title('Run 2: Marathi TTS LoRA Convergence Curve (3,500 Samples)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Optimization Steps', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.ylim(3.58, 4.45)
plt.xlim(0, 1100)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.annotate(
    f'Initial Loss: 4.392\n(Warmup @ 50s: 4.047)',
    xy=(20, 4.392), xytext=(80, 4.35),
    arrowprops=dict(facecolor='#0284c7', arrowstyle='->', lw=1.2),
    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f9ff", ec="#0284c7", lw=1)
)

plt.annotate(
    f'Best Eval Loss: 3.6980\n(Train: ~3.681)',
    xy=(1050, 3.6980), xytext=(820, 3.82),
    arrowprops=dict(facecolor='#e11d48', arrowstyle='->', lw=1.2),
    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff1f2", ec="#e11d48", lw=1)
)

plt.tight_layout()
plt.savefig('figures/run2_loss_curve.png')
print("Successfully generated figures/run2_loss_curve.png")
