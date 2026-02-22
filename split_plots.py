# Save as split_plots.py in your project folder
# Run: python split_plots.py

import matplotlib.pyplot as plt
import glob, os

# ── Auto-find the latest log folder ──────────────────────────
log_folders = sorted(glob.glob('logs/*'))
latest = log_folders[-1] if log_folders else None

# ── If training history exists, re-plot from the saved data ──
# We re-run training_history splitting into 2 separate PNGs.
# Paste your actual values from the terminal output below:

epochs     = list(range(1, 11))   # 10 epochs
train_loss = [0.85,0.72,0.61,0.52,0.45,0.39,0.34,0.30,0.27,0.25]
val_loss   = [0.88,0.75,0.65,0.56,0.49,0.43,0.39,0.36,0.33,0.31]
train_iou  = [0.30,0.43,0.53,0.61,0.67,0.72,0.76,0.79,0.81,0.83]
val_iou    = [0.27,0.40,0.50,0.58,0.64,0.69,0.73,0.76,0.78,0.80]

# Figure 5.6 — Loss Curve
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=5, label='Training Loss')
plt.plot(epochs, val_loss,   'r-o', linewidth=2, markersize=5, label='Validation Loss')
plt.title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_6_loss_curve.png', dpi=150, bbox_inches='tight')
print("Saved: fig_5_6_loss_curve.png")

# Figure 5.7 — IoU Curve
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_iou, 'b-o', linewidth=2, markersize=5, label='Training IoU')
plt.plot(epochs, val_iou,   'r-o', linewidth=2, markersize=5, label='Validation IoU')
plt.title('Training and Validation IoU Score', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('IoU Score', fontsize=12)
plt.xticks(epochs)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_7_iou_curve.png', dpi=150, bbox_inches='tight')
print("Saved: fig_5_7_iou_curve.png")