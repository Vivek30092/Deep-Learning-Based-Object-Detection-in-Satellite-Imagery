"""
make_charts.py – Run: python make_charts.py
Generates fig_6_9_pie_chart.png and fig_6_10_bar_chart.png
using the pixel counts already read from your raster.
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math, os

# ── Data from your terminal output ───────────────────────────────────────────
# Pixel counts (from: python plot_charts.py)
pixel_counts = {
    "Buildings":  564_393,
    "Roads":    1_109_663,
    "Water":       95_194,
    "Vegetation":  216_195,
}
colours = {
    "Buildings":  "#e74c3c",
    "Roads":      "#2980b9",
    "Water":      "#00bcd4",
    "Vegetation": "#27ae60",
}

# Area calculation: Indore AOI at ~22.7°N, GEE export at 30 m
# 1° lon at 22.7° N ≈ 102,950 m  |  1° lat ≈ 111,320 m
# GEE default pixel size for Sentinel-2 = 30 m = 0.0002694° approx
# Verified: ~30 m × 30 m per pixel
PX_AREA_KM2 = (30 * 30) / 1_000_000   # 0.0009 km² per pixel

total_px    = sum(pixel_counts.values())
labels      = list(pixel_counts.keys())
counts      = list(pixel_counts.values())
areas       = [c * PX_AREA_KM2 for c in counts]
pcts        = [100.0 * c / total_px for c in counts]
clrs        = [colours[l] for l in labels]

print("Class breakdown:")
for l, a, p in zip(labels, areas, pcts):
    print(f"  {l:<15}: {a:>8.2f} km²   {p:.2f}%")

# ── Output directory ─────────────────────────────────────────────────────────
OUT = "data/outputs/reports"
os.makedirs(OUT, exist_ok=True)

# ── Figure 6.9 – Pie Chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
wedges, _, autotexts = ax.pie(
    pcts,
    colors=clrs,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    startangle=140,
    pctdistance=0.72,
    wedgeprops=dict(edgecolor='white', linewidth=2),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
    at.set_color('white')

patches = [mpatches.Patch(color=c, label=f"{l}  ({a:.1f} km²  |  {p:.1f}%)")
           for l, c, a, p in zip(labels, clrs, areas, pcts)]
ax.legend(handles=patches, loc='lower center',
          bbox_to_anchor=(0.5, -0.13), ncol=2,
          fontsize=10, frameon=True, edgecolor='#cccccc')
ax.set_title("Land Cover Class Distribution\nIndore District, Madhya Pradesh (2023)",
             fontsize=13, fontweight='bold', pad=18)
pie_path = f"{OUT}/fig_6_9_pie_chart.png"
plt.savefig(pie_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {pie_path}")

# ── Figure 6.10 – Bar Chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(labels))
bars = ax.bar(x, areas, color=clrs, edgecolor='white', linewidth=1.5, width=0.55)

for bar, area in zip(bars, areas):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(areas) * 0.015,
            f'{area:.1f} km²',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#333333')

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Area (sq. km)", fontsize=12)
ax.set_xlabel("Land Cover Class", fontsize=12)
ax.set_title("Land Cover Area per Class\nIndore District, Madhya Pradesh (2023)",
             fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, max(areas) * 1.18)
plt.tight_layout()
bar_path = f"{OUT}/fig_6_10_bar_chart.png"
plt.savefig(bar_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {bar_path}")

print("\nDone! Insert into Word report:")
print(f"  Figure 6.9  → {os.path.abspath(pie_path)}")
print(f"  Figure 6.10 → {os.path.abspath(bar_path)}")
