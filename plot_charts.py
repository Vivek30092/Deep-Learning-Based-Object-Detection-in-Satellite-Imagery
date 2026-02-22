"""
plot_charts.py
--------------
Generates two separate publication-quality chart images:
  fig_6_9_pie_chart.png  – Figure 6.9: Pie chart (% land cover)
  fig_6_10_bar_chart.png – Figure 6.10: Bar chart (area in sq km)

Uses the GEE classification raster (Indore_Classification.tif) which
has real non-zero class values (unlike the under-trained U-Net output).

Run from project root:
    python plot_charts.py
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# ── Config ───────────────────────────────────────────────────────────────────
RASTER_CANDIDATES = [
    "data/raw/Indore_Classification.tif",
    "data/outputs/classification_filtered.tif",
    "data/outputs/classification.tif",
]

CLASS_INFO = {
    # value : (label,         hex colour)
    0: ("Background",  "#4a4a4a"),
    1: ("Buildings",   "#e74c3c"),
    2: ("Roads",       "#2980b9"),
    3: ("Water",       "#00bcd4"),
    4: ("Vegetation",  "#27ae60"),
}
OUTPUT_DIR = Path("data/outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Find raster ───────────────────────────────────────────────────────────────
raster_path = None
for c in RASTER_CANDIDATES:
    if Path(c).exists():
        raster_path = c
        break
if raster_path is None:
    print("ERROR: No classification raster found.")
    sys.exit(1)
print(f"Reading: {raster_path}")

# ── Read raster & compute areas ───────────────────────────────────────────────
with rasterio.open(raster_path) as src:
    data      = src.read(1)
    crs       = src.crs
    transform = src.transform
    bounds    = src.bounds

    px_w = abs(transform[0])   # pixel width  in CRS units
    px_h = abs(transform[4])   # pixel height in CRS units

    if crs and crs.is_geographic:
        # CRS is in degrees → convert to metres using centre latitude
        centre_lat = (bounds.top + bounds.bottom) / 2.0
        import math
        metres_per_deg_lat = 111_320.0
        metres_per_deg_lon = 111_320.0 * math.cos(math.radians(centre_lat))
        px_area_m2 = (px_w * metres_per_deg_lon) * (px_h * metres_per_deg_lat)
        print(f"  CRS is geographic. Pixel size ≈ {px_w*metres_per_deg_lon:.1f} m × {px_h*metres_per_deg_lat:.1f} m")
    else:
        # CRS is projected (metres already)
        px_area_m2 = px_w * px_h
        print(f"  CRS is projected. Pixel size = {px_w:.2f} m × {px_h:.2f} m")

    print(f"  Pixel area = {px_area_m2:.2f} m²  ({px_area_m2/1e6:.6f} km²)")

total_pixels = data.size
classes, labels, colours, areas_sqkm, pcts = [], [], [], [], []

for val, (label, colour) in CLASS_INFO.items():
    count = int(np.sum(data == val))
    area_sqkm = count * px_area_m2 / 1_000_000
    pct = 100.0 * count / total_pixels
    print(f"  {label:<15}: {count:>10,} px  |  {area_sqkm:>10.3f} sq km  |  {pct:.2f}%")
    classes.append(val)
    labels.append(label)
    colours.append(colour)
    areas_sqkm.append(round(area_sqkm, 3))
    pcts.append(round(pct, 2))

# Filter out zero-area classes for cleaner charts
non_zero = [(l, c, a, p) for l, c, a, p in zip(labels, colours, areas_sqkm, pcts) if a > 0]
if not non_zero:
    print("\nWARNING: All classes have 0 area. Check raster values.")
    sys.exit(1)

nz_labels  = [x[0] for x in non_zero]
nz_colours = [x[1] for x in non_zero]
nz_areas   = [x[2] for x in non_zero]
nz_pcts    = [x[3] for x in non_zero]

# ── Figure 6.9 – Pie Chart ────────────────────────────────────────────────────
plt.figure(figsize=(8, 7))
wedges, texts, autotexts = plt.pie(
    nz_pcts,
    labels=None,              # legend instead of inline labels (cleaner)
    colors=nz_colours,
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
    at.set_color('white')

# Legend
legend_patches = [mpatches.Patch(color=c, label=f"{l}  ({a:.1f} km²)")
                  for l, c, a in zip(nz_labels, nz_colours, nz_areas)]
plt.legend(handles=legend_patches, loc='lower center',
           bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10,
           frameon=True, edgecolor='#cccccc')

plt.title("Land Cover Class Distribution\nIndore District, Madhya Pradesh (2023)",
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
pie_path = OUTPUT_DIR / "fig_6_9_pie_chart.png"
plt.savefig(pie_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {pie_path}")

# ── Figure 6.10 – Bar Chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x_pos = range(len(nz_labels))
bars = ax.bar(x_pos, nz_areas, color=nz_colours,
              edgecolor='white', linewidth=1.2, width=0.6)

# Value labels on top of each bar
for bar, area in zip(bars, nz_areas):
    if area > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(nz_areas) * 0.01,
                f'{area:.1f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#333333')

ax.set_xticks(list(x_pos))
ax.set_xticklabels(nz_labels, fontsize=11)
ax.set_ylabel("Area (sq. km)", fontsize=11)
ax.set_xlabel("Land Cover Class", fontsize=11)
ax.set_title("Land Cover Area per Class\nIndore District, Madhya Pradesh (2023)",
             fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, max(nz_areas) * 1.15)

plt.tight_layout()
bar_path = OUTPUT_DIR / "fig_6_10_bar_chart.png"
plt.savefig(bar_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {bar_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("DONE — insert these into your Word report:")
print(f"  Figure 6.9  → {pie_path.resolve()}")
print(f"  Figure 6.10 → {bar_path.resolve()}")
