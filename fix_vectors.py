"""
fix_vectors.py
--------------
STEP 1: Diagnoses your classification_filtered.tif
STEP 2: Extracts valid non-empty vector layers for each class
STEP 3: Saves as both .shp AND .gpkg (GeoPackage, more reliable)

Run from project root:
    python fix_vectors.py
"""

import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
# Try classification_filtered first, fall back to classification
RASTER_CANDIDATES = [
    "data/outputs/classification_filtered.tif",
    "data/outputs/classification.tif",
    "data/raw/Indore_Classification.tif",   # GEE export fallback
]
OUTPUT_DIR = Path("data/outputs/vectors")

CLASS_MAP = {
    1: "buildings",
    2: "roads",
    3: "water",
    4: "vegetation",
}
COLOURS = {
    "buildings":  "#e74c3c",
    "roads":      "#2980b9",
    "water":      "#00bcd4",
    "vegetation": "#27ae60",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Find the raster
# ─────────────────────────────────────────────────────────────────────────────
raster_path = None
for c in RASTER_CANDIDATES:
    if Path(c).exists():
        raster_path = c
        break

if raster_path is None:
    print("ERROR: No classification raster found. Checked:")
    for c in RASTER_CANDIDATES:
        print(f"  {c}")
    sys.exit(1)

print(f"Using raster: {raster_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Diagnose the raster
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

with rasterio.open(raster_path) as src:
    data     = src.read(1)
    crs      = src.crs
    transform = src.transform
    print(f"  Shape      : {data.shape}")
    print(f"  Dtype      : {data.dtype}")
    print(f"  CRS        : {crs}")
    print(f"  No-data val: {src.nodata}")
    unique, counts = np.unique(data, return_counts=True)
    print(f"\n  Pixel value distribution:")
    for u, c in zip(unique, counts):
        pct = 100.0 * c / data.size
        label = CLASS_MAP.get(int(u), "background" if u == 0 else "unknown")
        print(f"    Value {u:2d} ({label:12s}): {c:8,d} pixels  ({pct:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check if any target class pixels exist
# ─────────────────────────────────────────────────────────────────────────────
class_pixels = {v: int(np.sum(data == v)) for v in CLASS_MAP}
has_any = any(n > 0 for n in class_pixels.values())

if not has_any:
    print("\n[!] WARNING: The raster contains NO pixels for classes 1-4.")
    print("    This means the model classified everything as background (0).")
    print("    Likely cause: model was not fully trained (only 10 epochs).")
    print("\n    SOLUTION OPTIONS:")
    print("    A) Train for more epochs (50-100) and re-run inference.")
    print("    B) Use the GEE classification (Indore_Classification.tif)")
    print("       as a temporary substitute for report figures.")
    print("\n    Trying Indore_Classification.tif as fallback...")
    fallback = "data/raw/Indore_Classification.tif"
    if Path(fallback).exists():
        raster_path = fallback
        with rasterio.open(raster_path) as src:
            data      = src.read(1)
            crs       = src.crs
            transform = src.transform
        unique, counts = np.unique(data, return_counts=True)
        print(f"\n  Fallback raster values:")
        for u, c in zip(unique, counts):
            pct = 100.0 * c / data.size
            print(f"    Value {u}: {c:,d} pixels ({pct:.2f}%)")
        class_pixels = {v: int(np.sum(data == v)) for v in CLASS_MAP}
    else:
        print("    Fallback not found either. Cannot proceed.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Extract vectors for each class
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXTRACTING VECTORS")
print("="*60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_AREA_PX = 50   # keep polygons with area > 50 pixels (avoid noise)

summary = []

with rasterio.open(raster_path) as src:
    data      = src.read(1)
    crs       = src.crs
    transform = src.transform

    for class_id, class_name in CLASS_MAP.items():
        n_px = class_pixels.get(class_id, 0)
        print(f"\nClass {class_id} – {class_name}: {n_px:,} pixels")

        if n_px == 0:
            print(f"  SKIPPED – no pixels found for this class.")
            summary.append((class_name, 0, "SKIPPED – no pixels in raster"))
            continue

        # Create binary mask for this class
        mask = (data == class_id).astype(np.uint8)

        # Polygonize
        shapes_gen = rfeatures.shapes(mask, mask=mask, transform=transform)

        geometries = []
        for geom_dict, val in shapes_gen:
            if val == 1:
                geom = shape(geom_dict)
                if geom.area >= MIN_AREA_PX:
                    # Simplify slightly for cleaner look
                    geom = geom.simplify(1.0, preserve_topology=True)
                    geometries.append(geom)

        if not geometries:
            print(f"  WARNING – pixels exist but all polygons are tiny (<{MIN_AREA_PX} px area).")
            print(f"           Retrying with no area filter...")
            shapes_gen2 = rfeatures.shapes(mask, mask=mask, transform=transform)
            geometries  = [shape(g) for g, v in shapes_gen2 if v == 1]

        n_features = len(geometries)
        print(f"  Extracted {n_features} polygons")

        if n_features == 0:
            summary.append((class_name, 0, "FAILED – extraction produced 0 polygons"))
            continue

        # Build GeoDataFrame
        areas   = [g.area for g in geometries]
        gdf = gpd.GeoDataFrame(
            {
                "class_id":  [class_id]   * n_features,
                "class_name":[class_name] * n_features,
                "area_sqm":  areas,
                "area_sqkm": [a / 1_000_000 for a in areas],
            },
            geometry=geometries,
            crs=crs,
        )

        # Save as GeoPackage (recommended – no field-name truncation)
        gpkg_path = OUTPUT_DIR / f"{class_name}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"  Saved GeoPackage: {gpkg_path}")

        # Also try to save as Shapefile — may fail if QGIS has the file open
        shp_path = OUTPUT_DIR / f"{class_name}.shp"
        try:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            print(f"  Saved Shapefile : {shp_path}")
        except PermissionError:
            print(f"  SKIPPED .shp   : {shp_path} is open in QGIS — close that layer and re-run if you need the .shp")
            print(f"  The .gpkg file is already saved and works perfectly in QGIS.")

        total_area_sqkm = sum(areas) / 1_000_000
        summary.append((class_name, n_features, f"{total_area_sqkm:.4f} sq km"))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Class':<15} {'Features':>10}  {'Total Area':>15}")
print("-"*45)
for class_name, n, area_info in summary:
    print(f"{class_name:<15} {str(n):>10}  {area_info:>15}")

print(f"\nVector files saved to: {OUTPUT_DIR.resolve()}")
print("\nNow you can load these in QGIS:")
print("  Layer -> Add Layer -> Add Vector Layer")
print("  Browse to:", OUTPUT_DIR.resolve())
print("  Select any .gpkg file -> Add")
