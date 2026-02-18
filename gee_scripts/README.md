# Google Earth Engine Workflow Guide

## Overview

This guide explains how to acquire and preprocess satellite imagery for Indore District using Google Earth Engine (GEE).

---

## Prerequisites

1. **GEE Account**: Sign up at https://earthengine.google.com/signup/
2. **GEE Code Editor**: Access at https://code.earthengine.google.com/

---

## Script Execution Order

### Script 1: Data Acquisition

**File**: `01_data_acquisition.js`

**Purpose**: Download Sentinel-2 imagery for Indore District

**Steps**:
1. Open the script in GEE Code Editor
2. The AOI is pre-defined for Indore District (can be modified)
3. Run the script
4. Check the **Tasks** tab (top-right orange button)
5. Click **RUN** on the export task
6. Monitor progress (takes 5-30 minutes depending on area size)
7. Download from Google Drive → GEE_Exports folder
8. Save to `data/raw/` in your project

**Output**: `Indore_Sentinel2_Composite.tif` (4 bands: Blue, Green, Red, NIR)

---

### Script 2: Feature Engineering

**File**: `02_feature_engineering.js`

**Purpose**: Calculate spectral indices and texture features

**Features Calculated**:
- NDVI (vegetation index)
- NDWI (water index)
- NDBI (built-up index)
- Texture contrast
- Texture entropy

**Steps**:
1. Open the script in GEE Code Editor
2. Run the script
3. Visualize the indices on the map
4. Export task will appear in **Tasks** tab
5. Run the export
6. Download from Google Drive
7. Save to `data/raw/`

**Output**: `Indore_Enhanced_Features.tif` (9 bands)

---

### Script 3: AI-Assisted Classification

**File**: `03_ml_classification.js`

**Purpose**: Create initial training data using Random Forest classifier

**Steps**:

#### Part A: Digitize Training Samples

1. Open the script in GEE Code Editor
2. Use the **geometry tools** (top-left toolbar):
   - Select polygon tool
   - Digitize 5-10 samples for each class:
     - **Buildings**: Red polygons over building clusters
     - **Roads**: Blue polylines/polygons over roads
     - **Water**: Cyan polygons over water bodies
     - **Vegetation**: Green polygons over green areas

3. For each geometry:
   - Click **+ new layer**
   - Draw samples
   - Configure settings (click gear icon):
     - Name: `buildings`, `roads`, `water`, `vegetation`
     - Import as: `FeatureCollection`
     - Add property: `class` with value (1, 2, 3, 4)

#### Part B: Run Classification

1. Uncomment the code section (lines marked with `/*` and `*/`)
2. Update geometry names to match your imports
3. Run the script
4. View classification on the map
5. Export results from **Tasks** tab

**Outputs**:
- `Indore_Classification.tif` - Full classification
- `Indore_Buildings_Mask.tif` - Building mask only

---

## Tips for Digitizing Training Samples

### Quality Guidelines

- **Coverage**: Digitize samples across the entire study area
- **Variability**: Include different types (e.g., large buildings, small buildings)
- **Purity**: Ensure polygons contain only one class
- **Quantity**: 50-100 total samples minimum (10-20 per class)

### Visual Interpretation

**Buildings**:
- Look for: Rectangular shapes, shadows, regular patterns
- Avoid: Confusion with parking lots, bare ground

**Roads**:
- Look for: Linear features, networks
- Avoid: Railways, bare pathways

**Water**:
- Look for: Dark blue/black areas, smooth texture
- Avoid: Shadows

**Vegetation**:
- Look for: Green/red (in false color), patches or continuous cover
- Avoid: Agricultural fields during non-growing season

---

## Common Issues

### Issue 1: Export Failed - Too Many Pixels

**Solution**:
- Reduce the AOI area
- Increase `scale` parameter (e.g., from 10 to 30)
- Process in smaller tiles

### Issue 2: No Images Found

**Solution**:
- Adjust date range (try 2022-2023)
- Increase cloud threshold (try 20 or 30)
- Check if AOI is correct

### Issue 3: Computation Timeout

**Solution**:
- Use `.median()` instead of `.mosaic()`
- Reduce the number of bands in export
- Simplify processing steps

---

## Google Earth Engine Resources

- **Documentation**: https://developers.google.com/earth-engine/
- **Tutorials**: https://developers.google.com/earth-engine/tutorials
- **Dataset Catalog**: https://developers.google.com/earth-engine/datasets/
- **Community Forum**: https://groups.google.com/g/google-earth-engine-developers

---

## Recommended Workflow

1. **Start Small**: Test on small area first (e.g., city center only)
2. **Visual Validation**: Always visualize before exporting
3. **Incremental Processing**: Run one script at a time
4. **Check Outputs**: Verify each downloaded file in QGIS before proceeding

---

## Advanced Options

### Custom AOI from Shapefile

1. Upload shapefile to GEE:
   - **Assets** tab → **New** → **Shape files**
   - Upload `.shp`, `.shx`, `.dbf`, `.prj` files
   - Wait for ingestion

2. Update script:
```javascript
var aoi = ee.FeatureCollection("users/YOUR_USERNAME/indore_boundary");
var aoi = aoi.geometry();
```

### Time Series Analysis

Modify date filter to specific seasons:
```javascript
.filterDate('2023-05-01', '2023-07-31')  // Summer
.filterDate('2023-11-01', '2024-01-31')  // Winter
```

### Cloud Masking

Add cloud masking for better composites:
```javascript
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask);
}

var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(indore)
  .filterDate('2023-01-01', '2023-12-31')
  .map(maskS2clouds);
```

---

**Next Steps**: After downloading imagery, proceed to `preprocessing/patch_extraction.py` to create training data.
