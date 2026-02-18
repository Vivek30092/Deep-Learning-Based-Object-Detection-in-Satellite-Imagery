"""
Example Script: Quick Start Guide
Run this to verify setup and test basic functionality
"""

import sys
from pathlib import Path

print("=" * 60)
print("Project P7: Deep Learning Object Detection - Setup Verification")
print("=" * 60)

# Test imports
print("\n1. Testing Python Dependencies...")

try:
    import numpy as np
    print("   ✓ NumPy:", np.__version__)
except ImportError as e:
    print("   ✗ NumPy not found:", e)

try:
    import tensorflow as tf
    print("   ✓ TensorFlow:", tf.__version__)
except ImportError as e:
    print("   ✗ TensorFlow not found:", e)

try:
    import rasterio
    print("   ✓ Rasterio:", rasterio.__version__)
except ImportError as e:
    print("   ✗ Rasterio not found:", e)

try:
    import geopandas as gpd
    print("   ✓ GeoPandas:", gpd.__version__)
except ImportError as e:
    print("   ✗ GeoPandas not found:", e)

try:
    import cv2
    print("   ✓ OpenCV:", cv2.__version__)
except ImportError as e:
    print("   ✗ OpenCV not found:", e)

try:
    import yaml
    print("   ✓ PyYAML installed")
except ImportError as e:
    print("   ✗ PyYAML not found:", e)

try:
    import ee
    print("   ✓ Earth Engine API installed")
except ImportError as e:
    print("   ✗ Earth Engine API not found:", e)

try:
    import segmentation_models as sm
    print("   ✓ Segmentation Models:", sm.__version__)
except ImportError as e:
    print("   ✗ Segmentation Models not found:", e)

# Verify project structure
print("\n2. Verifying Project Structure...")

project_root = Path(__file__).parent
required_dirs = [
    'data/raw',
    'data/aoi',
    'data/training/images',
    'data/training/masks',
    'data/validation',
    'data/outputs',
    'models/saved_models',
    'gee_scripts',
    'preprocessing',
    'postprocessing',
    'config'
]

all_exist = True
for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} - MISSING")
        all_exist = False

# Check configuration file
print("\n3. Checking Configuration...")

config_path = project_root / 'config' / 'config.yaml'
if config_path.exists():
    print(f"   ✓ Configuration file found")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Configuration loaded successfully")
        print(f"   - Study area: {config['project']['study_area']}")
        print(f"   - Number of classes: {config['model']['n_classes']}")
        print(f"   - Patch size: {config['image']['patch_size']}")
    except Exception as e:
        print(f"   ✗ Error loading configuration: {e}")
else:
    print(f"   ✗ Configuration file not found")

# Check GEE scripts
print("\n4. Checking GEE Scripts...")

gee_scripts = [
    'gee_scripts/01_data_acquisition.js',
    'gee_scripts/02_feature_engineering.js',
    'gee_scripts/03_ml_classification.js'
]

for script in gee_scripts:
    script_path = project_root / script
    if script_path.exists():
        print(f"   ✓ {script}")
    else:
        print(f"   ✗ {script} - MISSING")

# Summary
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)

if all_exist:
    print("\n✓ All required directories exist")
else:
    print("\n⚠ Some directories are missing - they will be created automatically when needed")

print("\n📋 NEXT STEPS:")
print("1. Sign up for Google Earth Engine: https://earthengine.google.com/")
print("2. Run GEE scripts to download satellite imagery")
print("3. Digitize training samples in GEE")
print("4. Extract training patches using: python -m preprocessing.patch_extraction")
print("5. Train model using: python -m models.train")
print("\n📖 See README.md for detailed workflow")
print("=" * 60)
