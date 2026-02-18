"""
Quick Training Data Preparation Script
Extract patches from GEE-downloaded imagery
"""

import sys
from pathlib import Path
import argparse
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing.image_utils import read_geotiff, normalize_image
from preprocessing.patch_extraction import extract_patches, create_train_val_split


def prepare_training_data():
    """
    Quick script to prepare training data from GEE exports.
    """
    print("=" * 60)
    print("Training Data Preparation")
    print("=" * 60)
    
    # Load configuration
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define paths
    raw_dir = Path('data/raw')
    training_dir = Path('data/training')
    
    # Check if imagery exists
    image_files = list(raw_dir.glob('*.tif'))
    
    if not image_files:
        print("\n❌ No GeoTIFF files found in data/raw/")
        print("Please run Google Earth Engine scripts first and download imagery.")
        print("See gee_scripts/README.md for instructions.")
        return
    
    print(f"\nFound {len(image_files)} image(s) in data/raw/:")
    for img in image_files:
        print(f"  - {img.name}")
    
    # Select image and mask
    print("\n" + "=" * 60)
    print("Available files:")
    for idx, img in enumerate(image_files, 1):
        print(f"  {idx}. {img.name}")
    
    try:
        choice = int(input("\nSelect image file number: ")) - 1
        image_path = str(image_files[choice])
    except (ValueError, IndexError):
        print("Invalid selection. Using first file.")
        image_path = str(image_files[0])
    
    print(f"\nSelected: {Path(image_path).name}")
    
    # Ask for mask
    mask_files = [f for f in image_files if 'mask' in f.name.lower() or 'classification' in f.name.lower()]
    
    if mask_files:
        print("\nFound potential mask files:")
        for idx, msk in enumerate(mask_files, 1):
            print(f"  {idx}. {msk.name}")
        
        try:
            mask_choice = input("\nSelect mask file number (or press Enter to skip): ")
            if mask_choice:
                mask_path = str(mask_files[int(mask_choice) - 1])
            else:
                mask_path = None
        except (ValueError, IndexError):
            mask_path = None
    else:
        print("\n⚠ No mask files found.")
        print("You'll need to create masks from GEE classification results.")
        mask_path = None
    
    # Extract patches
    print("\n" + "=" * 60)
    print("Extracting Patches")
    print("=" * 60)
    
    patch_size = config['image']['patch_size']
    overlap = config['image']['overlap']
    
    print(f"\nPatch size: {patch_size}x{patch_size}")
    print(f"Overlap: {overlap} pixels")
    
    extracted, skipped = extract_patches(
        image_path=image_path,
        mask_path=mask_path,
        patch_size=patch_size,
        overlap=overlap,
        output_dir=str(training_dir),
        min_valid_pixels=0.5
    )
    
    if extracted > 0:
        # Create train/val split
        print("\n" + "=" * 60)
        print("Creating Train/Validation Split")
        print("=" * 60)
        
        val_split = config['training']['validation_split']
        
        create_train_val_split(
            data_dir=str(training_dir),
            val_split=val_split,
            seed=42
        )
        
        print("\n✓ Training data preparation complete!")
        print(f"\nNext step: Train the model using:")
        print(f"  python -m models.train")
    else:
        print("\n❌ No patches extracted. Check your imagery.")


if __name__ == "__main__":
    prepare_training_data()
