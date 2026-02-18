"""
Patch Extraction for Training
Extract fixed-size patches from large satellite imagery
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import yaml


def extract_patches(image_path: str, mask_path: Optional[str] = None,
                   patch_size: int = 256, overlap: int = 32,
                   output_dir: str = 'data/training',
                   min_valid_pixels: float = 0.5) -> Tuple[int, int]:
    """
    Extract patches from large GeoTIFF imagery.
    
    Args:
        image_path: Path to input image
        mask_path: Path to mask/label image (optional)
        patch_size: Size of patches (square)
        overlap: Overlap between patches
        output_dir: Output directory
        min_valid_pixels: Minimum fraction of valid (non-zero) pixels
        
    Returns:
        (num_patches_extracted, num_patches_skipped)
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    
    image_dir.mkdir(parents=True, exist_ok=True)
    if mask_path:
        mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        num_bands = src.count
        
        print(f"Image size: {width}x{height}, Bands: {num_bands}")
        
        stride = patch_size - overlap
        num_patches_h = (height - overlap) // stride
        num_patches_w = (width - overlap) // stride
        total_patches = num_patches_h * num_patches_w
        
        print(f"Extracting {total_patches} patches ({num_patches_h}x{num_patches_w})")
        
        # Open mask if provided
        mask_src = None
        if mask_path:
            mask_src = rasterio.open(mask_path)
        
        extracted = 0
        skipped = 0
        
        with tqdm(total=total_patches, desc="Extracting patches") as pbar:
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    # Calculate patch coordinates
                    row_start = i * stride
                    col_start = j * stride
                    
                    # Create window
                    window = Window(col_start, row_start, patch_size, patch_size)
                    
                    # Read patch
                    patch = src.read(window=window)  # (C, H, W)
                    patch = np.transpose(patch, (1, 2, 0))  # (H, W, C)
                    
                    # Check if patch is valid
                    valid_pixels = np.sum(patch > 0) / patch.size
                    
                    if valid_pixels < min_valid_pixels:
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Save image patch
                    patch_name = f"patch_{i:04d}_{j:04d}.npy"
                    np.save(image_dir / patch_name, patch)
                    
                    # Read and save mask patch if available
                    if mask_src:
                        mask_patch = mask_src.read(window=window)
                        mask_patch = np.transpose(mask_patch, (1, 2, 0))
                        
                        # Handle single-band masks
                        if mask_patch.shape[-1] == 1:
                            mask_patch = mask_patch.squeeze(-1)
                        
                        np.save(mask_dir / patch_name, mask_patch)
                    
                    extracted += 1
                    pbar.update(1)
        
        if mask_src:
            mask_src.close()
    
    print(f"\nExtraction complete!")
    print(f"Extracted: {extracted} patches")
    print(f"Skipped: {skipped} patches (insufficient valid pixels)")
    
    return extracted, skipped


def create_train_val_split(data_dir: str, val_split: float = 0.2, 
                          seed: int = 42):
    """
    Split patches into training and validation sets.
    
    Args:
        data_dir: Directory containing 'images' and 'masks' folders
        val_split: Fraction of data for validation
        seed: Random seed
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    # Get all patch files
    patch_files = sorted(list(image_dir.glob('*.npy')))
    num_patches = len(patch_files)
    
    print(f"Total patches: {num_patches}")
    
    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(num_patches)
    
    val_size = int(num_patches * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    print(f"Training: {len(train_indices)}, Validation: {len(val_indices)}")
    
    # Create validation directories
    val_dir = data_dir.parent / 'validation'
    val_image_dir = val_dir / 'images'
    val_mask_dir = val_dir / 'masks'
    
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Move validation patches
    for idx in tqdm(val_indices, desc="Moving validation patches"):
        patch_name = patch_files[idx].name
        
        # Move image
        src = image_dir / patch_name
        dst = val_image_dir / patch_name
        src.rename(dst)
        
        # Move mask
        if mask_dir.exists():
            src = mask_dir / patch_name
            dst = val_mask_dir / patch_name
            if src.exists():
                src.rename(dst)
    
    print("Train/validation split complete!")


def load_patches(data_dir: str, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all patches from directory.
    
    Args:
        data_dir: Directory with 'images' and 'masks'
        batch_size: Batch size for loading
        
    Returns:
        (images, masks) as NumPy arrays
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    image_files = sorted(list(image_dir.glob('*.npy')))
    
    images = []
    masks = []
    
    for img_file in tqdm(image_files, desc="Loading patches"):
        image = np.load(img_file)
        images.append(image)
        
        mask_file = mask_dir / img_file.name
        if mask_file.exists():
            mask = np.load(mask_file)
            masks.append(mask)
    
    images = np.array(images)
    
    if masks:
        masks = np.array(masks)
        return images, masks
    
    return images, None


if __name__ == "__main__":
    print("Patch Extraction Module")
    print("Usage:")
    print("  extract_patches(image_path, mask_path, patch_size=256)")
    print("  create_train_val_split(data_dir, val_split=0.2)")
