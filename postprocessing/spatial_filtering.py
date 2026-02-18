"""
Spatial Filtering and Post-processing
Refine segmentation outputs using morphological operations
"""

import numpy as np
import cv2
from scipy import ndimage
import rasterio
from pathlib import Path
import yaml


def morphological_operations(image: np.ndarray, operation: str = 'closing',
                             kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological operations to binary mask.
    
    Args:
        image: Binary mask
        operation: 'opening', 'closing', 'both'
        kernel_size: Size of structuring element
        
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'opening':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'both':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    else:
        result = image
    
    return result


def remove_small_objects(image: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small connected components.
    
    Args:
        image: Binary mask
        min_size: Minimum size in pixels
        
    Returns:
        Filtered mask
    """
    # Label connected components
    labeled, num_features = ndimage.label(image)
    
    # Calculate sizes
    sizes = ndimage.sum(image, labeled, range(num_features + 1))
    
    # Create mask of large components
    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0  # Background
    
    # Filter
    filtered = mask_sizes[labeled]
    
    return filtered.astype(np.uint8)


def fill_holes(image: np.ndarray, max_hole_size: int = 50) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        image: Binary mask
        max_hole_size: Maximum hole size to fill
        
    Returns:
        Filled mask
    """
    # Invert image to find holes
    inverted = 1 - image
    
    # Label holes
    labeled, num_features = ndimage.label(inverted)
    
    # Calculate sizes
    sizes = ndimage.sum(inverted, labeled, range(num_features + 1))
    
    # Create mask of small holes
    mask_sizes = sizes <= max_hole_size
    mask_sizes[0] = 0  # Background
    
    # Fill small holes
    holes_to_fill = mask_sizes[labeled]
    
    result = np.logical_or(image, holes_to_fill).astype(np.uint8)
    
    return result


def smooth_boundaries(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth object boundaries using Gaussian filter.
    
    Args:
        image: Binary mask
        sigma: Gaussian kernel sigma
        
    Returns:
        Smoothed mask
    """
    # Apply Gaussian filter
    smoothed = ndimage.gaussian_filter(image.astype(float), sigma=sigma)
    
    # Threshold back to binary
    result = (smoothed > 0.5).astype(np.uint8)
    
    return result


def postprocess_mask(mask_path: str, output_path: str,
                    config_path: str = 'config/config.yaml'):
    """
    Apply post-processing to segmentation mask.
    
    Args:
        mask_path: Path to input mask
        output_path: Path to output mask
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    postproc_config = config.get('postprocessing', {})
    
    # Read mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        metadata = src.meta.copy()
    
    print(f"Processing mask: {mask_path}")
    print(f"Original unique values: {np.unique(mask)}")
    
    # Process binary mask (assuming single class)
    if postproc_config.get('morphology', {}).get('enabled', True):
        operation = postproc_config['morphology']['operation']
        kernel_size = postproc_config['morphology']['kernel_size']
        
        print(f"Applying morphological operation: {operation}")
        mask = morphological_operations(mask, operation, kernel_size)
    
    if postproc_config.get('fill_holes', True):
        print("Filling holes...")
        mask = fill_holes(mask, max_hole_size=50)
    
    if postproc_config.get('min_area', 0) > 0:
        min_area = postproc_config['min_area']
        print(f"Removing small objects (< {min_area} pixels)...")
        mask = remove_small_objects(mask, min_size=min_area)
    
    if postproc_config.get('smooth_boundaries', True):
        print("Smoothing boundaries...")
        mask = smooth_boundaries(mask, sigma=1.0)
    
    # Save result
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(mask, 1)
    
    print(f"Saved processed mask to: {output_path}")
    print(f"Final unique values: {np.unique(mask)}")


def postprocess_all_classes(input_dir: str, output_dir: str,
                           config_path: str = 'config/config.yaml'):
    """
    Post-process all class masks.
    
    Args:
        input_dir: Input directory with masks
        output_dir: Output directory
        config_path: Path to configuration file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all mask files
    for mask_file in input_dir.glob('mask_*.tif'):
        output_file = output_dir / mask_file.name
        postprocess_mask(str(mask_file), str(output_file), config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Post-process segmentation masks')
    parser.add_argument('--input', required=True, help='Input directory or file')
    parser.add_argument('--output', required=True, help='Output directory or file')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        postprocess_all_classes(args.input, args.output, args.config)
    else:
        postprocess_mask(args.input, args.output, args.config)
