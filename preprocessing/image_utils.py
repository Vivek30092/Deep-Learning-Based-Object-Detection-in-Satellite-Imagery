"""
Image Processing Utilities
Deep Learning-Based Object Detection Project
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


def read_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Read GeoTIFF file and return image array with metadata.
    
    Args:
        file_path: Path to GeoTIFF file
        
    Returns:
        image: NumPy array (H, W, C)
        metadata: Dictionary with spatial reference info
    """
    with rasterio.open(file_path) as src:
        # Read all bands
        image = src.read()  # Shape: (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # Shape: (H, W, C)
        
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0]
        }
        
    return image, metadata


def save_geotiff(image: np.ndarray, output_path: str, metadata: dict, 
                 dtype: str = 'uint8'):
    """
    Save NumPy array as GeoTIFF with spatial reference.
    
    Args:
        image: NumPy array (H, W) or (H, W, C)
        output_path: Output file path
        metadata: Metadata dictionary from read_geotiff
        dtype: Output data type
    """
    # Handle single vs multi-band
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)  # (1, H, W)
    else:
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
    
    count, height, width = image.shape
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=metadata.get('crs'),
        transform=metadata.get('transform'),
        compress='lzw'
    ) as dst:
        dst.write(image)


def normalize_image(image: np.ndarray, method: str = 'percentile',
                   percentile_range: Tuple[int, int] = (2, 98)) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
        method: 'percentile', 'minmax', or 'standard'
        percentile_range: Percentile range for percentile method
        
    Returns:
        Normalized image
    """
    if method == 'percentile':
        p_low, p_high = percentile_range
        p_low_val = np.percentile(image, p_low, axis=(0, 1), keepdims=True)
        p_high_val = np.percentile(image, p_high, axis=(0, 1), keepdims=True)
        normalized = (image - p_low_val) / (p_high_val - p_low_val + 1e-8)
        
    elif method == 'minmax':
        min_val = np.min(image, axis=(0, 1), keepdims=True)
        max_val = np.max(image, axis=(0, 1), keepdims=True)
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'standard':
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        std = np.std(image, axis=(0, 1), keepdims=True)
        normalized = (image - mean) / (std + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Clip to [0, 1]
    normalized = np.clip(normalized, 0, 1)
    
    return normalized.astype(np.float32)


def calculate_spectral_indices(image: np.ndarray, 
                               band_indices: dict) -> np.ndarray:
    """
    Calculate spectral indices from multi-band image.
    
    Args:
        image: Multi-band image (H, W, C)
        band_indices: Dictionary mapping band names to indices
                     e.g., {'red': 2, 'nir': 3, 'green': 1}
        
    Returns:
        Stacked image with additional index bands
    """
    indices = []
    
    # NDVI: (NIR - Red) / (NIR + Red)
    if 'nir' in band_indices and 'red' in band_indices:
        nir = image[:, :, band_indices['nir']].astype(np.float32)
        red = image[:, :, band_indices['red']].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-8)
        indices.append(ndvi)
    
    # NDWI: (Green - NIR) / (Green + NIR)
    if 'nir' in band_indices and 'green' in band_indices:
        nir = image[:, :, band_indices['nir']].astype(np.float32)
        green = image[:, :, band_indices['green']].astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-8)
        indices.append(ndwi)
    
    # Stack all indices as new bands
    if indices:
        indices = np.stack(indices, axis=-1)
        enhanced_image = np.concatenate([image, indices], axis=-1)
        return enhanced_image
    
    return image


def visualize_image(image: np.ndarray, bands: List[int] = [0, 1, 2],
                   title: str = "Image", figsize: Tuple[int, int] = (10, 10)):
    """
    Visualize multi-band image.
    
    Args:
        image: Image array
        bands: Band indices to display as RGB
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if len(bands) == 3:
        rgb = image[:, :, bands]
        rgb = np.clip(rgb, 0, 1)
        plt.imshow(rgb)
    else:
        plt.imshow(image[:, :, bands[0]], cmap='gray')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                interpolation: str = 'bilinear') -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: (height, width)
        interpolation: 'bilinear', 'nearest', or 'cubic'
        
    Returns:
        Resized image
    """
    interp_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }
    
    method = interp_methods.get(interpolation, cv2.INTER_LINEAR)
    resized = cv2.resize(image, (target_size[1], target_size[0]), 
                        interpolation=method)
    
    return resized


if __name__ == "__main__":
    print("Image Utilities Module")
    print("Functions: read_geotiff, save_geotiff, normalize_image,")
    print("           calculate_spectral_indices, visualize_image, resize_image")
