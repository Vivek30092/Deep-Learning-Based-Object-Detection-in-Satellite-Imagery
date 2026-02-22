"""
Data Augmentation Pipeline
Augmentation techniques for satellite imagery
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(patch_size: int = 256):
    """
    Get augmentation pipeline for training.
    
    Args:
        patch_size: Size of input patches
        
    Returns:
        Albumentations Compose object
    """
    transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        
        # Color/intensity transformations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),  # Scaled for [0,1] normalized data
        
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        # Resize to ensure consistent size
        A.Resize(patch_size, patch_size),
    ], is_check_shapes=False)  # Resize above normalizes sizes; skip pre-check
    
    return transform


def get_validation_augmentation(patch_size: int = 256):
    """
    Get augmentation pipeline for validation (minimal).
    
    Args:
        patch_size: Size of input patches
        
    Returns:
        Albumentations Compose object
    """
    transform = A.Compose([
        A.Resize(patch_size, patch_size),
    ], is_check_shapes=False)  # Resize normalizes sizes; skip pre-check
    
    return transform


def get_preprocessing(normalization: str = 'percentile',
                     percentile_range: Tuple[int, int] = (2, 98)):
    """
    Get preprocessing pipeline.
    
    Args:
        normalization: Normalization method
        percentile_range: Range for percentile normalization
        
    Returns:
        Albumentations Compose object
    """
    if normalization == 'percentile':
        # Note: Percentile normalization needs to be done beforehand
        # This is just a placeholder for normalization to [0, 1]
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ])
    
    return transform


def augment_batch(images: np.ndarray, masks: np.ndarray,
                 augmentation) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to a batch of images and masks.
    
    Args:
        images: Batch of images (B, H, W, C)
        masks: Batch of masks (B, H, W) or (B, H, W, C)
        augmentation: Albumentations transform
        
    Returns:
        Augmented (images, masks)
    """
    aug_images = []
    aug_masks = []
    
    for i in range(len(images)):
        augmented = augmentation(image=images[i], mask=masks[i])
        aug_images.append(augmented['image'])
        aug_masks.append(augmented['mask'])
    
    return np.array(aug_images), np.array(aug_masks)


# Traditional augmentation functions (without albumentations)

def random_flip(image: np.ndarray, mask: Optional[np.ndarray] = None,
               horizontal: bool = True, vertical: bool = True) -> Tuple:
    """
    Randomly flip image and mask.
    
    Args:
        image: Input image
        mask: Input mask (optional)
        horizontal: Enable horizontal flip
        vertical: Enable vertical flip
        
    Returns:
        (flipped_image, flipped_mask)
    """
    if horizontal and np.random.rand() > 0.5:
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
    
    if vertical and np.random.rand() > 0.5:
        image = np.flipud(image)
        if mask is not None:
            mask = np.flipud(mask)
    
    return image, mask


def random_rotate(image: np.ndarray, mask: Optional[np.ndarray] = None,
                 angles: list = [0, 90, 180, 270]) -> Tuple:
    """
    Randomly rotate image and mask.
    
    Args:
        image: Input image
        mask: Input mask (optional)
        angles: List of rotation angles
        
    Returns:
        (rotated_image, rotated_mask)
    """
    angle = np.random.choice(angles)
    
    if angle != 0:
        k = angle // 90
        image = np.rot90(image, k)
        if mask is not None:
            mask = np.rot90(mask, k)
    
    return image, mask


def adjust_brightness_contrast(image: np.ndarray,
                              brightness_range: Tuple[float, float] = (0.8, 1.2),
                              contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Adjust brightness and contrast.
    
    Args:
        image: Input image
        brightness_range: Brightness adjustment range
        contrast_range: Contrast adjustment range
        
    Returns:
        Adjusted image
    """
    # Brightness
    brightness = np.random.uniform(*brightness_range)
    image = image * brightness
    
    # Contrast
    contrast = np.random.uniform(*contrast_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast + mean
    
    return np.clip(image, 0, 1)


def add_gaussian_noise(image: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image
        std: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


if __name__ == "__main__":
    print("Data Augmentation Module")
    print("Functions: get_training_augmentation, get_validation_augmentation")
    print("           random_flip, random_rotate, adjust_brightness_contrast")
