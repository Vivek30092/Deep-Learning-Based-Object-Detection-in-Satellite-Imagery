"""
Model Inference Script
Run trained model on new satellite imagery
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import yaml
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.image_utils import read_geotiff, save_geotiff, normalize_image


def sliding_window_inference(model, image: np.ndarray,
                             patch_size: int = 256,
                             overlap: int = 32,
                             batch_size: int = 4) -> np.ndarray:
    """
    Perform inference on large image using sliding window approach.
    
    Args:
        model: Trained Keras model
        image: Input image (H, W, C)
        patch_size: Size of patches
        overlap: Overlap between patches
        batch_size: Batch size for prediction
        
    Returns:
        Prediction mask (H, W, n_classes)
    """
    height, width, channels = image.shape
    
    # Initialize output
    predictions = np.zeros((height, width, model.output_shape[-1]), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    stride = patch_size - overlap
    
    # Calculate number of patches
    num_patches_h = int(np.ceil((height - overlap) / stride))
    num_patches_w = int(np.ceil((width - overlap) / stride))
    
    # Collect all patches
    patches = []
    positions = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            row_start = i * stride
            col_start = j * stride
            
            row_end = min(row_start + patch_size, height)
            col_end = min(col_start + patch_size, width)
            
            # Extract patch
            patch = image[row_start:row_end, col_start:col_end, :]
            
            # Pad if necessary
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, channels), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((row_start, col_start, row_end, col_end))
    
    patches = np.array(patches)
    
    # Predict in batches
    all_predictions = []
    
    for i in tqdm(range(0, len(patches), batch_size), desc="Predicting"):
        batch = patches[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        all_predictions.append(preds)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Stitch predictions back together
    for idx, (row_start, col_start, row_end, col_end) in enumerate(positions):
        pred_patch = all_predictions[idx]
        
        # Get actual patch size
        actual_h = row_end - row_start
        actual_w = col_end - col_start
        
        # Accumulate predictions
        predictions[row_start:row_end, col_start:col_end, :] += pred_patch[:actual_h, :actual_w, :]
        counts[row_start:row_end, col_start:col_end] += 1
    
    # Average overlapping predictions
    counts = np.expand_dims(counts, axis=-1)
    predictions = predictions / (counts + 1e-8)
    
    return predictions


def predict_on_image(model_path: str, image_path: str, output_dir: str,
                    config_path: str = 'config/config.yaml'):
    """
    Run inference on satellite imagery.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input GeoTIFF
        output_dir: Output directory
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    
    # Load image
    print(f"Loading image from: {image_path}")
    image, metadata = read_geotiff(image_path)
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Normalize image
    print("Normalizing image...")
    image = normalize_image(
        image,
        method=config['image']['normalization_method'],
        percentile_range=config['image']['percentile_range']
    )
    
    # Run inference
    print("Running inference...")
    predictions = sliding_window_inference(
        model,
        image,
        patch_size=config['image']['patch_size'],
        overlap=config['image']['overlap'],
        batch_size=config['inference']['batch_size']
    )
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Convert to class labels
    class_predictions = np.argmax(predictions, axis=-1).astype(np.uint8)
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save probability maps
    print("\nSaving probability maps...")
    for class_idx in range(predictions.shape[-1]):
        class_name = config['classes'][class_idx]['name']
        prob_map = predictions[:, :, class_idx]
        
        output_path = output_dir / f'prob_{class_name}.tif'
        save_geotiff(prob_map, str(output_path), metadata, dtype='float32')
        print(f"  Saved: {output_path}")
    
    # Save class predictions
    print("\nSaving classification map...")
    output_path = output_dir / 'classification.tif'
    save_geotiff(class_predictions, str(output_path), metadata, dtype='uint8')
    print(f"  Saved: {output_path}")
    
    # Save individual class masks
    print("\nSaving individual class masks...")
    for class_idx in range(1, predictions.shape[-1]):  # Skip background
        class_name = config['classes'][class_idx]['name']
        class_mask = (class_predictions == class_idx).astype(np.uint8)
        
        output_path = output_dir / f'mask_{class_name}.tif'
        save_geotiff(class_mask, str(output_path), metadata, dtype='uint8')
        print(f"  Saved: {output_path}")
    
    print("\nInference complete!")
    print(f"Results saved to: {output_dir}")
    
    return class_predictions, predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on satellite imagery')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', required=True, help='Path to input GeoTIFF')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    predict_on_image(args.model, args.image, args.output, args.config)
