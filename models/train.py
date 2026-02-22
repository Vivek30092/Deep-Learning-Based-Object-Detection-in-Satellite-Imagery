"""
Model Training Script
Train U-Net for object detection from satellite imagery
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable MKL/oneDNN to prevent memory allocation errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress verbose TF logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import build_custom_unet, get_loss_function, get_metrics
from preprocessing.data_augmentation import get_training_augmentation, get_validation_augmentation


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for loading patches in batches.
    """
    
    def __init__(self, data_dir, batch_size=8, shuffle=True, augmentation=None, n_channels=9, **kwargs):
        super().__init__(**kwargs)  # Fixes PyDataset warning
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.n_channels = n_channels  # Expected channel count; enforced per patch
        
        # Get all patch files
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        self.patch_files = sorted(list(self.image_dir.glob('*.npy')))
        
        self.indexes = np.arange(len(self.patch_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.patch_files) / self.batch_size))
    
    def __getitem__(self, index):
        """Get batch."""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.patch_files))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        images = []
        masks = []
        
        for idx in batch_indexes:
            # Load image
            img_path = self.patch_files[idx]
            image = np.load(img_path).astype(np.float32)
            img_h, img_w = image.shape[:2]
            
            # Load mask — create blank if file is missing
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
            else:
                mask = np.load(mask_path)
            
            # Ensure mask is 2D (H, W)
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
            if mask.ndim != 2 or mask.size == 0:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
            
            # Resize mask to match image dims if stale patches caused mismatch
            if mask.shape != (img_h, img_w):
                mask = tf.image.resize(
                    mask[..., np.newaxis], [img_h, img_w], method='nearest'
                ).numpy().squeeze(-1).astype(np.uint8)
            
            # Normalize image to exactly n_channels (handles stale 1-band vs 9-band patches)
            if image.ndim == 2:  # (H, W) -> (H, W, 1)
                image = image[..., np.newaxis]
            c = image.shape[-1]
            if c < self.n_channels:  # Too few channels -> zero-pad
                pad = np.zeros((*image.shape[:2], self.n_channels - c), dtype=np.float32)
                image = np.concatenate([image, pad], axis=-1)
            elif c > self.n_channels:  # Too many channels -> trim
                image = image[..., :self.n_channels]

            # --- CRITICAL: Normalize to [0, 1] BEFORE augmentation ---
            # Must come first: albumentations RandomBrightnessContrast assumes [0,1] float.
            # Sentinel-2 DN values are 0-10000+; without this, activations explode -> NaN loss.
            p2  = np.percentile(image, 2)
            p98 = np.percentile(image, 98)
            if p98 > p2:
                image = np.clip(image, p2, p98)
                image = (image - p2) / (p98 - p2)
            else:
                image = np.zeros_like(image)  # degenerate patch -> blank
            image = image.astype(np.float32)

            # Apply augmentation (on [0,1] normalized data)
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            images.append(image)
            masks.append(mask)
        
        images = np.array(images)
        masks = np.array(masks)
        
        # Convert masks to one-hot encoding
        if len(masks.shape) == 3:  # (B, H, W)
            masks = tf.keras.utils.to_categorical(masks, num_classes=5)
        
        return images, masks
    
    def on_epoch_end(self):
        """Shuffle after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_model(config_path: str = 'config/config.yaml'):
    """
    Train U-Net model.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Deep Learning-Based Object Detection - Training")
    print("=" * 60)
    
    # Extract configuration
    model_config = config['model']
    training_config = config['training']
    data_config = config['paths']
    
    # Create data generators
    print("\nCreating data generators...")
    
    n_channels = model_config['input_shape'][2]

    train_gen = DataGenerator(
        data_dir=data_config['training_dir'],
        batch_size=training_config['batch_size'],
        shuffle=True,
        augmentation=get_training_augmentation(model_config['input_shape'][0]),
        n_channels=n_channels
    )
    
    val_gen = DataGenerator(
        data_dir=data_config['validation_dir'],
        batch_size=training_config['batch_size'],
        shuffle=False,
        augmentation=get_validation_augmentation(model_config['input_shape'][0]),
        n_channels=n_channels
    )
    
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    
    # Build model
    print("\nBuilding U-Net model...")
    
    model = build_custom_unet(
        input_shape=tuple(model_config['input_shape']),
        n_classes=model_config['n_classes'],
        activation=model_config['activation']
    )
    
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    
    loss = get_loss_function(training_config['loss_function'])
    metrics = get_metrics()
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=training_config['learning_rate'],
            clipnorm=1.0  # Gradient clipping: prevents a single bad batch from exploding weights
        ),
        loss=loss,
        metrics=metrics
    )
    
    # Callbacks
    callbacks = []
    
    # Always create checkpoint dir (needed for final model save too)
    checkpoint_dir = Path(data_config['models_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint callback
    if training_config['model_checkpoint']['enabled']:
        checkpoint_path = checkpoint_dir / 'best_model.keras'
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=training_config['model_checkpoint']['monitor'],
            mode=training_config['model_checkpoint']['mode'],
            save_best_only=training_config['model_checkpoint']['save_best_only'],
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Early stopping
    if training_config['early_stopping']['enabled']:
        early_stop = keras.callbacks.EarlyStopping(
            monitor=training_config['early_stopping']['monitor'],
            patience=training_config['early_stopping']['patience'],
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stop)
    
    # Reduce learning rate
    if training_config['reduce_lr']['enabled']:
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_config['reduce_lr']['factor'],
            patience=training_config['reduce_lr']['patience'],
            min_lr=training_config['reduce_lr']['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    # TensorBoard
    log_dir = Path('logs') / datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(log_dir))
    callbacks.append(tensorboard)
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=training_config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.keras'
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    plot_training_history(history, log_dir)
    
    return model, history


def plot_training_history(history, save_dir):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU
    plt.subplot(1, 3, 2)
    plt.plot(history.history['iou_score'], label='Train IoU')
    plt.plot(history.history['val_iou_score'], label='Val IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    # Plot Precision & Recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150)
    print(f"Training plots saved to: {save_dir / 'training_history.png'}")


if __name__ == "__main__":
    # Train model
    model, history = train_model()
