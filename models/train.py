"""
Model Training Script
Train U-Net for object detection from satellite imagery
"""

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
import segmentation_models as sm


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for loading patches in batches.
    """
    
    def __init__(self, data_dir, batch_size=8, shuffle=True, augmentation=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        
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
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.patch_files))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Load batch
        images = []
        masks = []
        
        for idx in batch_indexes:
            # Load image
            img_path = self.patch_files[idx]
            image = np.load(img_path).astype(np.float32)
            
            # Load mask
            mask_path = self.mask_dir / img_path.name
            mask = np.load(mask_path)
            
            # Apply augmentation
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
    
    train_gen = DataGenerator(
        data_dir=data_config['training_dir'],
        batch_size=training_config['batch_size'],
        shuffle=True,
        augmentation=get_training_augmentation(model_config['input_shape'][0])
    )
    
    val_gen = DataGenerator(
        data_dir=data_config['validation_dir'],
        batch_size=training_config['batch_size'],
        shuffle=False,
        augmentation=get_validation_augmentation(model_config['input_shape'][0])
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
        optimizer=keras.optimizers.Adam(learning_rate=training_config['learning_rate']),
        loss=loss,
        metrics=metrics
    )
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    if training_config['model_checkpoint']['enabled']:
        checkpoint_dir = Path(data_config['models_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / 'best_model.h5'
        
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
    final_model_path = checkpoint_dir / 'final_model.h5'
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
    
    # Plot F1
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Train F1')
    plt.plot(history.history['val_f1_score'], label='Val F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150)
    print(f"Training plots saved to: {save_dir / 'training_history.png'}")


if __name__ == "__main__":
    # Train model
    model, history = train_model()
