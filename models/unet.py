"""
U-Net Architecture for Semantic Segmentation
Deep Learning-Based Object Detection Project
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import segmentation_models as sm


def unet_model(input_shape: Tuple[int, int, int] = (256, 256, 8),
              n_classes: int = 5,
              encoder: str = 'resnet34',
              encoder_weights: str = 'imagenet',
              activation: str = 'softmax') -> Model:
    """
    Create U-Net model using segmentation_models library.
    
    Args:
        input_shape: Input shape (H, W, C)
        n_classes: Number of output classes
        encoder: Encoder backbone (resnet34, efficientnetb0, etc.)
        encoder_weights: Pre-trained weights ('imagenet' or None)
        activation: Final activation ('softmax' or 'sigmoid')
        
    Returns:
        Keras Model
    """
    # Note: segmentation_models only supports 3-channel inputs for imagenet weights
    # We'll need to handle multi-band inputs differently
    
    if input_shape[-1] == 3 and encoder_weights == 'imagenet':
        # Use pre-trained encoder
        model = sm.Unet(
            encoder,
            input_shape=input_shape,
            classes=n_classes,
            activation=activation,
            encoder_weights=encoder_weights
        )
    else:
        # Build custom U-Net for multi-band inputs
        model = build_custom_unet(input_shape, n_classes, activation)
    
    return model


def build_custom_unet(input_shape: Tuple[int, int, int],
                     n_classes: int,
                     activation: str = 'softmax') -> Model:
    """
    Build custom U-Net architecture from scratch.
    Supports multi-band inputs beyond RGB.
    
    Args:
        input_shape: Input shape (H, W, C)
        n_classes: Number of classes
        activation: Final activation
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    # Block 2
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Block 3
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Block 4
    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    # Bottleneck
    conv5 = conv_block(pool4, 1024)
    
    # Decoder (Expanding Path)
    # Block 6
    up6 = upconv_block(conv5, conv4, 512)
    
    # Block 7
    up7 = upconv_block(up6, conv3, 256)
    
    # Block 8
    up8 = upconv_block(up7, conv2, 128)
    
    # Block 9
    up9 = upconv_block(up8, conv1, 64)
    
    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(up9)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    
    return model


def conv_block(inputs, filters: int, kernel_size: int = 3):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        kernel_size: Kernel size
        
    Returns:
        Output tensor
    """
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def upconv_block(inputs, skip_connection, filters: int):
    """
    Upsampling block with skip connection.
    
    Args:
        inputs: Input tensor
        skip_connection: Skip connection from encoder
        filters: Number of filters
        
    Returns:
        Output tensor
    """
    # Upsample
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    
    # Concatenate with skip connection
    x = layers.concatenate([x, skip_connection])
    
    # Convolution
    x = conv_block(x, filters)
    
    return x


def get_loss_function(loss_name: str = 'categorical_crossentropy'):
    """
    Get loss function by name.
    
    Args:
        loss_name: Loss function name
        
    Returns:
        Loss function
    """
    if loss_name == 'categorical_crossentropy':
        return keras.losses.CategoricalCrossentropy()
    
    elif loss_name == 'dice_loss':
        return sm.losses.DiceLoss()
    
    elif loss_name == 'focal_loss':
        return sm.losses.CategoricalFocalLoss()
    
    elif loss_name == 'dice_focal':
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.CategoricalFocalLoss()
        return dice_loss + focal_loss
    
    else:
        return loss_name


def get_metrics():
    """
    Get evaluation metrics.
    
    Returns:
        List of metrics
    """
    metrics = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        sm.metrics.IOUScore(name='iou_score'),
        sm.metrics.FScore(name='f1_score'),
    ]
    
    return metrics


if __name__ == "__main__":
    # Test model creation
    print("Building U-Net model...")
    
    model = build_custom_unet(
        input_shape=(256, 256, 8),
        n_classes=5,
        activation='softmax'
    )
    
    model.summary()
    
    print(f"\nModel created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
