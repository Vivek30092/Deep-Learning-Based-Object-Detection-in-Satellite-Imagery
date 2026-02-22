"""
U-Net Architecture for Semantic Segmentation
Deep Learning-Based Object Detection Project
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
from typing import Tuple, Optional





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
    
    # Encoder — filters halved from [64,128,256,512,1024] to [32,64,128,256,512]
    # to reduce model size from 31M to ~8M params for CPU training
    conv1 = conv_block(inputs, 32)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv_block(pool1, 64)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv_block(pool2, 128)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = conv_block(pool3, 256)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    # Bottleneck
    conv5 = conv_block(pool4, 512)
    
    # Decoder
    up6 = upconv_block(conv5, conv4, 256)
    up7 = upconv_block(up6, conv3, 128)
    up8 = upconv_block(up7, conv2, 64)
    up9 = upconv_block(up8, conv1, 32)
    
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


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for segmentation — pure Keras implementation.
    Works with Keras 2.x and 3.x.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Categorical focal loss — pure Keras implementation.
    Handles class imbalance by down-weighting easy examples.
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fn


def get_loss_function(loss_name: str = 'categorical_crossentropy'):
    """
    Get loss function by name.

    Args:
        loss_name: 'categorical_crossentropy', 'dice_loss',
                   'focal_loss', or 'dice_focal'
    Returns:
        Loss function
    """
    if loss_name == 'categorical_crossentropy':
        # label_smoothing prevents log(0) when softmax produces hard 1.0 outputs
        return keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    elif loss_name == 'dice_loss':
        return dice_loss

    elif loss_name == 'focal_loss':
        return focal_loss()

    elif loss_name == 'dice_focal':
        fl = focal_loss()
        def combined(y_true, y_pred):
            return dice_loss(y_true, y_pred) + fl(y_true, y_pred)
        return combined

    else:
        return loss_name


class IOUScore(keras.metrics.Metric):
    """
    Mean Intersection-over-Union across all classes.

    Correct formula per class c:
        intersection_c = sum(y_true_c * y_pred_c)          # true positives for class c
        union_c        = sum(y_true_c) + sum(y_pred_c) - intersection_c
        iou_c          = (intersection_c + smooth) / (union_c + smooth)
    MeanIoU = mean(iou_c for all c)

    Previous implementation was wrong: used total_pixels as part of union,
    which produced IoU > 1.0 (mathematically impossible for correct formula).
    """
    def __init__(self, name='iou_score', smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (B, H, W, C) one-hot float  |  y_pred: (B, H, W, C) softmax float
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        n_classes = tf.shape(y_pred)[-1]

        # Convert predictions to one-hot via argmax
        y_pred_idx = tf.argmax(y_pred, axis=-1)                     # (B, H, W)
        y_pred_oh = tf.one_hot(y_pred_idx, n_classes, dtype=tf.float32)  # (B, H, W, C)

        # Per-class IoU then average
        axes = [0, 1, 2]  # reduce over batch, height, width
        intersection = tf.reduce_sum(y_true * y_pred_oh, axis=axes)  # (C,)
        true_sum = tf.reduce_sum(y_true, axis=axes)                   # (C,)
        pred_sum = tf.reduce_sum(y_pred_oh, axis=axes)                # (C,)
        union = true_sum + pred_sum - intersection                    # (C,)

        iou_per_class = (intersection + self.smooth) / (union + self.smooth)  # (C,)
        mean_iou = tf.reduce_mean(iou_per_class)  # scalar, guaranteed in [0, 1]

        self.iou_sum.assign_add(mean_iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.iou_sum / self.count

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)



def get_metrics():
    """
    Get evaluation metrics — pure Keras implementation.

    Returns:
        List of metrics
    """
    metrics = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        IOUScore(name='iou_score'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
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
