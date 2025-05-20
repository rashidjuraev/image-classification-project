"""
Configuration file for the classification models
"""

import os
import torch

# Paths
DATASET_ROOT = "../../DATASETS/classification_datasets"
FACE_EXPR_TRAIN_PATH = os.path.join(DATASET_ROOT, "face_expression_recognation/train")
FACE_EXPR_VAL_PATH = os.path.join(DATASET_ROOT, "face_expression_recognation/validation")
CIFAR100_ROOT = "./data"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default settings for all models
DEFAULT_CONFIG = {
    'batch_size': 16,
    'num_workers': 2,
    'epochs': 50,
    'learning_rate': 3e-4,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'lr_scheduler': 'exponential',  # 'exponential', 'step', 'cosine', etc.
    'lr_scheduler_params': {
        'decay_rate': 0.1,
    },
    'optimizer': 'adam',  # 'adam', 'sgd', etc.
}

# CIFAR-100 configuration 
CIFAR100_CONFIG = {
    'batch_size': 128,
    'num_workers': 4,
    'epochs': 100,
    'learning_rate': 0.001,  # Higher learning rate for CIFAR
    'momentum': 0.9,
    'weight_decay': 5e-4,  # More regularization for smaller dataset
    'lr_scheduler': 'cosine_warmup',  # Better for CIFAR training
    'lr_scheduler_params': {
        'T_max': 100,  # Total epochs
        'eta_min': 1e-6,  # Minimum LR
        'warmup_epochs': 5,  # Warmup for first few epochs
    },
    'optimizer': 'sgd',  # SGD usually works better on CIFAR
    'nesterov': True,  # Use Nesterov momentum
    'augmentation_level': 'aggressive',  # Use stronger augmentation
    'dataset_type': 'cifar100',
    'use_pretrained': True,  # Use pretrained weights
    'progressive_resizing': True,  # Train with progressively larger sizes
    'mixup_alpha': 0.2,  # Use mixup augmentation
    'validation_split': 0.1,  # Percentage of training data for validation
}

# Model-specific configurations
MODEL_CONFIGS = {
    # Face expression recognition configs
    'xception': {
        'in_channels': 3,
        'input_size': 299,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'xception',
        'checkpoints_dir': 'checkpoints/xception',
        **DEFAULT_CONFIG
    },
    'inception': {
        'in_channels': 3,
        'input_size': 299,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'inception',
        'checkpoints_dir': 'checkpoints/inception',
        **DEFAULT_CONFIG
    },
    'resnet': {
        'in_channels': 3,
        'input_size': 224,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'resnet',
        'checkpoints_dir': 'checkpoints/resnet',
        **DEFAULT_CONFIG
    },
    'vgg': {
        'in_channels': 3,
        'input_size': 224,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'vgg',
        'checkpoints_dir': 'checkpoints/vgg',
        **DEFAULT_CONFIG
    },
    'mobilenet_v2': {
        'in_channels': 3,
        'input_size': 224,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'mobilenet_v2',
        'checkpoints_dir': 'checkpoints/mobilenet_v2',
        **DEFAULT_CONFIG
    },
    'squeezenet': {
        'in_channels': 3,
        'input_size': 224,
        'train_data_path': FACE_EXPR_TRAIN_PATH,
        'val_data_path': FACE_EXPR_VAL_PATH,
        'dataset_type': 'face_expression',
        'model_name': 'squeezenet',
        'checkpoints_dir': 'checkpoints/squeezenet',
        **DEFAULT_CONFIG
    },
    
    # CIFAR-100 configs - higher accuracy settings 
    'cifar_resnet50': {
        'in_channels': 3,
        'input_size': 224,  # We'll resize CIFAR images to this
        'dataset_type': 'cifar100',
        'model_name': 'resnet50',
        'model_variant': 'cifar',  # Specialized variant for CIFAR
        'checkpoints_dir': 'checkpoints/cifar_resnet50',
        **CIFAR100_CONFIG
    },
    'cifar_xception': {
        'in_channels': 3,
        'input_size': 299,  # Xception uses 299x299
        'dataset_type': 'cifar100',
        'model_name': 'xception',
        'checkpoints_dir': 'checkpoints/cifar_xception',
        **CIFAR100_CONFIG
    },
    'cifar_mobilenetv2': {
        'in_channels': 3,
        'input_size': 224,
        'dataset_type': 'cifar100',
        'model_name': 'mobilenet_v2',
        'checkpoints_dir': 'checkpoints/cifar_mobilenetv2',
        **CIFAR100_CONFIG
    },
    'cifar_vgg16': {
        'in_channels': 3,
        'input_size': 224,
        'dataset_type': 'cifar100',
        'model_name': 'vgg16',
        'checkpoints_dir': 'checkpoints/cifar_vgg16',
        **{**CIFAR100_CONFIG, 'batch_size': 64}  # VGG has more parameters, reduce batch size
    },
    'cifar_inception': {
        'in_channels': 3,
        'input_size': 299,  # Inception uses 299x299
        'dataset_type': 'cifar100',
        'model_name': 'inception_v3',
        'checkpoints_dir': 'checkpoints/cifar_inception',
        **CIFAR100_CONFIG
    },
    'cifar_squeezenet': {
        'in_channels': 3,
        'input_size': 224,
        'dataset_type': 'cifar100',
        'model_name': 'squeezenet',
        'checkpoints_dir': 'checkpoints/cifar_squeezenet',
        **CIFAR100_CONFIG
    },
    
    # Ensemble config
    'cifar_ensemble': {
        'in_channels': 3,
        'input_size': 224,
        'dataset_type': 'cifar100',
        'model_name': 'ensemble',
        'checkpoints_dir': 'checkpoints/cifar_ensemble',
        'models': ['resnet50', 'xception', 'mobilenet_v2'],  # Models to include in ensemble
        **CIFAR100_CONFIG
    }
}

def get_config(model_name):
    """
    Get configuration for a specific model
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing model configuration
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in configurations")
    return MODEL_CONFIGS[model_name]