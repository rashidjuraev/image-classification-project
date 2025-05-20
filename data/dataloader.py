import torch
from torch.utils.data import DataLoader
from data.dataset import ImageClassificationDataset, FaceExpressionDataset
from data.augmentation import DataAugmentation
from data.cifar100 import CIFAR100Handler

def get_dataloaders(config):
    """
    Create data loaders for training and validation
    
    Args:
        config (dict): Configuration dictionary containing dataset paths and model settings
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # Check dataset type
    if config['dataset_type'] == 'cifar100':
        # CIFAR-100 dataset
        return get_cifar100_dataloaders(config)
    else:
        # Custom datasets (face expression, etc.)
        return get_custom_dataloaders(config)

def get_cifar100_dataloaders(config):
    """
    Get dataloaders for CIFAR-100 dataset
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Create CIFAR-100 handler
    cifar_handler = CIFAR100Handler(root=config.get('cifar100_root', './data'))
    
    # Get model-specific transforms
    augmentation_level = config.get('augmentation_level', 'standard')
    train_transform, test_transform = CIFAR100Handler.get_transforms_for_model(
        config['model_name'],
        image_size=config['input_size'],
        augmentation_level=augmentation_level
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = cifar_handler.get_dataloaders(
        batch_size=config['batch_size'],
        train_transform=train_transform,
        test_transform=test_transform,
        validation_split=config.get('validation_split', 0.1),
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Print dataset information
    print(f"CIFAR-100 Dataset:")
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Input size: {config['input_size']}")
    print(f"Augmentation level: {augmentation_level}")
    
    return train_loader, val_loader, test_loader, num_classes

def get_custom_dataloaders(config):
    """
    Get dataloaders for custom datasets (face expression, etc.)
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # Get the transforms based on the model name
    augmentation_level = config.get('augmentation_level', 'standard')
    train_transforms = DataAugmentation.get_train_transforms(
        config['model_name'], 
        config.get('input_size', 224),
        augmentation_level
    )
    val_transforms = DataAugmentation.get_val_transforms(
        config['model_name'], 
        config.get('input_size', 224)
    )
    
    # Create datasets
    if config['dataset_type'] == 'face_expression':
        train_dataset = FaceExpressionDataset(config['train_data_path'], transforms=train_transforms)
        val_dataset = FaceExpressionDataset(config['val_data_path'], transforms=val_transforms)
    else:
        train_dataset = ImageClassificationDataset(config['train_data_path'], transforms=train_transforms)
        val_dataset = ImageClassificationDataset(config['val_data_path'], transforms=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    num_classes = len(train_dataset.get_class_names())
    
    # Print dataset information
    print(f"Dataset: {config['dataset_type']}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {train_dataset.get_class_distribution()}")
    
    return train_loader, val_loader, None, num_classes  # None for test_loader to match CIFAR return