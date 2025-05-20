import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from PIL import Image

class CIFAR100Handler:
    """
    Handler for CIFAR-100 dataset
    """
    
    def __init__(self, root='./data', download=True):
        """
        Initialize the CIFAR-100 handler
        
        Args:
            root (str): Root directory for dataset storage
            download (bool): Whether to download the dataset if not found
        """
        self.root = root
        self.download = download
        self.class_names = None
        self.train_dataset = None
        self.test_dataset = None
        
        # Create data directory if it doesn't exist
        os.makedirs(root, exist_ok=True)
    
    def get_datasets(self, train_transform=None, test_transform=None, validation_split=0.1):
        """
        Get CIFAR-100 datasets with specified transforms
        
        Args:
            train_transform: Transform to apply to training set
            test_transform: Transform to apply to test set
            validation_split (float): Percentage of training data to use as validation
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, num_classes)
        """
        # Default transforms if none provided
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        
        # Load CIFAR-100 dataset
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=self.download,
            transform=train_transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=False,
            download=self.download,
            transform=test_transform
        )
        
        # Get class names
        self.class_names = self.train_dataset.classes
        
        # Create validation set from training set
        train_size = int(len(self.train_dataset) * (1 - validation_split))
        val_size = len(self.train_dataset) - train_size
        
        # Create random indices for splitting
        indices = torch.randperm(len(self.train_dataset)).tolist()
        
        # Create subsets
        train_subset = Subset(self.train_dataset, indices[:train_size])
        val_subset = Subset(self.train_dataset, indices[train_size:])
        
        # For validation set, we need to change the transform
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=False,  # No need to download again
            transform=test_transform  # Use test transform for validation
        )
        val_subset = Subset(val_dataset, indices[train_size:])
        
        return train_subset, val_subset, self.test_dataset, 100  # CIFAR-100 has 100 classes
    
    def get_dataloaders(self, batch_size=128, train_transform=None, test_transform=None, 
                       validation_split=0.1, num_workers=4, pin_memory=True):
        """
        Get CIFAR-100 data loaders
        
        Args:
            batch_size (int): Batch size
            train_transform: Transform to apply to training set
            test_transform: Transform to apply to test set
            validation_split (float): Percentage of training data to use as validation
            num_workers (int): Number of worker threads for loading data
            pin_memory (bool): Whether to pin memory for faster GPU transfer
            
        Returns:
            tuple: (train_loader, val_loader, test_loader, num_classes)
        """
        # Get datasets
        train_dataset, val_dataset, test_dataset, num_classes = self.get_datasets(
            train_transform, test_transform, validation_split
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader, num_classes
    
    def get_class_distribution(self):
        """
        Get class distribution from the training set
        
        Returns:
            dict: Dictionary with class names as keys and counts as values
        """
        if self.train_dataset is None:
            # Load dataset if not already loaded
            self.get_datasets()
        
        # Count samples per class
        class_counts = {}
        for _, label in self.train_dataset:
            class_name = self.class_names[label]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        return class_counts
    
    def print_class_info(self):
        """
        Print information about the CIFAR-100 classes
        """
        if self.class_names is None:
            # Load dataset if not already loaded
            self.get_datasets()
        
        print("CIFAR-100 Classes:")
        for i, class_name in enumerate(self.class_names):
            print(f"{i}: {class_name}")
    
    @staticmethod
    def get_transforms_for_model(model_name, image_size=224, augmentation_level='standard'):
        """
        Get transforms specifically designed for the model when using CIFAR-100
        
        Args:
            model_name (str): Name of the model
            image_size (int): Target image size
            augmentation_level (str): Level of augmentation: 'minimal', 'standard', or 'aggressive'
            
        Returns:
            tuple: (train_transform, test_transform)
        """
        # CIFAR-100 normalization values (more accurate than ImageNet)
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
        
        # Basic test transform - resize and normalize
        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])
        
        # Train transform based on augmentation level
        if augmentation_level == 'minimal':
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        elif augmentation_level == 'standard':
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize
            ])
        
        elif augmentation_level == 'aggressive':
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                normalize
            ])
        
        else:
            raise ValueError(f"Unknown augmentation level: {augmentation_level}")
        
        return train_transform, test_transform