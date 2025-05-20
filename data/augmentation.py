import torchvision.transforms as transforms
import torch
import random
import numpy as np
from PIL import ImageFilter, ImageOps

class RandomGaussianBlur:
    """Apply Gaussian blur with random sigma"""
    def __init__(self, radius_min=0.1, radius_max=2.0, p=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class RandomGrayscale:
    """Convert image to grayscale with probability p"""
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.grayscale(img).convert('RGB')
        return img

class DataAugmentation:
    """
    Class for handling various data augmentation strategies
    """
    
    @staticmethod
    def get_train_transforms(model_name, input_size=224, augmentation_level='standard'):
        """
        Get transforms for training images based on the model
        
        Args:
            model_name (str): Name of the model
            input_size (int): Input size for the model
            augmentation_level (str): Level of augmentation: 'minimal', 'standard', or 'aggressive'
            
        Returns:
            transforms: Transforms for training images
        """
        # Define the size based on the model
        if model_name in ["inception", "xception"]:
            # Inception and Xception models typically use 299x299 inputs
            size = 299
        else:
            size = input_size
        
        # Create normalization transform with ImageNet mean and std
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Define augmentation based on level
        if augmentation_level == 'minimal':
            # Basic augmentation for when data is limited or similar to test data
            return transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        elif augmentation_level == 'standard':
            # Standard augmentation suite good for most scenarios
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                RandomGaussianBlur(),
                transforms.ToTensor(),
                normalize,
            ])
        
        elif augmentation_level == 'aggressive':
            # Aggressive augmentation for better generalization
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomRotation(30),
                RandomGaussianBlur(p=0.6),
                RandomGrayscale(p=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        
        else:
            raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    @staticmethod
    def get_val_transforms(model_name, input_size=224):
        """
        Get transforms for validation images based on the model
        
        Args:
            model_name (str): Name of the model
            input_size (int): Input size for the model
            
        Returns:
            transforms: Transforms for validation images
        """
        # Define the size based on the model
        if model_name in ["inception", "xception"]:
            size = 299
        else:
            size = input_size
            
        # Validation transforms - center crop with normalization
        return transforms.Compose([
            transforms.Resize(int(size * 1.1)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    @staticmethod
    def get_test_time_augmentation_transforms(model_name, input_size=224):
        """
        Get multiple transforms for test-time augmentation
        
        Args:
            model_name (str): Name of the model
            input_size (int): Input size for the model
            
        Returns:
            list: List of different transforms for test-time augmentation
        """
        # Define the size based on the model
        if model_name in ["inception", "xception"]:
            size = 299
        else:
            size = input_size
            
        # Normalization transform
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Define a list of different transforms for test-time augmentation
        tta_transforms = [
            # Original image (center crop)
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]),
            # Horizontally flipped
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]),
            # Slightly rotated (5 degrees)
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.CenterCrop(size),
                transforms.RandomRotation(degrees=(5, 5)),
                transforms.ToTensor(),
                normalize,
            ]),
            # Slightly rotated (-5 degrees)
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.CenterCrop(size),
                transforms.RandomRotation(degrees=(-5, -5)),
                transforms.ToTensor(),
                normalize,
            ]),
            # Slight brightness adjustment
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.CenterCrop(size),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                normalize,
            ]),
        ]
        
        return tta_transforms