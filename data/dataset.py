import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageClassificationDataset(Dataset):
    """
    Generic dataset class for image classification tasks
    """
    
    def __init__(self, data_path, transforms=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            transforms: Optional transforms to apply to the images
        """
        super(ImageClassificationDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.classes = os.listdir(data_path)
        
        # Build image paths and labels
        self._build_dataset()
    
    def _build_dataset(self):
        """
        Build the dataset by creating the image paths and labels lists
        """
        images_path = []
        labels_dict = {}
        
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                images_path.append(image_path)
                labels_dict[image_path] = i
        
        self.images_path_list = images_path
        self.labels_dict = labels_dict
    
    def __len__(self):
        """
        Return the number of samples in the dataset
        
        Returns:
            int: Number of samples
        """
        return len(self.images_path_list)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        image_path = self.images_path_list[index]
        label = self.labels_dict[image_path]
        image = Image.open(image_path).convert("RGB")
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, label
    
    def get_class_names(self):
        """
        Get the class names
        
        Returns:
            list: List of class names
        """
        return self.classes
    
    def get_class_distribution(self):
        """
        Get the class distribution
        
        Returns:
            dict: Dictionary mapping class names to counts
        """
        class_dist = {cls: 0 for cls in self.classes}
        
        for image_path, label in self.labels_dict.items():
            class_name = self.classes[label]
            class_dist[class_name] += 1
        
        return class_dist

# Face expression recognition dataset, a specific implementation
class FaceExpressionDataset(ImageClassificationDataset):
    """
    Dataset class for face expression recognition
    """
    
    def __init__(self, data_path, transforms=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            transforms: Optional transforms to apply to the images
        """
        super(FaceExpressionDataset, self).__init__(data_path, transforms)