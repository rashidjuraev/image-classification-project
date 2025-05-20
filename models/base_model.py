import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Base class for all classification models.
    All models should inherit from this class to ensure a common interface.
    """
    def __init__(self, in_channels, num_classes):
        """
        Initialize the base model.
        
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
        """
        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        pass
    
    def get_num_params(self):
        """
        Get the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        self.load_state_dict(torch.load(path))