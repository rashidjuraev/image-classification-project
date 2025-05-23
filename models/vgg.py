import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class VGG(BaseModel):
    """
    VGG architecture (VGG16 by default)
    """
    def __init__(self, in_channels=3, num_classes=7, config='D'):
        super(VGG, self).__init__(in_channels, num_classes)
        
        # Define VGG configurations
        configurations = {
            # VGG11
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            # VGG13
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            # VGG16
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            # VGG19
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        # Ensure valid configuration
        if config not in configurations:
            raise ValueError(f"Invalid VGG configuration '{config}'. Choose from 'A', 'B', 'D', or 'E'.")
        
        # Build feature extractor
        self.features = self._make_layers(configurations[config], in_channels)
        
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def vgg11(in_channels=3, num_classes=7):
        """
        Create a VGG11 model
        """
        return VGG(in_channels, num_classes, 'A')
    
    @staticmethod
    def vgg13(in_channels=3, num_classes=7):
        """
        Create a VGG13 model
        """
        return VGG(in_channels, num_classes, 'B')
    
    @staticmethod
    def vgg16(in_channels=3, num_classes=7):
        """
        Create a VGG16 model
        """
        return VGG(in_channels, num_classes, 'D')
    
    @staticmethod
    def vgg19(in_channels=3, num_classes=7):
        """
        Create a VGG19 model
        """
        return VGG(in_channels, num_classes, 'E')