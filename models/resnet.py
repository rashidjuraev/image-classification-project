import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class ResidualBlock(nn.Module):
    """
    Residual block for ResNet
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet (deeper variants)
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(BaseModel):
    """
    ResNet architecture (ResNet-50 by default)
    """
    def __init__(self, in_channels=3, num_classes=7, blocks_config=[3, 4, 6, 3], block=Bottleneck):
        super(ResNet, self).__init__(in_channels, num_classes)
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, blocks_config[0])
        self.layer2 = self._make_layer(block, 128, blocks_config[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_config[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_config[3], stride=2)
        
        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if needed (to match dimensions)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # First block may have a downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update input channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and final classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    @staticmethod
    def resnet18(in_channels=3, num_classes=7):
        """
        Create a ResNet-18 model
        """
        return ResNet(in_channels, num_classes, [2, 2, 2, 2], ResidualBlock)
    
    @staticmethod
    def resnet34(in_channels=3, num_classes=7):
        """
        Create a ResNet-34 model
        """
        return ResNet(in_channels, num_classes, [3, 4, 6, 3], ResidualBlock)
    
    @staticmethod
    def resnet50(in_channels=3, num_classes=7):
        """
        Create a ResNet-50 model
        """
        return ResNet(in_channels, num_classes, [3, 4, 6, 3], Bottleneck)
    
    @staticmethod
    def resnet101(in_channels=3, num_classes=7):
        """
        Create a ResNet-101 model
        """
        return ResNet(in_channels, num_classes, [3, 4, 23, 3], Bottleneck)
    
    @staticmethod
    def resnet152(in_channels=3, num_classes=7):
        """
        Create a ResNet-152 model
        """
        return ResNet(in_channels, num_classes, [3, 8, 36, 3], Bottleneck)