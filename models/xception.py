import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.act:
            return F.relu(x)
        else: 
            return x

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=False):
        super(SeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        if self.act:
            x = F.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        return x

class MiddleFlow(nn.Module):
    def __init__(self, channels):
        super(MiddleFlow, self).__init__()
        self.channels = channels

        self.block1 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))
        
        self.block2 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))
        
        self.block3 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))

        self.block4 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))

        self.block5 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))
        
        self.block6 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))

        self.block7 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))
        
        self.block8 = nn.Sequential(
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True),
            SeparableConv(channels, channels, act=True))
            
    def forward(self, x):
        block1 = self.block1(x)
        fusion1 = x + block1

        block2 = self.block2(fusion1)
        fusion2 = fusion1 + block2

        block3 = self.block3(fusion2)
        fusion3 = fusion2 + block3

        block4 = self.block4(fusion3)
        fusion4 = fusion3 + block4

        block5 = self.block5(fusion4)
        fusion5 = fusion4 + block5

        block6 = self.block6(fusion5)
        fusion6 = fusion5 + block6

        block7 = self.block7(fusion6)
        fusion7 = fusion6 + block7

        block8 = self.block8(fusion7)
        fusion8 = fusion7 + block8

        return fusion8

class Xception(BaseModel):
    def __init__(self, in_channels=3, num_classes=7):
        super(Xception, self).__init__(in_channels, num_classes)

        # Entry flow
        self.conv1 = ConvBlock(in_channels, 32, kernel_size=3, stride=2, act=True)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, act=True)
        self.residual1 = ConvBlock(64, 128, kernel_size=1, stride=2)

        self.sepconv1 = SeparableConv(64, 128)
        self.sepconv2 = SeparableConv(128, 128, act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual2 = ConvBlock(128, 256, kernel_size=1, stride=2)

        self.sepconv3 = SeparableConv(128, 256, act=True)
        self.sepconv4 = SeparableConv(256, 256, act=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual3 = ConvBlock(256, 728, kernel_size=1, stride=2)

        self.sepconv5 = SeparableConv(256, 728, act=True)
        self.sepconv6 = SeparableConv(728, 728, act=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Middle flow
        self.middleflow = MiddleFlow(728)

        # Exit Flow
        self.exitsep1 = SeparableConv(728, 728, act=True)
        self.exitsep2 = SeparableConv(728, 1024, act=True)
        self.exitmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.exitresidual = ConvBlock(728, 1024, kernel_size=1, stride=2)

        self.exitsep3 = SeparableConv(1024, 1536)
        self.exitsep4 = SeparableConv(1536, 2048)
        self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry Flow
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        res1 = self.residual1(conv2)
        
        sep1 = self.sepconv1(conv2)
        sep2 = self.sepconv2(sep1)
        pool1 = self.maxpool1(sep2)
        fusion1 = pool1 + res1
        res2 = self.residual2(fusion1)

        sep3 = self.sepconv3(fusion1)
        sep4 = self.sepconv4(sep3)
        pool2 = self.maxpool2(sep4)
        fusion2 = pool2 + res2
        res3 = self.residual3(fusion2)

        sep5 = self.sepconv5(fusion2)
        sep6 = self.sepconv6(sep5)
        pool3 = self.maxpool3(sep6)
        fusion3 = pool3 + res3

        # Middle Flow
        middleflow = self.middleflow(fusion3)

        # Exit Flow
        exitsep1 = self.exitsep1(middleflow)
        exitsep2 = self.exitsep2(exitsep1)
        exitmaxpool = self.exitmaxpool(exitsep2)
        exitresidual = self.exitresidual(middleflow)
        exitfusion = exitresidual + exitmaxpool

        exitsep3 = F.relu(self.exitsep3(exitfusion))
        exitsep4 = F.relu(self.exitsep4(exitsep3))
        globalavgpool = self.globalavgpool(exitsep4)
        dropped = self.dropout(globalavgpool)
        x = torch.flatten(dropped, start_dim=1)
        classifier = self.classifier(x)

        return classifier