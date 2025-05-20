import torch
import torchvision.models as models
from models.base_model import BaseModel
from models.xception import Xception
from models.inception import Inception
from models.resnet import ResNet
from models.vgg import VGG
from models.mobilenet_v2 import MobileNetV2
from classification.models.squeezenet import SqueezeNet

def load_pretrained_model(model_name, num_classes, use_pretrained=True, freeze_backbone=False):
    """
    Load a model with pre-trained weights and adjust for the target dataset.
    
    Args:
        model_name (str): Name of the model to load
        num_classes (int): Number of classes in the target dataset
        use_pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze the backbone features
        
    Returns:
        nn.Module: Model with pre-trained weights
    """
    model = None
    
    # Load appropriate model with pre-trained weights
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=use_pretrained)
        if num_classes != 1000:  # ImageNet has 1000 classes
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=use_pretrained)
        if num_classes != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=use_pretrained)
        if num_classes != 1000:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
    
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=use_pretrained, aux_logits=True)
        if num_classes != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=use_pretrained)
        if num_classes != 1000:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=use_pretrained)
        if num_classes != 1000:
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
    
    # For Xception, we need custom handling since it's not in torchvision
    elif model_name == 'xception':
        model = Xception(in_channels=3, num_classes=num_classes)
        # Load pre-trained weights if available (would need to be downloaded separately)
        if use_pretrained:
            try:
                model.load_state_dict(torch.load('pretrained_weights/xception.pth'))
                print("Loaded pre-trained Xception weights")
            except:
                print("Pre-trained Xception weights not found. Using random initialization.")
    
    # Freeze backbone if specified
    if freeze_backbone and model is not None:
        if model_name.startswith('resnet'):
            for param in list(model.parameters())[:-2]:  # Exclude FC layer
                param.requires_grad = False
        elif model_name.startswith('vgg'):
            for param in model.features.parameters():
                param.requires_grad = False
        elif model_name == 'inception_v3':
            for param in list(model.parameters())[:-2]:  # Exclude FC layer
                param.requires_grad = False
        elif model_name == 'mobilenet_v2':
            for param in model.features.parameters():
                param.requires_grad = False
        elif model_name == 'squeezenet':
            for param in model.features.parameters():
                param.requires_grad = False
        elif model_name == 'xception':
            # Freeze all layers except the classifier for Xception
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
    
    return model

def convert_to_feature_extractor(model, model_name):
    """
    Convert a pre-trained model to a feature extractor by removing the classifier layers.
    
    Args:
        model: Pre-trained model
        model_name (str): Name of the model architecture
        
    Returns:
        tuple: (feature_extractor, feature_dim)
    """
    feature_dim = 0
    
    if model_name.startswith('resnet'):
        # Remove the classifier
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_dim = model.fc.in_features
    
    elif model_name.startswith('vgg'):
        # Keep only the feature extractor part
        feature_extractor = model.features
        feature_dim = 512 * 7 * 7  # For 224x224 input
    
    elif model_name == 'inception_v3':
        # This is more complex due to auxiliary outputs
        # For simplicity, we'll create a new model without the final classifier
        class InceptionFeatureExtractor(torch.nn.Module):
            def __init__(self, inception_model):
                super(InceptionFeatureExtractor, self).__init__()
                self.model = inception_model
            
            def forward(self, x):
                # Reshape if needed
                if x.shape[2] != 299 or x.shape[3] != 299:
                    x = torch.nn.functional.interpolate(x, size=(299, 299))
                
                # Forward pass through the model
                x = self.model.Conv2d_1a_3x3(x)
                x = self.model.Conv2d_2a_3x3(x)
                x = self.model.Conv2d_2b_3x3(x)
                x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
                x = self.model.Conv2d_3b_1x1(x)
                x = self.model.Conv2d_4a_3x3(x)
                x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
                x = self.model.Mixed_5b(x)
                x = self.model.Mixed_5c(x)
                x = self.model.Mixed_5d(x)
                x = self.model.Mixed_6a(x)
                x = self.model.Mixed_6b(x)
                x = self.model.Mixed_6c(x)
                x = self.model.Mixed_6d(x)
                x = self.model.Mixed_6e(x)
                x = self.model.Mixed_7a(x)
                x = self.model.Mixed_7b(x)
                x = self.model.Mixed_7c(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                return x
        
        feature_extractor = InceptionFeatureExtractor(model)
        feature_dim = model.fc.in_features
    
    elif model_name == 'mobilenet_v2':
        # Remove classifier
        feature_extractor = model.features
        feature_dim = model.classifier[1].in_features
    
    elif model_name == 'squeezenet':
        # Complex due to final convolution
        class SqueezeNetFeatureExtractor(torch.nn.Module):
            def __init__(self, squeezenet_model):
                super(SqueezeNetFeatureExtractor, self).__init__()
                self.features = squeezenet_model.features
            
            def forward(self, x):
                x = self.features(x)
                return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        feature_extractor = SqueezeNetFeatureExtractor(model)
        feature_dim = 512
    
    elif model_name == 'xception':
        # Remove the classifier
        class XceptionFeatureExtractor(torch.nn.Module):
            def __init__(self, xception_model):
                super(XceptionFeatureExtractor, self).__init__()
                self.model = xception_model
            
            def forward(self, x):
                # Copy all forward logic except the classifier
                # Entry Flow
                x = self.model.conv1(x)
                x = self.model.conv2(x)
                res1 = self.model.residual1(x)
                
                x = self.model.sepconv1(x)
                x = self.model.sepconv2(x)
                x = self.model.maxpool1(x)
                x = x + res1
                res2 = self.model.residual2(x)
                
                x = self.model.sepconv3(x)
                x = self.model.sepconv4(x)
                x = self.model.maxpool2(x)
                x = x + res2
                res3 = self.model.residual3(x)
                
                x = self.model.sepconv5(x)
                x = self.model.sepconv6(x)
                x = self.model.maxpool3(x)
                x = x + res3
                
                # Middle Flow
                x = self.model.middleflow(x)
                
                # Exit Flow
                x = self.model.exitsep1(x)
                x = self.model.exitsep2(x)
                x = self.model.exitmaxpool(x)
                exitresidual = self.model.exitresidual(x)
                x = x + exitresidual
                
                x = torch.nn.functional.relu(self.model.exitsep3(x))
                x = torch.nn.functional.relu(self.model.exitsep4(x))
                x = self.model.globalavgpool(x)
                x = torch.flatten(x, 1)
                return x
        
        feature_extractor = XceptionFeatureExtractor(model)
        feature_dim = 2048  # Xception's final feature dimension
    
    return feature_extractor, feature_dim