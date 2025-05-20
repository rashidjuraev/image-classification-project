import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import get_config
from data.dataloader import get_dataloaders
from models.xception import Xception
from models.inception import Inception
from models.resnet import ResNet
from models.vgg import VGG
from models.mobilenet_v2 import MobileNetV2
from models.squeezenet import SqueezeNet
from utils.metrics import Metrics
from utils.visualization import Visualization
from utils.checkpoints import CheckpointManager

def get_model(model_name, in_channels, num_classes):
    """
    Get model instance based on model name
    
    Args:
        model_name (str): Name of the model
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
    
    Returns:
        Model instance
    """
    # Handle specific ResNet variants
    if model_name == 'resnet50':
        from models.resnet import ResNet
        return ResNet.resnet50(in_channels, num_classes)
    elif model_name == 'resnet18':
        from models.resnet import ResNet
        return ResNet.resnet18(in_channels, num_classes)
    elif model_name == 'resnet34':
        from models.resnet import ResNet
        return ResNet.resnet34(in_channels, num_classes)
    elif model_name == 'resnet101':
        from models.resnet import ResNet
        return ResNet.resnet101(in_channels, num_classes)
    # Handle base models
    elif model_name == 'xception':
        from models.xception import Xception
        return Xception(in_channels, num_classes)
    elif model_name == 'inception' or model_name == 'inception_v3':
        from models.inception import Inception
        return Inception(in_channels, num_classes)
    elif model_name == 'resnet':
        from models.resnet import ResNet
        return ResNet(in_channels, num_classes)
    elif model_name.startswith('vgg'):
        from models.vgg import VGG
        if model_name == 'vgg16':
            return VGG.vgg16(in_channels, num_classes)
        elif model_name == 'vgg19':
            return VGG.vgg19(in_channels, num_classes)
        else:
            return VGG(in_channels, num_classes, 'D')  # Default to VGG16
    elif model_name == 'mobilenet_v2':
        from models.mobilenet_v2 import MobileNetV2
        return MobileNetV2(in_channels, num_classes)
    elif model_name == 'squeezenet':
        from models.squeezenet import SqueezeNet
        return SqueezeNet(in_channels, num_classes)
    else:
        # Try using pretrained model from torchvision
        try:
            from utils.pretrained import load_pretrained_model
            print(f"Attempting to load {model_name} from pretrained models...")
            return load_pretrained_model(model_name, num_classes, use_pretrained=True)
        except:
            raise ValueError(f"Model {model_name} not implemented")

def get_optimizer(model, config):
    """
    Get optimizer based on configuration
    
    Args:
        model: PyTorch model
        config (dict): Configuration dictionary
    
    Returns:
        PyTorch optimizer
    """
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not implemented")

def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config (dict): Configuration dictionary
    
    Returns:
        PyTorch learning rate scheduler
    """
    if config['lr_scheduler'] == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['lr_scheduler_params']['decay_rate']
        )
    elif config['lr_scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=config['lr_scheduler_params']['decay_rate']
        )
    elif config['lr_scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    else:
        return None

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        loader: PyTorch data loader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: PyTorch device
    
    Returns:
        tuple: (mean_loss, mean_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        processed_size += inputs.size(0)
    
    epoch_loss = running_loss / processed_size
    epoch_acc = running_corrects / processed_size * 100
    
    return epoch_loss, epoch_acc

def eval_epoch(model, loader, criterion, device):
    """
    Evaluate the model for one epoch
    
    Args:
        model: PyTorch model
        loader: PyTorch data loader
        criterion: Loss function
        device: PyTorch device
    
    Returns:
        tuple: (mean_loss, mean_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            processed_size += inputs.size(0)
    
    epoch_loss = running_loss / processed_size
    epoch_acc = running_corrects / processed_size * 100
    
    return epoch_loss, epoch_acc

def train(config, resume_training=False, resume_path=None):
    """
    Train the model
    
    Args:
        config (dict): Configuration dictionary
        resume_training (bool): Whether to resume training from checkpoint
        resume_path (str): Path to checkpoint to resume from
    """
    # Get dataloaders
    dataloaders = get_dataloaders(config)
    
    if len(dataloaders) == 4:
        train_loader, val_loader, test_loader, num_classes = dataloaders
    else:
        train_loader, val_loader, num_classes = dataloaders
        test_loader = None
    
    # Get model
    model = get_model(config['model_name'], config['in_channels'], num_classes)
    model = model.to(config['device'])
    
    # Print model summary
    print(f"Model: {config['model_name']}")
    print(f"Number of parameters: {model.get_num_params()}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = get_optimizer(model, config)
    
    # Define learning rate scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config['model_name'])
    
    # Initialize training history
    start_epoch = 0
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_val_acc = 0.0
    
    # Resume training if requested
    if resume_training and resume_path is not None:
        model, optimizer, start_epoch, metrics, _ = checkpoint_manager.load_checkpoint(
            resume_path, model, optimizer
        )
        if 'train_loss' in metrics:
            history['train_loss'] = metrics['train_loss']
        if 'train_accuracy' in metrics:
            history['train_accuracy'] = metrics['train_accuracy']
        if 'val_loss' in metrics:
            history['val_loss'] = metrics['val_loss']
        if 'val_accuracy' in metrics:
            history['val_accuracy'] = metrics['val_accuracy']
        if 'best_val_acc' in metrics:
            best_val_acc = metrics['best_val_acc']
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Evaluate on validation set
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, config['device']
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        metrics = {
            'train_loss': history['train_loss'],
            'train_accuracy': history['train_accuracy'],
            'val_loss': history['val_loss'],
            'val_accuracy': history['val_accuracy'],
            'best_val_acc': best_val_acc
        }
        
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch + 1, metrics, config, is_best
        )
    
    # Save training history
    checkpoint_manager.save_training_history(history)
    
    # Plot training history
    fig = Visualization.plot_training_history(history)
    plt.savefig(os.path.join(checkpoint_manager.model_dir, f"{config['model_name']}_training_history.png"))
    plt.close(fig)
    
    print("Training completed!")

def main():
    """
    Main function for training models
    """
    parser = argparse.ArgumentParser(description="Train classification models")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Get model configuration
    config = get_config(args.model)
    
    # Add device to config
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    train(config, args.resume, args.checkpoint)

if __name__ == "__main__":
    main()