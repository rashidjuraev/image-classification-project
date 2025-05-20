import torch
import torch.nn as nn
import numpy as np
import argparse
import os
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
    if model_name == 'xception':
        return Xception(in_channels, num_classes)
    elif model_name == 'inception':
        return Inception(in_channels, num_classes)
    elif model_name == 'resnet':
        return ResNet(in_channels, num_classes)
    elif model_name == 'vgg':
        return VGG(in_channels, num_classes)
    elif model_name == 'mobilenet_v2':
        return MobileNetV2(in_channels, num_classes)
    elif model_name == 'squeezenet':
        return SqueezeNet(in_channels, num_classes)
    else:
        raise ValueError(f"Model {model_name} not implemented")

def test_model(config, checkpoint_path=None):
    """
    Test the model on the validation set
    
    Args:
        config (dict): Configuration dictionary
        checkpoint_path (str): Path to checkpoint to load
    """
    # Get dataloaders
    _, val_loader, num_classes = get_dataloaders(config)
    
    # Get class names
    class_names = val_loader.dataset.get_class_names()
    
    # Get model
    model = get_model(config['model_name'], config['in_channels'], num_classes)
    model = model.to(config['device'])
    
    # Load model checkpoint
    if checkpoint_path is None:
        # Try to load best model
        checkpoint_manager = CheckpointManager(config['model_name'])
        try:
            model, _, _, _, _ = checkpoint_manager.load_best_model(model)
        except FileNotFoundError:
            print("Best model checkpoint not found. Please specify a checkpoint path.")
            return
    else:
        # Load specified checkpoint
        checkpoint_manager = CheckpointManager(config['model_name'])
        model, _, _, _, _ = checkpoint_manager.load_checkpoint(checkpoint_path, model)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics
    all_outputs = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Testing"):
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Save outputs and labels for metrics computation
            all_outputs.append(outputs)
            all_labels.append(labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            processed_size += inputs.size(0)
            
            # Visualize some predictions (on the first batch only)
            if len(all_outputs) == 1:
                fig = Visualization.visualize_model_predictions(
                    model, inputs, labels, class_names, config['device']
                )
                plt.savefig(os.path.join(checkpoint_manager.model_dir, f"{config['model_name']}_predictions.png"))
                plt.close(fig)
    
    # Compute overall metrics
    test_loss = running_loss / processed_size
    test_acc = running_corrects / processed_size * 100
    
    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Compute confusion matrix
    cm, _ = Metrics.compute_confusion_matrix(all_outputs, all_labels, class_names)
    
    # Compute per-class metrics
    class_metrics = Metrics.compute_class_metrics(all_outputs, all_labels, class_names)
    
    # Print overall metrics
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot confusion matrix
    cm_fig = Visualization.plot_confusion_matrix(cm, class_names)
    plt.savefig(os.path.join(checkpoint_manager.model_dir, f"{config['model_name']}_confusion_matrix.png"))
    plt.close(cm_fig)
    
    print("Testing completed!")

def main():
    """
    Main function for testing models
    """
    parser = argparse.ArgumentParser(description="Test classification models")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load")
    args = parser.parse_args()
    
    # Get model configuration
    config = get_config(args.model)
    
    # Add device to config
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test model
    test_model(config, args.checkpoint)

if __name__ == "__main__":
    main()