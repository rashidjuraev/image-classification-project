import torch
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

from config import get_config
from models.xception import Xception
from models.inception import Inception
from models.resnet import ResNet
from models.vgg import VGG
from models.mobilenet_v2 import MobileNetV2
from models.squeezenet import SqueezeNet
from data.preprocessing import Preprocessing
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

def predict_image(model, image_path, class_names, config):
    """
    Predict class for a single image
    
    Args:
        model: PyTorch model
        image_path (str): Path to image
        class_names (list): List of class names
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (prediction, probabilities)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess image based on model
    processed_image = Preprocessing.preprocess_for_model(image, config['model_name'])
    
    # Convert to tensor
    if processed_image.ndim == 3:  # If in format (H, W, C)
        processed_image = np.transpose(processed_image, (2, 0, 1))
    
    image_tensor = torch.FloatTensor(processed_image).unsqueeze(0)
    
    # Move to device
    image_tensor = image_tensor.to(config['device'])
    model = model.to(config['device'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, 1)
    
    return prediction.item(), probabilities.squeeze().cpu().numpy()

def visualize_prediction(image_path, prediction, probabilities, class_names):
    """
    Visualize prediction for a single image
    
    Args:
        image_path (str): Path to image
        prediction (int): Predicted class index
        probabilities (numpy.ndarray): Class probabilities
        class_names (list): List of class names
    
    Returns:
        matplotlib.figure.Figure: Figure containing the visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {class_names[prediction]}")
    ax1.axis('off')
    
    # Display probabilities
    top_indices = np.argsort(probabilities)[::-1][:5]  # Top 5 predictions
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    ax2.barh(top_classes, top_probs)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predictions')
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function for evaluating models on new images
    """
    parser = argparse.ArgumentParser(description="Evaluate classification models on new images")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--class_names", type=str, required=True, help="Path to class names file (one class per line)")
    parser.add_argument("--output", type=str, help="Path to save visualization")
    args = parser.parse_args()
    
    # Get model configuration
    config = get_config(args.model)
    
    # Add device to config
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class names
    with open(args.class_names, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Get model
    model = get_model(config['model_name'], config['in_channels'], len(class_names))
    
    # Load model checkpoint
    if args.checkpoint is None:
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
        model, _, _, _, _ = checkpoint_manager.load_checkpoint(args.checkpoint, model)
    
    # Make prediction
    prediction, probabilities = predict_image(model, args.image, class_names, config)
    
    # Print prediction
    print(f"Prediction: {class_names[prediction]}")
    
    # Print top 5 predictions
    top_indices = np.argsort(probabilities)[::-1][:5]  # Top 5 predictions
    print("\nTop 5 predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {probabilities[idx]:.4f}")
    
    # Visualize prediction
    fig = visualize_prediction(args.image, prediction, probabilities, class_names)
    
    # Save visualization if output path is specified
    if args.output is not None:
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()