import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from torchvision.utils import make_grid
import pandas as pd
import io
import PIL.Image

class Visualization:
    """
    Class for visualization utilities
    """
    
    @staticmethod
    def plot_training_history(history):
        """
        Plot training and validation loss and accuracy
        
        Args:
            history (dict): Dictionary containing training and validation metrics
        
        Returns:
            matplotlib.figure.Figure: Figure containing the plots
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axs[0].plot(history['train_loss'], label='Train loss')
        axs[0].plot(history['val_loss'], label='Validation loss')
        axs[0].set_title('Loss per epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot accuracy
        axs[1].plot(history['train_accuracy'], label='Train accuracy')
        axs[1].plot(history['val_accuracy'], label='Validation accuracy')
        axs[1].set_title('Accuracy per epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Plot confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        
        Returns:
            matplotlib.figure.Figure: Figure containing the confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        return plt.gcf()
    
    @staticmethod
    def plot_class_distribution(dataset):
        """
        Plot class distribution
        
        Args:
            dataset: Dataset object with get_class_distribution method
        
        Returns:
            matplotlib.figure.Figure: Figure containing the class distribution
        """
        dist_dict = dataset.get_class_distribution()
        classes = list(dist_dict.keys())
        counts = list(dist_dict.values())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = plt.bar(classes, counts, color='blue')
        
        # Add annotations on top of each bar
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.xlabel('Classes')
        plt.ylabel('Number of Items')
        plt.title('Class Distribution with Counts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def visualize_model_predictions(model, images, labels, class_names, device, num_images=8):
        """
        Visualize model predictions
        
        Args:
            model: PyTorch model
            images (torch.Tensor): Batch of images
            labels (torch.Tensor): Ground truth labels
            class_names (list): List of class names
            device: PyTorch device
            num_images (int): Number of images to visualize
        
        Returns:
            matplotlib.figure.Figure: Figure containing the visualizations
        """
        model.eval()
        with torch.no_grad():
            images = images[:num_images].to(device)
            labels = labels[:num_images]
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # Create a figure
        fig = plt.figure(figsize=(15, 8))
        
        # Convert images from tensor to numpy for visualization
        images = images.cpu()
        preds = preds.cpu()
        
        # Plot images
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(2, 4, i + 1)
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            
            ax.imshow(img)
            
            color = 'green' if preds[i] == labels[i] else 'red'
            ax.set_title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}", color=color)
            ax.axis('off')
        
        plt.tight_layout()
        return fig