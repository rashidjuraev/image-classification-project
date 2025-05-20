import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Metrics:
    """
    Class for computing various evaluation metrics
    """
    
    @staticmethod
    def accuracy(outputs, targets):
        """
        Compute accuracy
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            float: Accuracy score
        """
        _, preds = torch.max(outputs, dim=1)
        return (preds == targets).float().mean().item()
    
    @staticmethod
    def compute_batch_metrics(outputs, targets):
        """
        Compute metrics for a batch
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            dict: Dictionary containing metrics
        """
        _, preds = torch.max(outputs, dim=1)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='macro', zero_division=0),
            'recall': recall_score(targets, preds, average='macro', zero_division=0),
            'f1': f1_score(targets, preds, average='macro', zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(outputs, targets, class_names):
        """
        Compute confusion matrix
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Ground truth targets
            class_names (list): List of class names
            
        Returns:
            numpy.ndarray: Confusion matrix
        """
        _, preds = torch.max(outputs, dim=1)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        cm = confusion_matrix(targets, preds)
        return cm, class_names
    
    @staticmethod
    def compute_class_metrics(outputs, targets, class_names):
        """
        Compute per-class metrics
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Ground truth targets
            class_names (list): List of class names
            
        Returns:
            dict: Dictionary containing per-class metrics
        """
        _, preds = torch.max(outputs, dim=1)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        class_precision = precision_score(targets, preds, average=None, zero_division=0)
        class_recall = recall_score(targets, preds, average=None, zero_division=0)
        class_f1 = f1_score(targets, preds, average=None, zero_division=0)
        
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i]
            }
        
        return class_metrics