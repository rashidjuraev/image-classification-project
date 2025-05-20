import os
import torch
import json
import datetime

class CheckpointManager:
    """
    Class for handling model checkpoints and saving/loading training state
    """
    
    def __init__(self, model_name, save_dir='checkpoints'):
        """
        Initialize checkpoint manager
        
        Args:
            model_name (str): Name of the model
            save_dir (str): Directory to save checkpoints
        """
        self.model_name = model_name
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create model-specific directory
        self.model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, config=None, is_best=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch (int): Current epoch
            metrics (dict): Dictionary of metrics
            config (dict, optional): Configuration dictionary
            is_best (bool): Whether this is the best model so far
        """
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if config is not None:
            checkpoint['config'] = config
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_dir, f'{self.model_name}_epoch_{epoch}_{timestamp}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best checkpoint if this is the best model so far
        if is_best:
            best_path = os.path.join(self.model_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path, model, optimizer=None):
        """
        Load model checkpoint
        
        Args:
            path (str): Path to checkpoint
            model: PyTorch model
            optimizer (optional): PyTorch optimizer
            
        Returns:
            tuple: (model, optimizer, epoch, metrics)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        config = checkpoint.get('config', None)
        
        print(f"Loaded checkpoint from {path} (epoch {epoch})")
        return model, optimizer, epoch, metrics, config
    
    def load_best_model(self, model, optimizer=None):
        """
        Load best model
        
        Args:
            model: PyTorch model
            optimizer (optional): PyTorch optimizer
            
        Returns:
            tuple: (model, optimizer, epoch, metrics)
        """
        best_path = os.path.join(self.model_dir, f'{self.model_name}_best.pth')
        return self.load_checkpoint(best_path, model, optimizer)
    
    def save_training_history(self, history, filename=None):
        """
        Save training history
        
        Args:
            history (dict): Dictionary of training history
            filename (str, optional): Filename to save history
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.model_name}_history_{timestamp}.json'
        
        history_path = os.path.join(self.model_dir, filename)
        
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        print(f"Training history saved to {history_path}")
    
    def load_training_history(self, filename):
        """
        Load training history
        
        Args:
            filename (str): Filename of training history
            
        Returns:
            dict: Dictionary of training history
        """
        history_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"Training history not found at {history_path}")
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return history