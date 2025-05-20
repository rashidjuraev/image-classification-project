import torch
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine Annealing with Warmup learning rate scheduler
    """
    def __init__(self, optimizer, T_total, T_warmup=0, eta_min=0, last_epoch=-1):
        """
        Initialize the scheduler
        
        Args:
            optimizer: PyTorch optimizer
            T_total (int): Total number of iterations
            T_warmup (int): Warmup iterations
            eta_min (float): Minimum learning rate
            last_epoch (int): Last epoch
        """
        self.T_total = T_total
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # Linear warmup
            alpha = self.last_epoch / self.T_warmup
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            T_cur = self.last_epoch - self.T_warmup
            T_total = self.T_total - self.T_warmup
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * T_cur / T_total)) / 2
                    for base_lr in self.base_lrs]

class OneCycleLR(_LRScheduler):
    """
    One Cycle Learning Rate Policy as described in the paper:
    'Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates'
    
    The policy cycles the learning rate between two boundaries with a constant
    frequency and gradually reduces the boundaries.
    """
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25., 
                 final_div_factor=1e4, last_epoch=-1):
        """
        Initialize the scheduler
        
        Args:
            optimizer: PyTorch optimizer
            max_lr (float or list): Upper learning rate boundary
            total_steps (int): Total number of training steps
            pct_start (float): Percentage of the cycle spent increasing the learning rate
            div_factor (float): Initial learning rate = max_lr / div_factor
            final_div_factor (float): Minimum learning rate = initial learning rate / final_div_factor
            last_epoch (int): Last epoch
        """
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
        
        # Initialize base learning rates
        self.base_lrs = []
        for max_lr_i in self.max_lr:
            self.base_lrs.append(max_lr_i / self.div_factor)
    
    def get_lr(self):
        if self.last_epoch <= self.step_size_up:
            # Increasing phase
            return [base_lr + (max_lr - base_lr) * 
                   (self.last_epoch / self.step_size_up)
                   for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]
        else:
            # Decreasing phase - cosine annealing to min_lr
            current_step = self.last_epoch - self.step_size_up
            cos_decay = 0.5 * (1 + math.cos(math.pi * current_step / self.step_size_down))
            min_lr = [base_lr / self.final_div_factor for base_lr in self.base_lrs]
            return [min_lr_i + (max_lr_i - min_lr_i) * cos_decay
                    for max_lr_i, min_lr_i in zip(self.max_lr, min_lr)]

def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config (dict): Configuration dictionary
    
    Returns:
        PyTorch learning rate scheduler
    """
    if config['lr_scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.1)
        )
    elif config['lr_scheduler'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('lr_milestones', [30, 60, 90]),
            gamma=config.get('lr_gamma', 0.1)
        )
    elif config['lr_scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', config['epochs']),
            eta_min=config.get('min_lr', 0)
        )
    elif config['lr_scheduler'] == 'cosine_warmup':
        return CosineAnnealingWarmupLR(
            optimizer,
            T_total=config['epochs'],
            T_warmup=config.get('warmup_epochs', 5),
            eta_min=config.get('min_lr', 0)
        )
    elif config['lr_scheduler'] == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', config['learning_rate'] * 10),
            total_steps=config['epochs'],
            pct_start=config.get('pct_start', 0.3),
            div_factor=config.get('div_factor', 25),
            final_div_factor=config.get('final_div_factor', 1e4)
        )
    elif config['lr_scheduler'] == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('lr_gamma', 0.9)
        )
    else:
        return None