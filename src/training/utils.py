"""
Training utilities for hierarchical grain classifier.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


class ModelCheckpoint:
    """
    Save and load model checkpoints with training state.
    """
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', save_best_only: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only save when metric improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric = float('-inf')
        
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Training configuration
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'checkpoint_epoch{epoch}_{timestamp}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict,
        metric_name: str = 'val_acc'
    ):
        """
        Save checkpoint if metric improved.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Training configuration
            metric_name: Metric to monitor for improvement
        """
        current_metric = metrics.get(metric_name, float('-inf'))
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            filename = f'best_model_{metric_name}_{current_metric:.4f}.pth'
            self.save(model, optimizer, epoch, metrics, config, filename)
            return True
        return False
    
    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary containing epoch, metrics, config
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics'],
            'config': checkpoint['config']
        }


class MetricTracker:
    """
    Track and log training metrics.
    """
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Args:
            log_dir: Directory to save metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'train': [],
            'val': []
        }
        
    def update(self, split: str, metrics: Dict[str, float], epoch: int):
        """
        Update metrics for current epoch.
        
        Args:
            split: 'train' or 'val'
            metrics: Dictionary of metric values
            epoch: Current epoch number
        """
        record = {'epoch': epoch, **metrics}
        self.history[split].append(record)
    
    def save(self, filename: str = 'metrics.json'):
        """Save metrics history to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to: {filepath}")
    
    def get_best(self, split: str, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get best metric value and epoch.
        
        Args:
            split: 'train' or 'val'
            metric_name: Name of metric to check
            mode: 'max' for highest value, 'min' for lowest
            
        Returns:
            Tuple of (best_value, best_epoch)
        """
        records = self.history[split]
        if not records:
            return None, None
        
        if mode == 'max':
            best_record = max(records, key=lambda x: x.get(metric_name, float('-inf')))
        else:
            best_record = min(records, key=lambda x: x.get(metric_name, float('inf')))
        
        return best_record[metric_name], best_record['epoch']
    
    def print_summary(self):
        """Print summary of best metrics."""
        print("\nTraining Summary:")
        print("=" * 60)
        
        for split in ['train', 'val']:
            if not self.history[split]:
                continue
            
            print(f"\n{split.capitalize()} Set:")
            
            # Get all metric names from last epoch
            last_record = self.history[split][-1]
            metric_names = [k for k in last_record.keys() if k != 'epoch']
            
            for metric_name in metric_names:
                best_val, best_epoch = self.get_best(split, metric_name, mode='max')
                if best_val is not None:
                    print(f"  Best {metric_name}: {best_val:.4f} (epoch {best_epoch})")
        
        print("=" * 60)


class EarlyStopping:
    """
    Early stopping to stop training when metric stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.should_stop = False
        
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'max':
            improved = metric > (self.best_value + self.min_delta)
        else:
            improved = metric < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


def print_epoch_metrics(epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
    """
    Print formatted metrics for current epoch.
    
    Args:
        epoch: Current epoch number
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        epoch_time: Time taken for epoch in seconds
    """
    print(f"\nEpoch {epoch} ({epoch_time:.2f}s):")
    print("-" * 60)
    
    # Print stage-wise metrics
    for stage in [1, 2, 3]:
        stage_key = f'stage{stage}'
        if f'{stage_key}_acc' in train_metrics:
            print(f"  Stage {stage}:")
            print(f"    Train - Loss: {train_metrics[f'{stage_key}_loss']:.4f}, "
                  f"Acc: {train_metrics[f'{stage_key}_acc']:.4f}")
            if f'{stage_key}_acc' in val_metrics:
                print(f"    Val   - Loss: {val_metrics[f'{stage_key}_loss']:.4f}, "
                      f"Acc: {val_metrics[f'{stage_key}_acc']:.4f}")
    
    # Print overall metrics
    if 'overall_acc' in train_metrics:
        print(f"  Overall:")
        print(f"    Train - Acc: {train_metrics['overall_acc']:.4f}")
        if 'overall_acc' in val_metrics:
            print(f"    Val   - Acc: {val_metrics['overall_acc']:.4f}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def print_training_config(config: Dict):
    """
    Print training configuration.
    
    Args:
        config: Dictionary containing training configuration
    """
    print("\nTraining Configuration:")
    print("=" * 60)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
