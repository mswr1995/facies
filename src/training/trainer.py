"""
Main training class for hierarchical grain classifier.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import time

from .metrics import HierarchicalMetrics
from .utils import ModelCheckpoint, MetricTracker, EarlyStopping, print_epoch_metrics, get_lr
from ..models import FocalLoss


class Trainer:
    """
    Trainer for hierarchical grain classification model.
    
    Supports:
    - Multi-stage training with independent loss functions
    - Stage-wise freezing/unfreezing
    - Automatic metric tracking and checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Args:
            model: Hierarchical grain classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.loss_functions = {}
        
        # Utilities
        self.checkpoint = ModelCheckpoint(checkpoint_dir)
        self.metrics_tracker = MetricTracker(log_dir)
        self.metric_computer = HierarchicalMetrics()
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        focal_loss_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Setup optimizer, loss functions, and scheduler.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization parameter
            focal_loss_params: Dictionary with alpha and gamma for each stage
            scheduler_params: Parameters for learning rate scheduler
            early_stopping_patience: Patience for early stopping (None to disable)
        """
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss functions for each stage
        if focal_loss_params is None:
            focal_loss_params = {
                'stage1': {'alpha': 0.25, 'gamma': 2.0},
                'stage2': {'alpha': 0.5, 'gamma': 2.0},
                'stage3': {'alpha': 0.75, 'gamma': 2.0}
            }
        
        for stage, params in focal_loss_params.items():
            self.loss_functions[stage] = FocalLoss(
                alpha=params['alpha'],
                gamma=params['gamma']
            )
        
        # Setup scheduler
        if scheduler_params is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5)
            )
        
        # Setup early stopping
        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                mode='max'
            )
        
        print("\nTraining setup complete:")
        print(f"  Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"  Loss functions: Focal Loss with stage-specific parameters")
        print(f"  Scheduler: {'ReduceLROnPlateau' if self.scheduler else 'None'}")
        print(f"  Early stopping: {'Enabled' if self.early_stopping else 'Disabled'}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = {'stage1': 0.0, 'stage2': 0.0, 'stage3': 0.0, 'total': 0.0}
        all_logits = {'stage1': [], 'stage2': [], 'stage3': []}
        all_labels = {'stage1': [], 'stage2': [], 'stage3': []}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, labels, metadata) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            for key in labels:
                labels[key] = labels[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # Compute losses for each stage
            total_loss = 0.0
            
            # Stage 1: All samples
            loss_stage1 = self.loss_functions['stage1'](
                logits['stage1'],
                labels['stage1'].float()
            )
            total_loss += loss_stage1
            epoch_losses['stage1'] += loss_stage1.item()
            
            # Stage 2: Only non-peloids
            stage2_mask = labels['stage2'] != -1
            if stage2_mask.sum() > 0:
                stage2_logits = logits['stage2'][stage2_mask].view(-1)
                stage2_labels = labels['stage2'][stage2_mask].float().view(-1)
                loss_stage2 = self.loss_functions['stage2'](
                    stage2_logits,
                    stage2_labels
                )
                total_loss += loss_stage2
                epoch_losses['stage2'] += loss_stage2.item()
            
            # Stage 3: Only ooid-likes
            stage3_mask = labels['stage3'] != -1
            if stage3_mask.sum() > 0:
                stage3_logits = logits['stage3'][stage3_mask].view(-1)
                stage3_labels = labels['stage3'][stage3_mask].float().view(-1)
                loss_stage3 = self.loss_functions['stage3'](
                    stage3_logits,
                    stage3_labels
                )
                total_loss += loss_stage3
                epoch_losses['stage3'] += loss_stage3.item()
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            
            # Collect predictions for metrics
            for key in logits:
                all_logits[key].append(logits[key].detach().cpu())
                all_labels[key].append(labels[key].detach().cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'lr': get_lr(self.optimizer)
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Concatenate all predictions and compute metrics
        for key in all_logits:
            all_logits[key] = torch.cat(all_logits[key], dim=0)
            all_labels[key] = torch.cat(all_labels[key], dim=0)
        
        metrics = self.metric_computer.compute_all_metrics(all_logits, all_labels, self.model)
        
        # Add losses to metrics
        metrics['stage1_loss'] = epoch_losses['stage1']
        metrics['stage2_loss'] = epoch_losses['stage2']
        metrics['stage3_loss'] = epoch_losses['stage3']
        metrics['total_loss'] = epoch_losses['total']
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_losses = {'stage1': 0.0, 'stage2': 0.0, 'stage3': 0.0, 'total': 0.0}
        all_logits = {'stage1': [], 'stage2': [], 'stage3': []}
        all_labels = {'stage1': [], 'stage2': [], 'stage3': []}
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]  ")
        
        with torch.no_grad():
            for images, labels, metadata in progress_bar:
                # Move to device
                images = images.to(self.device)
                for key in labels:
                    labels[key] = labels[key].to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Compute losses
                total_loss = 0.0
                
                loss_stage1 = self.loss_functions['stage1'](
                    logits['stage1'],
                    labels['stage1'].float()
                )
                total_loss += loss_stage1
                epoch_losses['stage1'] += loss_stage1.item()
                
                stage2_mask = labels['stage2'] != -1
                if stage2_mask.sum() > 0:
                    stage2_logits = logits['stage2'][stage2_mask].view(-1)
                    stage2_labels = labels['stage2'][stage2_mask].float().view(-1)
                    loss_stage2 = self.loss_functions['stage2'](
                        stage2_logits,
                        stage2_labels
                    )
                    total_loss += loss_stage2
                    epoch_losses['stage2'] += loss_stage2.item()
                
                stage3_mask = labels['stage3'] != -1
                if stage3_mask.sum() > 0:
                    stage3_logits = logits['stage3'][stage3_mask].view(-1)
                    stage3_labels = labels['stage3'][stage3_mask].float().view(-1)
                    loss_stage3 = self.loss_functions['stage3'](
                        stage3_logits,
                        stage3_labels
                    )
                    total_loss += loss_stage3
                    epoch_losses['stage3'] += loss_stage3.item()
                
                epoch_losses['total'] += total_loss.item()
                
                # Collect predictions
                for key in logits:
                    all_logits[key].append(logits[key].cpu())
                    all_labels[key].append(labels[key].cpu())
                
                progress_bar.set_postfix({'loss': total_loss.item()})
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Concatenate and compute metrics
        for key in all_logits:
            all_logits[key] = torch.cat(all_logits[key], dim=0)
            all_labels[key] = torch.cat(all_labels[key], dim=0)
        
        metrics = self.metric_computer.compute_all_metrics(all_logits, all_labels, self.model)
        
        # Add losses
        metrics['stage1_loss'] = epoch_losses['stage1']
        metrics['stage2_loss'] = epoch_losses['stage2']
        metrics['stage3_loss'] = epoch_losses['stage3']
        metrics['total_loss'] = epoch_losses['total']
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        config: Optional[Dict] = None
    ):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            config: Training configuration dictionary
        """
        if config is None:
            config = {}
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            # Print metrics
            print_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Track metrics
            self.metrics_tracker.update('train', train_metrics, epoch)
            self.metrics_tracker.update('val', val_metrics, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['overall_acc'])
            
            # Save checkpoint
            if val_metrics['overall_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['overall_acc']
                self.checkpoint.save_best(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    config,
                    metric_name='overall_acc'
                )
            
            # Check early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['overall_acc']):
                    print(f"\nStopping early at epoch {epoch}")
                    break
        
        # Save final metrics
        self.metrics_tracker.save()
        self.metrics_tracker.print_summary()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
