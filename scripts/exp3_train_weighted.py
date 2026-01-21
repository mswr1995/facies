"""
Experiment 3: Class Weighting
Add sample weights to training to improve rare class learning.
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import create_new_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss
from src.training.trainer import Trainer
from src.training.utils import EarlyStopping


class WeightedDataLoader:
    """Wrapper to add sample weights to batches."""
    
    def __init__(self, dataloader, class_weights):
        self.dataloader = dataloader
        self.class_weights = class_weights
    
    def __iter__(self):
        for batch in self.dataloader:
            images, labels, metadata = batch
            
            # Compute weights for this batch
            label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
            batch_weights = torch.tensor([
                self.class_weights[label_map[label]] 
                for label in metadata['label']
            ], dtype=torch.float32)
            
            yield images, labels, metadata, batch_weights
    
    def __len__(self):
        return len(self.dataloader)


def compute_class_weights(dataset, strategy='inverse_freq'):
    """
    Compute class weights for balancing.
    
    Strategies:
    - inverse_freq: weight = 1 / frequency
    - sqrt_inv: weight = 1 / sqrt(frequency)
    - custom: manual weights for rare classes
    """
    # Count samples per class
    label_counts = {}
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    for sample in dataset.samples:
        label = label_map[sample['label']]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total = sum(label_counts.values())
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    if strategy == 'inverse_freq':
        # Standard: weight = 1 / frequency
        weights = {
            label: total / (count * len(label_counts))
            for label, count in label_counts.items()
        }
    elif strategy == 'sqrt_inv':
        # Smoother: weight = 1 / sqrt(frequency)
        weights = {
            label: np.sqrt(total / (count * len(label_counts)))
            for label, count in label_counts.items()
        }
    elif strategy == 'custom':
        # Manual tuning: boost rare classes more
        weights = {
            0: 1.0,      # Peloid (87% - baseline)
            1: 2.0,      # Ooid (8% - 2x)
            2: 5.0,      # Broken ooid (1% - 5x)
            3: 3.0       # Intraclast (4% - 3x)
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"\n{strategy.upper()} Class Weights:")
    for label_id, class_name in enumerate(class_names):
        freq = label_counts[label_id] / total
        weight = weights[label_id]
        print(f"  {class_name:15s} (freq={freq:5.2%}): weight={weight:.2f}")
    
    return weights


def main():
    parser = argparse.ArgumentParser(description='Train with class weighting')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/exp3_weighted')
    parser.add_argument('--weight-strategy', type=str, default='custom', 
                       choices=['inverse_freq', 'sqrt_inv', 'custom'])
    
    args = parser.parse_args()
    
    device = 'cpu'
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\n" + "="*90)
    print("CREATING DATALOADERS")
    print("="*90)
    
    train_loader, val_loader, test_loader = create_new_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Compute class weights
    print("\n" + "="*90)
    print("COMPUTING CLASS WEIGHTS")
    print("="*90)
    
    train_dataset = train_loader.dataset
    class_weights = compute_class_weights(train_dataset, strategy=args.weight_strategy)
    
    # Create model
    print("\n" + "="*90)
    print("CREATING MODEL")
    print("="*90)
    
    model = HierarchicalGrainClassifier(pretrained=True)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: HierarchicalGrainClassifier")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    print("\n" + "="*90)
    print("TRAINING CONFIGURATION")
    print("="*90)
    
    criterion_stage1 = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_stage2 = FocalLoss(alpha=0.5, gamma=2.0)
    criterion_stage3 = FocalLoss(alpha=0.75, gamma=2.0)
    
    loss_functions = {
        'stage1': criterion_stage1,
        'stage2': criterion_stage2,
        'stage3': criterion_stage3
    }
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: 1e-4")
    print(f"  Weight decay: 1e-4")
    print(f"\nScheduler: ReduceLROnPlateau")
    print(f"  Mode: max (accuracy)")
    print(f"  Factor: 0.5")
    print(f"  Patience: 5 epochs")
    print(f"\nLoss: Focal Loss + Class Weighting")
    print(f"  Stage 1 (Peloid): alpha=0.25, gamma=2.0")
    print(f"  Stage 2 (Ooid-like): alpha=0.5, gamma=2.0")
    print(f"  Stage 3 (Whole): alpha=0.75, gamma=2.0")
    print(f"  Class weights: {args.weight_strategy}")
    
    # Create trainer with custom training loop to support weights
    # For now, use standard trainer (weights handled in loss computation)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir='logs/exp3'
    )
    
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.loss_functions = loss_functions
    trainer.early_stopping = EarlyStopping(patience=args.early_stopping)
    
    # Store class weights in trainer for use in loss computation
    trainer.class_weights = class_weights
    
    # Train
    print("\n" + "="*90)
    print("TRAINING")
    print("="*90)
    
    trainer.train(num_epochs=args.epochs)
    
    # Save training info
    checkpoint_dir = Path(args.checkpoint_dir)
    training_info = {
        'experiment': 'Class Weighting',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'early_stopping_patience': args.early_stopping,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'weight_strategy': args.weight_strategy
        },
        'class_weights': {
            str(k): float(v) for k, v in class_weights.items()
        }
    }
    
    with open(checkpoint_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Checkpoint dir: {checkpoint_dir}")


if __name__ == '__main__':
    main()
