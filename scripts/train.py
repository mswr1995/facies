"""
Training script for hierarchical grain classification.

Usage:
    python scripts/train.py --fold 0 --epochs 50
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import GrainDataset
from src.data.samplers import HierarchicalBalancedSampler, StageWiseBatchSampler
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.training.trainer import Trainer


def load_fold_config(fold: int):
    """Load fold configuration."""
    config_path = Path(f'data/processed/cv_splits/fold_{fold}.json')
    if not config_path.exists():
        raise ValueError(f"Fold {fold} not found at {config_path}")
    
    with open(config_path, 'r') as f:
        fold_data = json.load(f)
    
    return fold_data


def create_data_loaders(fold: int, batch_size: int = 32, num_workers: int = 0, 
                        use_sampler: str = 'none'):
    """Create train and validation data loaders.
    
    Args:
        fold: Fold number
        batch_size: Batch size
        num_workers: Number of data loader workers
        use_sampler: Sampling strategy ('none', 'weighted', 'stagewise')
    """
    metadata_path = f'data/processed/fold_{fold}_metadata.json'
    patches_dir = 'data/processed/patches'
    
    # Create datasets
    train_dataset = GrainDataset(
        metadata_path=metadata_path,
        patches_dir=patches_dir,
        split='train'
    )
    
    val_dataset = GrainDataset(
        metadata_path=metadata_path,
        patches_dir=patches_dir,
        split='val'
    )
    
    # Create train data loader with optional sampler
    if use_sampler == 'weighted':
        print("\nUsing HierarchicalBalancedSampler (weighted sampling)")
        train_sampler = HierarchicalBalancedSampler(
            train_dataset,
            samples_per_epoch=len(train_dataset) * 2,
            stage1_balance=0.5,
            stage3_broken_weight=5.0
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    elif use_sampler == 'stagewise':
        print("\nUsing StageWiseBatchSampler (batch-level balance)")
        batch_sampler = StageWiseBatchSampler(
            train_dataset,
            batch_size=batch_size,
            peloid_ratio=0.5,
            broken_ooid_oversample=3
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        print("\nUsing standard random sampling (no class-aware sampling)")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    # Validation loader always uses standard sampling
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sampling strategy: {use_sampler}")
    
    return train_loader, val_loader


def main(args):
    """Main training function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_sampler=args.sampler
    )
    
    # Create model
    model = HierarchicalGrainClassifier(
        pretrained=args.pretrained
    )
    print(f"\nModel created: {model.__class__.__name__}")
    
    # Create trainer
    checkpoint_dir = f'checkpoints/fold_{args.fold}'
    log_dir = f'logs/fold_{args.fold}'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        focal_loss_params={
            'stage1': {'alpha': 0.25, 'gamma': 2.0},
            'stage2': {'alpha': 0.5, 'gamma': 2.0},
            'stage3': {'alpha': 0.75, 'gamma': 2.0}
        },
        scheduler_params={
            'factor': 0.5,
            'patience': 5
        },
        early_stopping_patience=args.patience
    )
    
    # Training config
    config = {
        'fold': args.fold,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'pretrained': args.pretrained,
        'device': device,
        'sampler': args.sampler
    }
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        config=config
    )
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hierarchical grain classifier')
    
    # Data parameters
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--sampler', type=str, default='stagewise', 
                       choices=['none', 'weighted', 'stagewise'],
                       help='Sampling strategy: none (random), weighted (class-weighted), stagewise (batch-level balance)')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    main(args)
