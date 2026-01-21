"""
Experiment 2: Data Augmentation Training
Train with stronger augmentation to improve rare class detection.
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_augmented import create_augmented_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss
from src.training.trainer import Trainer
from src.training.utils import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description='Train with strong augmentation')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/exp2_augmented')
    parser.add_argument('--strong-aug', type=bool, default=True)
    
    args = parser.parse_args()
    
    device = 'cpu'
    print(f"Device: {device}")
    
    # Create dataloaders with strong augmentation
    print("\n" + "="*90)
    print("CREATING DATALOADERS WITH STRONG AUGMENTATION")
    print("="*90)
    
    train_loader, val_loader, test_loader = create_augmented_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_augmentation=args.strong_aug
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
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
    
    # Loss function with focal loss for imbalance
    criterion_stage1 = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_stage2 = FocalLoss(alpha=0.5, gamma=2.0)
    criterion_stage3 = FocalLoss(alpha=0.75, gamma=2.0)
    
    loss_functions = {
        'stage1': criterion_stage1,
        'stage2': criterion_stage2,
        'stage3': criterion_stage3
    }
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # Scheduler
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
    print(f"\nLoss: Focal Loss")
    print(f"  Stage 1 (Peloid): alpha=0.25, gamma=2.0")
    print(f"  Stage 2 (Ooid-like): alpha=0.5, gamma=2.0")
    print(f"  Stage 3 (Whole): alpha=0.75, gamma=2.0")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir='logs/exp2'
    )
    
    # Setup optimizer and loss in trainer
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.loss_functions = loss_functions
    trainer.early_stopping = EarlyStopping(patience=args.early_stopping)
    
    # Train
    print("\n" + "="*90)
    print("TRAINING")
    print("="*90)
    
    trainer.train(num_epochs=args.epochs)
    
    # Save training info
    checkpoint_dir = Path(args.checkpoint_dir)
    training_info = {
        'experiment': 'Data Augmentation (Strong)',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'early_stopping_patience': args.early_stopping,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'strong_augmentation': args.strong_aug
        },
        'augmentation': {
            'train': 'Rotate, Flip, ElasticTransform, Perspective, ShiftScaleRotate, BrightnessContrast, HueSaturation, RandomGamma, GaussNoise, GaussianBlur, MotionBlur, CoarseDropout',
            'val': 'Normalize only',
            'test': 'Normalize only'
        }
    }
    
    with open(checkpoint_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Checkpoint dir: {checkpoint_dir}")


if __name__ == '__main__':
    main()
