"""
Experiment 4: Alternative Architecture (EfficientNet-B0 + Attention)

Hypothesis: A more powerful backbone with attention mechanisms can learn
better discriminative features for fine-grained grain morphology, especially
for minority classes (intraclast, broken ooid).

Architecture changes from baseline:
- EfficientNet-B0 backbone (vs ResNet-18)
- Channel attention (Squeeze-and-Excitation)
- Spatial attention
- Dropout regularization (0.3)

Training configuration matches baseline for fair comparison.
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_model import EfficientNetHierarchicalClassifier
from src.data.dataset_new import GrainDatasetNew
from src.training.trainer import Trainer


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ===========================
    # DATALOADERS
    # ===========================
    print("=" * 90)
    print("CREATING DATALOADERS")
    print("=" * 90)
    
    # Load datasets
    train_dataset = GrainDatasetNew(split='train')
    val_dataset = GrainDatasetNew(split='val')
    test_dataset = GrainDatasetNew(split='test')
    
    # Print class distributions (dataset already prints distribution in __init__)
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # ===========================
    # MODEL
    # ===========================
    print("\n" + "=" * 90)
    print("CREATING MODEL")
    print("=" * 90)
    
    model = EfficientNetHierarchicalClassifier(pretrained=True, dropout=0.3)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: EfficientNetHierarchicalClassifier")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Backbone: EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Attention: Channel + Spatial")
    print(f"  Dropout: 0.3")
    
    # ===========================
    # TRAINING SETUP
    # ===========================
    print("\n" + "=" * 90)
    print("TRAINING CONFIGURATION")
    print("=" * 90)
    
    # Loss function parameters (same as baseline)
    focal_loss_params = {
        'stage1': {'alpha': 0.25, 'gamma': 2.0},
        'stage2': {'alpha': 0.5, 'gamma': 2.0},
        'stage3': {'alpha': 0.75, 'gamma': 2.0}
    }
    
    # Scheduler parameters
    scheduler_params = {
        'factor': 0.5,
        'patience': 5
    }
    
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    print(f"\nScheduler: ReduceLROnPlateau")
    print(f"  Mode: max (accuracy)")
    print(f"  Factor: {scheduler_params['factor']}")
    print(f"  Patience: {scheduler_params['patience']} epochs")
    
    print(f"\nLoss: Focal Loss (no class weighting)")
    print(f"  Stage 1 (Peloid): alpha={focal_loss_params['stage1']['alpha']}, gamma={focal_loss_params['stage1']['gamma']}")
    print(f"  Stage 2 (Ooid-like): alpha={focal_loss_params['stage2']['alpha']}, gamma={focal_loss_params['stage2']['gamma']}")
    print(f"  Stage 3 (Whole): alpha={focal_loss_params['stage3']['alpha']}, gamma={focal_loss_params['stage3']['gamma']}")
    
    # ===========================
    # TRAINING
    # ===========================
    print("\n" + "=" * 90)
    print("TRAINING")
    print("=" * 90)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        focal_loss_params=focal_loss_params,
        scheduler_params=scheduler_params,
        early_stopping_patience=args.early_stopping if args.early_stopping > 0 else None
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\n✅ Training complete!")
    print(f"   Checkpoint dir: {args.checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 4: EfficientNet + Attention')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    
    # Data
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/exp4_efficientnet',
                        help='Directory to save checkpoints')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    args = parser.parse_args()
    
    main(args)
