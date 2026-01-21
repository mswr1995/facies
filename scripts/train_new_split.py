"""
Train hierarchical grain classifier on new clean train/val/test split.
"""
import argparse
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import create_new_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.training.trainer import Trainer


def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_loader, val_loader, _ = create_new_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = HierarchicalGrainClassifier(
        pretrained=True,
        freeze_backbone=False
    )
    model.to(device)
    model.print_parameter_summary()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Setup training
    print("\n" + "="*80)
    print("TRAINING SETUP")
    print("="*80)
    
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
        early_stopping_patience=args.early_stopping
    )
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"\nTraining for up to {args.epochs} epochs")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print()
    
    history = trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hierarchical grain classifier')
    
    # Data
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/new_split',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
