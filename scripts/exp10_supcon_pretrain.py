"""
Experiment 10: Supervised Contrastive Learning (SupCon) Pre-training + Fine-tuning

Strategy:
  Phase 1 - SupCon pre-training: Train the ResNet-18 backbone with supervised
            contrastive loss using class-balanced batches. This forces the backbone
            to learn an embedding space where all 4 classes form tight, well-separated
            clusters. Minority classes contribute equally to feature learning.
  
  Phase 2 - Classification fine-tuning: Load the pre-trained backbone, attach the
            3 hierarchical binary heads, and fine-tune with heavy oversampling (15x)
            + focal loss, exactly like exp8.

Why this should work:
  - Exp 1 proved the bottleneck is FEATURE REPRESENTATION, not decision thresholds
  - Current backbone is dominated by peloid gradients (87% of data)
  - SupCon with balanced batches forces the backbone to learn features that
    distinguish ALL classes equally, before any classification head is attached
  - Proven effective with class imbalance and small datasets (creates N^2 training
    pairs from N samples)

Usage:
    # Phase 1: Pre-train backbone (20-30 epochs)
    python scripts/exp10_supcon_pretrain.py --phase pretrain --epochs 30

    # Phase 2: Fine-tune with hierarchical heads (50 epochs) 
    python scripts/exp10_supcon_pretrain.py --phase finetune --epochs 50

    # Both phases sequentially
    python scripts/exp10_supcon_pretrain.py --phase both --pretrain_epochs 30 --finetune_epochs 50
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.data.samplers import StageWiseBatchSampler
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss
from src.models.supcon_loss import SupConLoss, ProjectionHead


# ============================================================================
# Phase 1: SupCon Pre-training
# ============================================================================

def create_supcon_dataloader(batch_size=64):
    """
    Create a class-balanced dataloader for SupCon pre-training.
    
    Key: Each batch must have multiple samples per class so the contrastive
    loss has positive pairs to work with. We use weighted random sampling 
    to ensure all 4 classes appear roughly equally in each batch.
    """
    dataset = GrainDatasetNew(split='train')
    
    # Compute per-sample weights (inverse class frequency)
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    class_counts = defaultdict(int)
    for sample in dataset.samples:
        class_counts[sample['label']] += 1
    
    # Inverse frequency weighting
    weights = []
    for sample in dataset.samples:
        label = sample['label']
        w = 1.0 / class_counts[label]
        weights.append(w)
    
    weights = torch.DoubleTensor(weights)
    
    # WeightedRandomSampler: oversample minority classes so batches are balanced
    sampler = WeightedRandomSampler(
        weights, 
        num_samples=len(dataset) * 2,  # 2x epoch for more contrastive pairs
        replacement=True
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,
        drop_last=True  # Important: avoid tiny last batch with no positive pairs
    )
    
    return loader


class SupConBackbone(nn.Module):
    """
    ResNet-18 backbone + projection head for SupCon pre-training.
    After pre-training, we keep only the backbone (discard projection head).
    """
    
    def __init__(self, pretrained=True, proj_dim=128):
        super(SupConBackbone, self).__init__()
        
        # Same backbone as HierarchicalGrainClassifier
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head (discarded after pre-training)
        self.projection = ProjectionHead(
            input_dim=512, 
            hidden_dim=256, 
            output_dim=proj_dim
        )
    
    def forward(self, x):
        """
        Returns:
            projections: L2-normalized projected features (batch_size, proj_dim)
            features: Raw backbone features (batch_size, 512) - for monitoring
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)  # (B, 512)
        projections = self.projection(features)  # (B, proj_dim), L2-normalized
        return projections, features


def pretrain_supcon(args):
    """Phase 1: Pre-train backbone with supervised contrastive loss."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Phase 1: SupCon Pre-training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.pretrain_epochs}")
    print(f"Batch size: {args.supcon_batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Learning rate: {args.supcon_lr}")
    
    # Create balanced dataloader
    train_loader = create_supcon_dataloader(batch_size=args.supcon_batch_size)
    val_dataset = GrainDatasetNew(split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.supcon_batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = SupConBackbone(pretrained=True, proj_dim=128).to(device)
    
    # Loss and optimizer
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.supcon_lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.pretrain_epochs,
        eta_min=1e-6
    )
    
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(1, args.pretrain_epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        batch_count = 0
        class_seen = defaultdict(int)
        
        for images, labels_dict, metadata in tqdm(train_loader, desc=f"SupCon Epoch {epoch}", leave=False):
            images = images.to(device)
            
            # Convert string labels to integer class IDs
            class_labels = torch.tensor(
                [label_map[l] for l in metadata['label']], 
                dtype=torch.long, 
                device=device
            )
            
            # Track class distribution
            for l in metadata['label']:
                class_seen[l] += 1
            
            # Forward
            projections, _ = model(images)
            loss = criterion(projections, class_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        
        # --- Validate (compute loss on val set to monitor overfitting) ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, labels_dict, metadata in val_loader:
                images = images.to(device)
                class_labels = torch.tensor(
                    [label_map[l] for l in metadata['label']],
                    dtype=torch.long,
                    device=device
                )
                projections, _ = model(images)
                loss = criterion(projections, class_labels)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        class_str = " | ".join(f"{k}: {v}" for k, v in sorted(class_seen.items()))
        print(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr:.6f}")
        print(f"          Batch class distribution: {class_str}")
        
        history.append({
            'epoch': epoch, 
            'train_loss': avg_loss, 
            'val_loss': avg_val_loss,
            'lr': lr
        })
        
        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.backbone.state_dict(), checkpoint_dir / 'supcon_backbone_best.pth')
            print(f"          *** New best val loss: {avg_val_loss:.4f} ***")
    
    # Save final backbone
    torch.save(model.backbone.state_dict(), checkpoint_dir / 'supcon_backbone_final.pth')
    
    with open(checkpoint_dir / 'supcon_pretrain_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_loss': best_loss}, f, indent=2)
    
    print(f"\nSupCon pre-training complete!")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Backbone saved to: {checkpoint_dir / 'supcon_backbone_best.pth'}")
    
    return checkpoint_dir / 'supcon_backbone_best.pth'


# ============================================================================
# Phase 2: Fine-tuning with hierarchical heads (same as exp8 but with SupCon backbone)
# ============================================================================

def finetune_hierarchical(args, backbone_path=None):
    """Phase 2: Fine-tune with hierarchical heads using pre-trained backbone."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if backbone_path is None:
        backbone_path = Path(args.checkpoint_dir) / 'supcon_backbone_best.pth'
    
    print(f"\n{'='*60}")
    print(f"Phase 2: Hierarchical Fine-tuning")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Pretrained backbone: {backbone_path}")
    print(f"Epochs: {args.finetune_epochs}")
    print(f"Broken ooid oversample: {args.oversample}x")
    
    # Create dataloaders (same as exp8)
    train_dataset = GrainDatasetNew(split='train')
    val_dataset = GrainDatasetNew(split='val')
    
    train_sampler = StageWiseBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        broken_ooid_oversample=args.oversample
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create hierarchical model
    model = HierarchicalGrainClassifier(pretrained=True, freeze_backbone=False).to(device)
    
    # Load SupCon pre-trained backbone weights
    if backbone_path.exists():
        backbone_state = torch.load(backbone_path, map_location=device)
        model.backbone.load_state_dict(backbone_state)
        print(f"Loaded SupCon backbone from {backbone_path}")
    else:
        print(f"WARNING: Backbone checkpoint not found at {backbone_path}")
        print(f"Training from ImageNet initialization.")
    
    # Phase 2a: Briefly freeze backbone, train heads only (5 warmup epochs)
    # This prevents random head gradients from corrupting the good SupCon features
    model.freeze_backbone()
    
    loss_fns = {
        'stage1': FocalLoss(alpha=0.25, gamma=2.0),
        'stage2': FocalLoss(alpha=0.5, gamma=2.0),
        'stage3': FocalLoss(alpha=0.75, gamma=2.0),
    }
    
    # Warmup: train heads only
    head_params = list(model.head_stage1.parameters()) + \
                  list(model.head_stage2.parameters()) + \
                  list(model.head_stage3.parameters())
    
    warmup_optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=1e-4)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    best_acc = 0.0
    history = []
    
    warmup_epochs = 5
    print(f"\n--- Head warmup ({warmup_epochs} epochs, backbone frozen) ---")
    
    for epoch in range(1, warmup_epochs + 1):
        loss = _train_epoch(model, train_loader, warmup_optimizer, loss_fns, device)
        acc = _evaluate(model, val_loader, device, label_map)
        print(f"Warmup {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'epoch': epoch, 'phase': 'warmup', 'loss': loss, 'val_acc': acc})
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
    
    # Phase 2b: Unfreeze backbone and fine-tune everything
    model.unfreeze_backbone()
    
    # Use lower LR for backbone, higher for heads
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.finetune_lr * 0.1},  # 10x lower for backbone
        {'params': head_params, 'lr': args.finetune_lr},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.finetune_epochs,
        eta_min=1e-6
    )
    
    print(f"\n--- Full fine-tuning ({args.finetune_epochs} epochs, backbone unfrozen) ---")
    print(f"  Backbone LR: {args.finetune_lr * 0.1}")
    print(f"  Heads LR: {args.finetune_lr}")
    
    patience = 15
    no_improve = 0
    
    for epoch in range(1, args.finetune_epochs + 1):
        loss = _train_epoch(model, train_loader, optimizer, loss_fns, device)
        acc, per_class = _evaluate_detailed(model, val_loader, device, label_map)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, val_acc={acc:.4f}, lr={lr:.6f}")
        print(f"          Pel={per_class.get(0,0):.0%} Ooid={per_class.get(1,0):.0%} "
              f"Brok={per_class.get(2,0):.0%} Intra={per_class.get(3,0):.0%}")
        
        history.append({
            'epoch': warmup_epochs + epoch, 
            'phase': 'finetune', 
            'loss': loss, 
            'val_acc': acc,
            'per_class': {k: round(v, 4) for k, v in per_class.items()}
        })
        
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            print(f"          *** New best: {acc:.4f} ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break
    
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    
    with open(checkpoint_dir / 'finetune_history.json', 'w') as f:
        json.dump({
            'history': history, 
            'best_val_acc': best_acc,
            'backbone': str(backbone_path),
            'oversample': args.oversample
        }, f, indent=2)
    
    print(f"\nFine-tuning complete!")
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pth'}")


# ============================================================================
# Shared training/evaluation helpers
# ============================================================================

def _train_epoch(model, loader, optimizer, loss_fns, device):
    """Train one epoch (same logic as exp8)."""
    model.train()
    total_loss = 0
    
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        for k in labels:
            labels[k] = labels[k].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = loss_fns['stage1'](logits['stage1'], labels['stage1'].float())
        
        mask2 = labels['stage2'] != -1
        if mask2.sum() > 0:
            loss += loss_fns['stage2'](logits['stage2'][mask2], labels['stage2'][mask2].float())
        
        mask3 = labels['stage3'] != -1
        if mask3.sum() > 0:
            loss += loss_fns['stage3'](logits['stage3'][mask3], labels['stage3'][mask3].float())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def _evaluate(model, loader, device, label_map):
    """Evaluate overall accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            preds = model.get_predictions(logits)
            
            true_labels = torch.tensor([label_map[l] for l in metadata['label']])
            correct += (preds.cpu() == true_labels).sum().item()
            total += len(true_labels)
    
    return correct / total


def _evaluate_detailed(model, loader, device, label_map):
    """Evaluate with per-class recall."""
    model.eval()
    correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            preds = model.get_predictions(logits)
            
            true_labels = torch.tensor([label_map[l] for l in metadata['label']])
            correct += (preds.cpu() == true_labels).sum().item()
            total += len(true_labels)
            
            for i, (pred, true) in enumerate(zip(preds.cpu(), true_labels)):
                class_total[true.item()] += 1
                if pred == true:
                    class_correct[true.item()] += 1
    
    overall_acc = correct / total
    per_class = {}
    for cls in range(4):
        if class_total[cls] > 0:
            per_class[cls] = class_correct[cls] / class_total[cls]
        else:
            per_class[cls] = 0.0
    
    return overall_acc, per_class


# ============================================================================
# Evaluation on test set
# ============================================================================

def evaluate_test(args):
    """Evaluate the best model on the held-out test set."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation")
    print(f"{'='*60}")
    
    # Load model
    model = HierarchicalGrainClassifier(pretrained=False).to(device)
    model_path = checkpoint_dir / 'best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load test set
    test_dataset = GrainDatasetNew(split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    label_names = {v: k for k, v in label_map.items()}
    
    # Compute confusion matrix
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels, metadata in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = model.get_predictions(logits)
            
            true_labels = torch.tensor([label_map[l] for l in metadata['label']])
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(true_labels.tolist())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Overall accuracy
    overall_acc = (all_preds == all_true).mean()
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({(all_preds == all_true).sum()}/{len(all_true)})")
    
    # Per-class recall
    print(f"\nPer-Class Recall:")
    class_names_short = {0: 'Peloid', 1: 'Ooid', 2: 'Broken', 3: 'Intraclast'}
    for cls in range(4):
        mask = all_true == cls
        total = mask.sum()
        correct = ((all_preds == cls) & mask).sum()
        recall = correct / total if total > 0 else 0
        print(f"  {class_names_short[cls]:12s}: {correct}/{total} ({recall:.1%})")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':14s}  {'Predicted':^40s}")
    print(f"{'Actual':14s}  {'Pel':>8s}  {'Ooid':>8s}  {'Brok':>8s}  {'Intra':>8s}")
    for true_cls in range(4):
        row = []
        for pred_cls in range(4):
            count = ((all_true == true_cls) & (all_preds == pred_cls)).sum()
            row.append(str(count))
        print(f"{class_names_short[true_cls]:14s}  {'  '.join(f'{r:>8s}' for r in row)}")
    
    # Save results
    results = {
        'overall_accuracy': float(overall_acc),
        'per_class_recall': {},
        'confusion_matrix': {}
    }
    for cls in range(4):
        mask = all_true == cls
        total = int(mask.sum())
        correct = int(((all_preds == cls) & mask).sum())
        results['per_class_recall'][class_names_short[cls]] = {
            'correct': correct, 'total': total, 
            'recall': round(correct / total, 4) if total > 0 else 0
        }
    
    with open(checkpoint_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {checkpoint_dir / 'test_results.json'}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Exp 10: SupCon pre-training + hierarchical fine-tuning')
    
    # Phase selection
    parser.add_argument('--phase', type=str, default='both', 
                        choices=['pretrain', 'finetune', 'both', 'evaluate'],
                        help='Which phase to run')
    
    # SupCon pre-training params
    parser.add_argument('--pretrain_epochs', type=int, default=30,
                        help='Epochs for SupCon pre-training')
    parser.add_argument('--supcon_batch_size', type=int, default=64,
                        help='Batch size for SupCon (larger = more contrastive pairs)')
    parser.add_argument('--supcon_lr', type=float, default=5e-4,
                        help='Learning rate for SupCon pre-training')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    
    # Fine-tuning params
    parser.add_argument('--finetune_epochs', type=int, default=50,
                        help='Epochs for fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning heads')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--oversample', type=int, default=15,
                        help='Broken ooid oversample factor')
    
    # General
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/exp10_supcon')
    
    args = parser.parse_args()
    
    if args.phase == 'pretrain':
        pretrain_supcon(args)
    elif args.phase == 'finetune':
        finetune_hierarchical(args)
    elif args.phase == 'both':
        backbone_path = pretrain_supcon(args)
        finetune_hierarchical(args, backbone_path)
    elif args.phase == 'evaluate':
        evaluate_test(args)


if __name__ == '__main__':
    main()
