"""
Experiment 7: Staged Training

Train each classification head sequentially instead of jointly.
This prevents Stage 1 (peloid, 87% of data) from dominating backbone gradients.

Approach:
  Phase 1: Freeze backbone, train all heads (10 epochs) - learn head weights
  Phase 2: Unfreeze, train Stage 1 only (15 epochs) - tune backbone for peloid
  Phase 3: Freeze Stage 1, train Stage 2+3 (15 epochs) - tune for minority classes
  Phase 4: Fine-tune all together with low LR (10 epochs)
"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss


def create_dataloaders(batch_size=32):
    """Create train/val dataloaders."""
    train_dataset = GrainDatasetNew(split='train')
    val_dataset = GrainDatasetNew(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, loss_fns, device, active_stages):
    """Train one epoch with only active stages contributing to loss."""
    model.train()
    total_loss = 0
    
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        for k in labels:
            labels[k] = labels[k].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = torch.tensor(0.0, device=device)
        
        if 'stage1' in active_stages:
            loss += loss_fns['stage1'](logits['stage1'], labels['stage1'].float())
        
        if 'stage2' in active_stages:
            mask = labels['stage2'] != -1
            if mask.sum() > 0:
                loss += loss_fns['stage2'](logits['stage2'][mask], labels['stage2'][mask].float())
        
        if 'stage3' in active_stages:
            mask = labels['stage3'] != -1
            if mask.sum() > 0:
                loss += loss_fns['stage3'](logits['stage3'][mask], labels['stage3'][mask].float())
        
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            preds = model.get_predictions(logits)
            
            true_labels = torch.tensor([label_map[l] for l in metadata['label']])
            correct += (preds.cpu() == true_labels).sum().item()
            total += len(true_labels)
    
    return correct / total


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Setup
    train_loader, val_loader = create_dataloaders(args.batch_size)
    model = HierarchicalGrainClassifier(pretrained=True, freeze_backbone=False).to(device)
    
    loss_fns = {
        'stage1': FocalLoss(alpha=0.25, gamma=2.0),
        'stage2': FocalLoss(alpha=0.5, gamma=2.0),
        'stage3': FocalLoss(alpha=0.75, gamma=2.0),
    }
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0.0
    history = []
    
    # Phase 1: Freeze backbone, train heads only
    print("\n" + "="*60)
    print("PHASE 1: Train heads only (backbone frozen)")
    print("="*60)
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-3, weight_decay=1e-4
    )
    
    for epoch in range(1, args.phase1_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fns, device, 
                          ['stage1', 'stage2', 'stage3'])
        acc = evaluate(model, val_loader, device)
        print(f"Phase 1 Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'phase': 1, 'epoch': epoch, 'loss': loss, 'val_acc': acc})
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
    
    # Phase 2: Unfreeze backbone, train Stage 1 only
    print("\n" + "="*60)
    print("PHASE 2: Fine-tune backbone with Stage 1 only")
    print("="*60)
    model.unfreeze_backbone()
    
    # Freeze Stage 2 and 3 heads
    for param in model.head_stage2.parameters():
        param.requires_grad = False
    for param in model.head_stage3.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )
    
    for epoch in range(1, args.phase2_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fns, device, ['stage1'])
        acc = evaluate(model, val_loader, device)
        print(f"Phase 2 Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'phase': 2, 'epoch': epoch, 'loss': loss, 'val_acc': acc})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')

    # Phase 3: Freeze Stage 1, train Stage 2+3 (minority classes)
    print("\n" + "="*60)
    print("PHASE 3: Train Stage 2+3 only (Stage 1 frozen)")
    print("="*60)

    # Unfreeze Stage 2 and 3, freeze Stage 1
    for param in model.head_stage1.parameters():
        param.requires_grad = False
    for param in model.head_stage2.parameters():
        param.requires_grad = True
    for param in model.head_stage3.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )

    for epoch in range(1, args.phase3_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fns, device, ['stage2', 'stage3'])
        acc = evaluate(model, val_loader, device)
        print(f"Phase 3 Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'phase': 3, 'epoch': epoch, 'loss': loss, 'val_acc': acc})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')

    # Phase 4: Fine-tune all together with low LR
    print("\n" + "="*60)
    print("PHASE 4: Fine-tune all (low LR)")
    print("="*60)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    for epoch in range(1, args.phase4_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fns, device,
                          ['stage1', 'stage2', 'stage3'])
        acc = evaluate(model, val_loader, device)
        print(f"Phase 4 Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'phase': 4, 'epoch': epoch, 'loss': loss, 'val_acc': acc})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')

    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')

    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_acc}, f, indent=2)

    print("\n" + "="*60)
    print(f"Training complete. Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Staged training for hierarchical classifier')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--phase1_epochs', type=int, default=10, help='Epochs for phase 1 (heads only)')
    parser.add_argument('--phase2_epochs', type=int, default=15, help='Epochs for phase 2 (stage 1 focus)')
    parser.add_argument('--phase3_epochs', type=int, default=15, help='Epochs for phase 3 (stage 2+3 focus)')
    parser.add_argument('--phase4_epochs', type=int, default=10, help='Epochs for phase 4 (fine-tune all)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/exp7_staged')

    args = parser.parse_args()
    main(args)
