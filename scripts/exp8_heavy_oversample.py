"""
Experiment 8: Heavy Oversampling

Increase broken ooid oversampling from 3x (default) to 15x.
With only 17 broken ooid training samples, 3x may not be enough.

Uses StageWiseBatchSampler with broken_ooid_oversample=15.
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.data.samplers import StageWiseBatchSampler
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss


def create_dataloaders(batch_size=32, oversample=15):
    """Create train/val dataloaders with heavy oversampling."""
    train_dataset = GrainDatasetNew(split='train')
    val_dataset = GrainDatasetNew(split='val')
    
    # Use StageWiseBatchSampler with heavy oversampling
    train_sampler = StageWiseBatchSampler(
        train_dataset, 
        batch_size=batch_size,
        broken_ooid_oversample=oversample
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, loss_fns, device):
    """Train one epoch."""
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
    print(f"Broken ooid oversampling: {args.oversample}x")
    
    train_loader, val_loader = create_dataloaders(args.batch_size, args.oversample)
    model = HierarchicalGrainClassifier(pretrained=True, freeze_backbone=False).to(device)
    
    loss_fns = {
        'stage1': FocalLoss(alpha=0.25, gamma=2.0),
        'stage2': FocalLoss(alpha=0.5, gamma=2.0),
        'stage3': FocalLoss(alpha=0.75, gamma=2.0),
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        acc = evaluate(model, val_loader, device)
        scheduler.step(1 - acc)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.4f}")
        history.append({'epoch': epoch, 'loss': loss, 'val_acc': acc})
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
    
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_acc, 'oversample': args.oversample}, f, indent=2)
    
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training with heavy oversampling')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--oversample', type=int, default=15, help='Broken ooid oversample factor')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/exp8_heavy_oversample')
    
    args = parser.parse_args()
    main(args)

