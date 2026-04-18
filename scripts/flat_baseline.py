"""
Flat ResNet-18 baseline: standard 4-class softmax + cross-entropy.
No hierarchical branching, no focal loss, no oversampling.
"""
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew


CLASS_NAMES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


class FlatResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def make_loader(split, batch_size, shuffle):
    ds = GrainDatasetNew(split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0), ds


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(loader, desc='  train', leave=False):
        images = images.to(device)
        targets = labels['stage1'].clone()  # placeholder — we use metadata below
        optimizer.zero_grad()
        logits = model(images)
        # Build flat 4-class target from stage labels
        # stage1=1 → Peloid(0); stage2=0 → Intraclast(3); stage3=1 → Ooid(1); stage3=0 → Broken(2)
        flat_targets = torch.zeros(len(images), dtype=torch.long)
        s1 = labels['stage1']
        s2 = labels['stage2']
        s3 = labels['stage3']
        for i in range(len(images)):
            if s1[i] == 1:
                flat_targets[i] = 0  # Peloid
            elif s2[i] == 0:
                flat_targets[i] = 3  # Intraclast
            elif s3[i] == 1:
                flat_targets[i] = 1  # Ooid
            else:
                flat_targets[i] = 2  # Broken ooid
        flat_targets = flat_targets.to(device)
        loss = criterion(logits, flat_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, _, metadata in loader:
            images = images.to(device)
            logits = model(images)
            p = logits.argmax(dim=1).cpu().tolist()
            preds.extend(p)
            trues.extend([LABEL_MAP[l] for l in metadata['label']])
    ba = balanced_accuracy_score(trues, preds)
    oa = np.mean(np.array(preds) == np.array(trues))
    return ba, oa, np.array(preds), np.array(trues)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_loader, _ = make_loader('train', args.batch_size, shuffle=True)
    val_loader, _   = make_loader('val',   args.batch_size, shuffle=False)
    test_loader, _  = make_loader('test',  args.batch_size, shuffle=False)

    model = FlatResNet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ba = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        ba, oa, _, _ = evaluate(model, val_loader, device)
        scheduler.step(1 - ba)
        print(f'Epoch {epoch:3d}: loss={loss:.4f}  val_ba={ba:.4f}  val_oa={oa:.4f}')
        history.append({'epoch': epoch, 'loss': loss, 'val_ba': ba, 'val_oa': oa})
        if ba > best_ba:
            best_ba = ba
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')

    # Final test evaluation
    model.load_state_dict(torch.load(ckpt_dir / 'best_model.pth', map_location=device))
    test_ba, test_oa, preds, trues = evaluate(model, test_loader, device)
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2, 3])

    print(f'\n{"="*60}')
    print('TEST SET RESULTS')
    print(f'{"="*60}')
    print(f'Overall Accuracy : {test_oa:.4f}  ({int(test_oa*529)}/529)')
    print(f'Balanced Accuracy: {test_ba:.4f}')
    print('\nPer-class recall:')
    for i, cls in enumerate(CLASS_NAMES):
        mask = trues == i
        recall = (preds[mask] == i).sum() / mask.sum()
        print(f'  {cls:15s}: {(preds[mask]==i).sum()}/{mask.sum()} ({recall:.1%})')
    print('\nConfusion Matrix:')
    print(f'{"":>15}', '  '.join(f'{c[:6]:>6}' for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f'{cls:>15}', '  '.join(f'{cm[i,j]:>6}' for j in range(4)))

    results = {
        'test_ba': float(test_ba),
        'test_oa': float(test_oa),
        'confusion_matrix': cm.tolist(),
        'per_class_recall': {
            CLASS_NAMES[i]: float((preds[trues==i]==i).sum()/( trues==i).sum())
            for i in range(4)
        },
        'history': history,
        'best_val_ba': best_ba
    }
    out = Path('results/flat_baseline_results.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--epochs',         type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/flat_baseline')
    args = parser.parse_args()
    main(args)
