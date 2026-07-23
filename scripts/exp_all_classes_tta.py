"""
Seven-class ResNet-18 experiment with no-TTA and six-orientation TTA evaluation.

The existing hierarchical model is four-class only, so this experiment uses a
flat seven-class softmax head while preserving the same masked 96x96 patch
inputs, train augmentations, ImageNet normalization, and TTA orientations.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_all_classes import (
    ALL_CLASS_NAMES,
    ALL_LABEL_MAP,
    AllClassGrainDataset,
)


class FlatResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                self.model = models.resnet18(weights=weights)
            except AttributeError:
                self.model = models.resnet18(pretrained=True)
        else:
            try:
                self.model = models.resnet18(weights=None)
            except TypeError:
                self.model = models.resnet18(pretrained=False)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def make_train_loader(dataset, batch_size, samples_per_epoch):
    counts = Counter(sample['label'] for sample in dataset.samples)
    sample_weights = [1.0 / counts[sample['label']] for sample in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=samples_per_epoch or len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def make_class_weights(dataset, device):
    counts = Counter(sample['label'] for sample in dataset.samples)
    weights = []
    for label in ALL_CLASS_NAMES:
        # Square-root inverse frequency is less explosive for Bivalve's two train samples.
        weights.append(np.sqrt(len(dataset) / (len(ALL_CLASS_NAMES) * counts[label])))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, targets, _ in tqdm(loader, desc='  train', leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def apply_tta(image_tensor):
    augmented = [image_tensor]
    augmented.append(torch.flip(image_tensor, dims=[2]))
    augmented.append(torch.flip(image_tensor, dims=[1]))
    augmented.append(torch.rot90(image_tensor, k=1, dims=[1, 2]))
    augmented.append(torch.rot90(image_tensor, k=2, dims=[1, 2]))
    augmented.append(torch.rot90(image_tensor, k=3, dims=[1, 2]))
    return torch.stack(augmented)


def evaluate(model, dataset, device, use_tta=False):
    model.eval()
    preds = []
    trues = []
    probs = []

    with torch.no_grad():
        if use_tta:
            iterator = range(len(dataset))
            for idx in tqdm(iterator, desc='  eval tta', leave=False):
                image, target, _ = dataset[idx]
                images = apply_tta(image).to(device)
                logits = model(images).mean(dim=0, keepdim=True)
                prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                probs.append(prob)
                preds.append(int(prob.argmax()))
                trues.append(int(target.item()))
        else:
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
            for images, targets, _ in loader:
                logits = model(images.to(device))
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs.extend(batch_probs)
                preds.extend(batch_probs.argmax(axis=1).tolist())
                trues.extend(targets.numpy().tolist())

    return np.array(trues), np.array(preds), np.array(probs)


def summarize(y_true, y_pred):
    per_class = {}
    for idx, label in enumerate(ALL_CLASS_NAMES):
        mask = y_true == idx
        correct = int((y_pred[mask] == idx).sum())
        support = int(mask.sum())
        per_class[label] = {
            'support': support,
            'correct': correct,
            'recall': float(correct / support) if support else 0.0,
            'precision': float(precision_score(y_true == idx, y_pred == idx, zero_division=0)),
            'f1': float(f1_score(y_true == idx, y_pred == idx, zero_division=0)),
        }

    return {
        'overall_accuracy': float(np.mean(y_true == y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(ALL_CLASS_NAMES))),
        ).tolist(),
    }


def print_summary(name, summary):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print(f"Overall Accuracy : {summary['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"Macro F1         : {summary['macro_f1']:.4f}")
    print("\nPer-class recall:")
    for label in ALL_CLASS_NAMES:
        item = summary['per_class'][label]
        print(f"  {label:15s}: {item['correct']:3d}/{item['support']:<3d} ({item['recall']:.1%})")

    cm = np.array(summary['confusion_matrix'])
    print("\nConfusion Matrix:")
    print(f"{'':>15} " + " ".join(f"{name[:7]:>7}" for name in ALL_CLASS_NAMES))
    for i, label in enumerate(ALL_CLASS_NAMES):
        print(f"{label:>15} " + " ".join(f"{cm[i, j]:>7}" for j in range(len(ALL_CLASS_NAMES))))


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = AllClassGrainDataset(split='train', processed_dir=args.processed_dir)
    val_ds = AllClassGrainDataset(split='val', processed_dir=args.processed_dir)
    test_ds = AllClassGrainDataset(split='test', processed_dir=args.processed_dir)

    train_loader = make_train_loader(train_ds, args.batch_size, args.samples_per_epoch)
    model = FlatResNet18(num_classes=len(ALL_CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_ds, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_ba = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_val, p_val, _ = evaluate(model, val_ds, device, use_tta=False)
        val_ba = balanced_accuracy_score(y_val, p_val)
        val_oa = np.mean(y_val == p_val)
        scheduler.step(1.0 - val_ba)

        print(f"Epoch {epoch:3d}: loss={loss:.4f}  val_ba={val_ba:.4f}  val_oa={val_oa:.4f}")
        history.append({
            'epoch': epoch,
            'loss': float(loss),
            'val_ba': float(val_ba),
            'val_oa': float(val_oa),
        })

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')

    torch.save(model.state_dict(), ckpt_dir / 'final_model.pth')
    model.load_state_dict(torch.load(ckpt_dir / 'best_model.pth', map_location=device))

    y_no_tta, p_no_tta, probs_no_tta = evaluate(model, test_ds, device, use_tta=False)
    y_tta, p_tta, probs_tta = evaluate(model, test_ds, device, use_tta=True)

    no_tta_summary = summarize(y_no_tta, p_no_tta)
    tta_summary = summarize(y_tta, p_tta)

    print_summary("TEST WITHOUT TTA", no_tta_summary)
    print_summary("TEST WITH SIX-ORIENTATION TTA", tta_summary)
    print("\nTTA delta:")
    print(f"  Balanced Accuracy: {tta_summary['balanced_accuracy'] - no_tta_summary['balanced_accuracy']:+.4f}")
    print(f"  Overall Accuracy : {tta_summary['overall_accuracy'] - no_tta_summary['overall_accuracy']:+.4f}")

    results = {
        'class_names': ALL_CLASS_NAMES,
        'label_map': ALL_LABEL_MAP,
        'checkpoint': str(ckpt_dir / 'best_model.pth'),
        'best_val_ba': float(best_val_ba),
        'history': history,
        'test_without_tta': no_tta_summary,
        'test_with_tta': tta_summary,
        'predictions_without_tta': p_no_tta.tolist(),
        'predictions_with_tta': p_tta.tolist(),
        'true_labels': y_tta.tolist(),
        'probabilities_without_tta': probs_no_tta.tolist(),
        'probabilities_with_tta': probs_tta.tolist(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='data/processed_all_classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/all_classes_resnet18')
    parser.add_argument('--output', type=str, default='results/all_classes_tta_results.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--samples_per_epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', action='store_true')
    args = parser.parse_args()
    main(args)
