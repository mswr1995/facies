"""
Seven-class hierarchical ResNet-18 with no-TTA and six-orientation TTA evaluation.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_all_classes import ALL_CLASS_NAMES, AllClassGrainDataset
from src.models.focal_loss import FocalLoss
from src.models.hierarchical_model_all_classes import SevenClassHierarchicalClassifier


def make_hierarchical_labels(targets, device):
    labels = {
        'stage1': torch.full_like(targets, -1),
        'stage2': torch.full_like(targets, -1),
        'stage3': torch.full_like(targets, -1),
        'stage4': torch.full_like(targets, -1),
        'stage5': torch.full_like(targets, -1),
        'stage6': torch.full_like(targets, -1),
    }

    labels['stage1'] = (targets == 0).long()

    non_peloid = targets != 0
    carbonate = (targets == 1) | (targets == 2) | (targets == 3)
    ooid_like = (targets == 1) | (targets == 2)
    other = (targets == 4) | (targets == 5) | (targets == 6)
    fossil = (targets == 4) | (targets == 5)

    labels['stage2'][non_peloid] = carbonate[non_peloid].long()
    labels['stage3'][carbonate] = ooid_like[carbonate].long()
    labels['stage4'][ooid_like] = (targets[ooid_like] == 1).long()
    labels['stage5'][other] = fossil[other].long()
    labels['stage6'][fossil] = (targets[fossil] == 4).long()

    return {k: v.to(device) for k, v in labels.items()}


def make_train_loader(dataset, batch_size, samples_per_epoch):
    counts = Counter(sample['label'] for sample in dataset.samples)
    sample_weights = [1.0 / counts[sample['label']] for sample in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=samples_per_epoch or len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def train_epoch(model, loader, optimizer, loss_fns, device, active_stages):
    model.train()
    total_loss = 0.0

    for images, targets, _ in tqdm(loader, desc='  train', leave=False):
        images = images.to(device)
        targets = targets.to(device)
        labels = make_hierarchical_labels(targets, device)

        optimizer.zero_grad()
        logits = model(images)
        loss = torch.tensor(0.0, device=device)

        for stage in active_stages:
            mask = labels[stage] != -1
            if mask.sum() > 0:
                loss = loss + loss_fns[stage](logits[stage][mask], labels[stage][mask].float())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def apply_tta(image_tensor):
    return torch.stack([
        image_tensor,
        torch.flip(image_tensor, dims=[2]),
        torch.flip(image_tensor, dims=[1]),
        torch.rot90(image_tensor, k=1, dims=[1, 2]),
        torch.rot90(image_tensor, k=2, dims=[1, 2]),
        torch.rot90(image_tensor, k=3, dims=[1, 2]),
    ])


def evaluate(model, dataset, device, use_tta=False):
    model.eval()
    preds = []
    trues = []
    scores = []

    with torch.no_grad():
        if use_tta:
            for idx in tqdm(range(len(dataset)), desc='  eval tta', leave=False):
                image, target, _ = dataset[idx]
                logits = model(apply_tta(image).to(device))
                avg_logits = {k: v.mean(dim=0, keepdim=True) for k, v in logits.items()}
                pred = model.get_predictions(avg_logits).item()
                score = model.get_class_scores(avg_logits).squeeze(0).cpu().numpy()
                preds.append(pred)
                trues.append(int(target.item()))
                scores.append(score)
        else:
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
            for images, targets, _ in loader:
                logits = model(images.to(device))
                preds.extend(model.get_predictions(logits).cpu().numpy().tolist())
                trues.extend(targets.numpy().tolist())
                scores.extend(model.get_class_scores(logits).cpu().numpy().tolist())

    return np.array(trues), np.array(preds), np.array(scores)


def summarize(y_true, y_pred):
    per_class = {}
    for idx, label in enumerate(ALL_CLASS_NAMES):
        mask = y_true == idx
        support = int(mask.sum())
        correct = int((y_pred[mask] == idx).sum())
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


def set_trainable(model, active_groups):
    for param in model.parameters():
        param.requires_grad = False

    if 'backbone' in active_groups:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for stage in range(1, 7):
        if f'stage{stage}' in active_groups:
            head = getattr(model, f'head_stage{stage}')
            for param in head.parameters():
                param.requires_grad = True


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = AllClassGrainDataset(split='train', processed_dir=args.processed_dir)
    val_ds = AllClassGrainDataset(split='val', processed_dir=args.processed_dir)
    test_ds = AllClassGrainDataset(split='test', processed_dir=args.processed_dir)

    train_loader = make_train_loader(train_ds, args.batch_size, args.samples_per_epoch)
    model = SevenClassHierarchicalClassifier(pretrained=not args.no_pretrained).to(device)
    loss_fns = {
        'stage1': FocalLoss(alpha=0.25, gamma=2.0),
        'stage2': FocalLoss(alpha=0.25, gamma=2.0),
        'stage3': FocalLoss(alpha=0.50, gamma=2.0),
        'stage4': FocalLoss(alpha=0.25, gamma=2.0),
        'stage5': FocalLoss(alpha=0.50, gamma=2.0),
        'stage6': FocalLoss(alpha=0.25, gamma=2.0),
    }

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_ba = -1.0
    history = []

    phases = [
        (1, args.phase1_epochs, ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'], ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'], 1e-3),
        (2, args.phase2_epochs, ['stage1'], ['backbone', 'stage1'], 1e-4),
        (3, args.phase3_epochs, ['stage2', 'stage3', 'stage4', 'stage5', 'stage6'], ['backbone', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'], 1e-4),
        (4, args.phase4_epochs, ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'], ['backbone', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'], 1e-5),
    ]

    for phase_num, epochs, active_stages, active_groups, lr in phases:
        print(f"\nPHASE {phase_num}: stages={active_stages}, lr={lr}")
        set_trainable(model, active_groups)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, loss_fns, device, active_stages)
            y_val, p_val, _ = evaluate(model, val_ds, device, use_tta=False)
            val_ba = balanced_accuracy_score(y_val, p_val)
            val_oa = np.mean(y_val == p_val)
            print(f"  Phase {phase_num} Epoch {epoch:2d}: loss={loss:.4f}  val_ba={val_ba:.4f}  val_oa={val_oa:.4f}")
            history.append({
                'phase': phase_num,
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

    y_no_tta, p_no_tta, scores_no_tta = evaluate(model, test_ds, device, use_tta=False)
    y_tta, p_tta, scores_tta = evaluate(model, test_ds, device, use_tta=True)
    no_tta_summary = summarize(y_no_tta, p_no_tta)
    tta_summary = summarize(y_tta, p_tta)

    print_summary("SEVEN-CLASS HIERARCHICAL TEST WITHOUT TTA", no_tta_summary)
    print_summary("SEVEN-CLASS HIERARCHICAL TEST WITH SIX-ORIENTATION TTA", tta_summary)
    print("\nTTA delta:")
    print(f"  Balanced Accuracy: {tta_summary['balanced_accuracy'] - no_tta_summary['balanced_accuracy']:+.4f}")
    print(f"  Overall Accuracy : {tta_summary['overall_accuracy'] - no_tta_summary['overall_accuracy']:+.4f}")

    results = {
        'class_names': ALL_CLASS_NAMES,
        'checkpoint': str(ckpt_dir / 'best_model.pth'),
        'best_val_ba': float(best_val_ba),
        'history': history,
        'test_without_tta': no_tta_summary,
        'test_with_tta': tta_summary,
        'predictions_without_tta': p_no_tta.tolist(),
        'predictions_with_tta': p_tta.tolist(),
        'true_labels': y_tta.tolist(),
        'scores_without_tta': scores_no_tta.tolist(),
        'scores_with_tta': scores_tta.tolist(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='data/processed_all_classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/all_classes_hierarchical')
    parser.add_argument('--output', type=str, default='results/all_classes_hierarchical_tta_results.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--samples_per_epoch', type=int, default=None)
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=15)
    parser.add_argument('--phase3_epochs', type=int, default=15)
    parser.add_argument('--phase4_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', action='store_true')
    args = parser.parse_args()
    main(args)
