"""
Experiment 9: Test-Time Augmentation (TTA)

Apply augmentations at inference time and average predictions.
This can improve accuracy without retraining.

Augmentations applied:
- Original image
- Horizontal flip
- Vertical flip
- 90-degree rotations (90, 180, 270)

For each test sample: 6 augmented versions -> average logits -> final prediction
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.models.hierarchical_model import HierarchicalGrainClassifier


def apply_tta(image_tensor):
    """Generate TTA versions of an image tensor (C, H, W)."""
    augmented = [image_tensor]  # Original
    
    # Horizontal flip
    augmented.append(torch.flip(image_tensor, dims=[2]))
    
    # Vertical flip
    augmented.append(torch.flip(image_tensor, dims=[1]))
    
    # 90, 180, 270 degree rotations
    augmented.append(torch.rot90(image_tensor, k=1, dims=[1, 2]))
    augmented.append(torch.rot90(image_tensor, k=2, dims=[1, 2]))
    augmented.append(torch.rot90(image_tensor, k=3, dims=[1, 2]))
    
    return torch.stack(augmented)  # (6, C, H, W)


def predict_with_tta(model, image, device):
    """Predict with TTA, averaging logits across augmentations."""
    augmented = apply_tta(image).to(device)  # (6, C, H, W)
    
    with torch.no_grad():
        logits = model(augmented)
        
        # Average logits across augmentations
        avg_logits = {
            'stage1': logits['stage1'].mean().unsqueeze(0),
            'stage2': logits['stage2'].mean().unsqueeze(0),
            'stage3': logits['stage3'].mean().unsqueeze(0),
        }
    
    return avg_logits


def hierarchical_predict(logits, t1=0.5, t2=0.5, t3=0.5):
    """Convert logits to final class prediction."""
    p1 = torch.sigmoid(logits['stage1']).item()
    
    if p1 > t1:
        return 0  # Peloid
    
    p2 = torch.sigmoid(logits['stage2']).item()
    if p2 < t2:
        return 3  # Intraclast
    
    p3 = torch.sigmoid(logits['stage3']).item()
    if p3 > t3:
        return 1  # Ooid
    else:
        return 2  # Broken ooid


def evaluate_dataset(model, dataset, device, use_tta=True):
    """Evaluate model on dataset with or without TTA."""
    model.eval()
    
    preds = []
    true_labels = []
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    for i in range(len(dataset)):
        image, labels, metadata = dataset[i]
        true_labels.append(label_map[metadata['label']])
        
        if use_tta:
            logits = predict_with_tta(model, image, device)
        else:
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
                logits = {k: v[0:1] for k, v in logits.items()}
        
        pred = hierarchical_predict(logits)
        preds.append(pred)
    
    return np.array(preds), np.array(true_labels)


def print_results(name, y_true, y_pred, class_names):
    """Print evaluation results."""
    acc = (y_true == y_pred).mean()
    print(f"\n{name}")
    print("="*60)
    print(f"Overall Accuracy: {acc:.4f} ({(y_true == y_pred).sum()}/{len(y_true)})")
    
    print("\nPer-class accuracy:")
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {cls:12s}: {(y_pred[mask] == i).sum()}/{mask.sum()} ({cls_acc:.1%})")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"{'':>12} " + " ".join(f"{c[:6]:>6}" for c in class_names))
    for i, cls in enumerate(class_names):
        print(f"{cls:>12} " + " ".join(f"{cm[i,j]:>6}" for j in range(4)))
    
    return acc


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = HierarchicalGrainClassifier(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both direct state_dict and wrapped checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load test dataset (default transform includes normalization)
    test_dataset = GrainDatasetNew(split='test')
    
    # Evaluate without TTA
    print("\nEvaluating without TTA...")
    preds_no_tta, labels = evaluate_dataset(model, test_dataset, device, use_tta=False)
    acc_no_tta = print_results("Without TTA", labels, preds_no_tta, class_names)
    
    # Evaluate with TTA
    print("\nEvaluating with TTA (6 augmentations)...")
    preds_tta, _ = evaluate_dataset(model, test_dataset, device, use_tta=True)
    acc_tta = print_results("With TTA", labels, preds_tta, class_names)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Without TTA: {acc_no_tta:.4f}")
    print(f"With TTA:    {acc_tta:.4f}")
    print(f"Improvement: {acc_tta - acc_no_tta:+.4f}")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'without_tta': {'accuracy': float(acc_no_tta), 'predictions': preds_no_tta.tolist()},
        'with_tta': {'accuracy': float(acc_tta), 'predictions': preds_tta.tolist()},
        'true_labels': labels.tolist()
    }
    
    output_path = Path('results/exp9_tta_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-Time Augmentation evaluation')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/new_split_v2/best_model_overall_acc_0.9337.pth')
    args = parser.parse_args()
    main(args)

