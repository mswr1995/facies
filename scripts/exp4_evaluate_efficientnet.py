"""
Evaluate Experiment 4: EfficientNet + Attention model.

Loads best checkpoint and computes detailed metrics on validation and test sets.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_model import EfficientNetHierarchicalClassifier
from src.data.dataset_new import GrainDatasetNew


def compute_metrics_per_class(y_true, y_pred, y_probs, class_names):
    """Compute precision, recall, PR-AUC for each class."""
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Binary: class i vs all others
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Precision and recall
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # PR-AUC (use probabilities for this class)
        if class_name in y_probs and len(np.unique(y_true_binary)) > 1:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true_binary, y_probs[class_name])
            pr_auc = auc(rec_curve, prec_curve)
        else:
            pr_auc = 0.0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'pr_auc': pr_auc,
            'support': int(np.sum(y_true_binary))
        }
    
    return metrics


def evaluate_model(model, loader, device, class_names):
    """Evaluate model on a dataset."""
    model.eval()
    
    # Map hierarchical labels to class indices
    def hierarchical_to_class(labels_dict):
        """Convert hierarchical labels to class index."""
        stage1 = labels_dict['stage1'].item()
        stage2 = labels_dict['stage2'].item()
        stage3 = labels_dict['stage3'].item()
        
        if stage1 == 1:  # Peloid
            return 0
        else:  # Ooid-like
            if stage2 == 1:  # Ooid
                if stage3 == 1:  # Whole
                    return 1
                else:  # Broken
                    return 2
            else:  # Intraclast
                return 3
    
    all_labels = []
    all_preds = []
    all_probs = defaultdict(list)
    
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            
            # Get predictions
            preds, probs = model.predict(images, thresholds=(0.5, 0.5, 0.5))
            
            # Convert hierarchical labels to class indices
            batch_labels = []
            for i in range(len(images)):
                label_dict = {k: v[i] for k, v in labels.items()}
                class_idx = hierarchical_to_class(label_dict)
                batch_labels.append(class_idx)
            
            all_labels.extend(batch_labels)
            all_preds.extend(preds.cpu().numpy())
            
            # Store probabilities for each class
            batch_size = images.size(0)
            for i in range(batch_size):
                # Compute probability for each possible class
                for class_idx, class_name in enumerate(class_names):
                    if class_idx == 0:  # Peloid
                        class_prob = probs['stage1'][i].item()
                    elif class_idx == 1:  # Ooid
                        class_prob = (1 - probs['stage1'][i].item()) * probs['stage2'][i].item() * probs['stage3'][i].item()
                    elif class_idx == 2:  # Broken
                        class_prob = (1 - probs['stage1'][i].item()) * probs['stage2'][i].item() * (1 - probs['stage3'][i].item())
                    else:  # Intraclast
                        class_prob = (1 - probs['stage1'][i].item()) * (1 - probs['stage2'][i].item())
                    
                    all_probs[class_name].append(class_prob)
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = {k: np.array(v) for k, v in all_probs.items()}
    
    # Compute metrics
    metrics = compute_metrics_per_class(y_true, y_pred, y_probs, class_names)
    
    return metrics


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_dir = Path('checkpoints/exp4_efficientnet')
    checkpoint_files = sorted(checkpoint_dir.glob('best_model_*.pth'))
    
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    best_checkpoint = checkpoint_files[-1]
    print(f"Loading best checkpoint: {best_checkpoint.name}")
    
    model = EfficientNetHierarchicalClassifier(pretrained=False, dropout=0.3)
    checkpoint = torch.load(best_checkpoint, map_location=device)
    
    # Handle both direct state_dict and wrapped checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("✅ Model loaded\n")
    
    # Load datasets
    val_dataset = GrainDatasetNew(split='val')
    test_dataset = GrainDatasetNew(split='test')
    
    # Class names
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    print()
    
    # Create loaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluate
    print("=" * 90)
    print("EXPERIMENT 4: EFFICIENTNET + ATTENTION EVALUATION")
    print("=" * 90)
    
    print("\nEvaluating validation set...")
    val_metrics = evaluate_model(model, val_loader, device, class_names)
    
    print("\nVALIDATION SET:\n")
    print(f"{'Class':<15s} {'Precision':<12s} {'Recall':<12s} {'PR-AUC':<12s} {'Support':<8s}")
    print("-" * 90)
    for class_name in class_names:
        m = val_metrics[class_name]
        print(f"{class_name:<15s} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['pr_auc']:<12.4f} {m['support']:<8d}")
    
    print("\nEvaluating test set...")
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    
    print("\nTEST SET:\n")
    print(f"{'Class':<15s} {'Precision':<12s} {'Recall':<12s} {'PR-AUC':<12s} {'Support':<8s}")
    print("-" * 90)
    for class_name in class_names:
        m = test_metrics[class_name]
        print(f"{class_name:<15s} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['pr_auc']:<12.4f} {m['support']:<8d}")
    
    # Save results
    results = {
        'validation': val_metrics,
        'test': test_metrics,
        'checkpoint': str(best_checkpoint)
    }
    
    results_path = Path('results/exp4_efficientnet.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()
