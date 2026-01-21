"""
Experiment 6: Confidence-Based Routing Ensemble

Uses each model's strengths by routing predictions to the specialist model:
- Exp 2 for ooid predictions (best ooid recall: 88.10%)
- Exp 3 for intraclast predictions (best intraclast recall: 61.90%)
- Baseline for broken ooid and fallback (best broken recall: 83.33%)

Strategy:
1. Get predictions + confidences from all 3 models
2. Identify predicted class with highest confidence from each model
3. Route to specialist model based on predicted class:
   - If highest confidence predicts Ooid → use Exp 2 prediction
   - If highest confidence predicts Intraclast → use Exp 3 prediction
   - If highest confidence predicts Broken or low confidence → use Baseline
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, auc,
    classification_report, confusion_matrix
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.data.dataset_new import GrainDatasetNew


def hierarchical_to_class(stage1, stage2, stage3):
    """Convert hierarchical predictions to class index."""
    if stage1 == 1:  # Peloid
        return 0
    else:  # Non-peloid (stage1 == 0)
        if stage2 == 1:  # Ooid-like
            if stage3 == 1:
                return 1  # Whole ooid
            else:
                return 2  # Broken ooid
        else:  # Non-ooid-like (stage2 == 0)
            return 3  # Intraclast


def load_model(checkpoint_path):
    """Load a trained ResNet-18 hierarchical model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle wrapped checkpoints
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model
    model = HierarchicalGrainClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device


def get_model_predictions(model, loader, device):
    """
    Get predictions and confidences from a model.
    
    Returns:
        predictions: (N,) array of predicted class indices
        confidences: (N,) array of prediction confidences (max probability)
        class_probs: (N, 4) array of class probabilities
    """
    all_predictions = []
    all_confidences = []
    all_class_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get stage probabilities
            stage1_probs = torch.sigmoid(outputs['stage1']).squeeze()
            stage2_probs = torch.sigmoid(outputs['stage2']).squeeze()
            stage3_probs = torch.sigmoid(outputs['stage3']).squeeze()
            
            # Handle single-sample batches
            if stage1_probs.dim() == 0:
                stage1_probs = stage1_probs.unsqueeze(0)
                stage2_probs = stage2_probs.unsqueeze(0)
                stage3_probs = stage3_probs.unsqueeze(0)
            
            # Convert to class probabilities
            for i in range(len(images)):
                s1_prob = stage1_probs[i].item()
                s2_prob = stage2_probs[i].item()
                s3_prob = stage3_probs[i].item()
                
                # Compute class probabilities
                p_peloid = 1 - s1_prob
                p_ooid = s1_prob * (1 - s2_prob) * (1 - s3_prob)
                p_broken = s1_prob * (1 - s2_prob) * s3_prob
                p_intraclast = s1_prob * s2_prob
                
                # Normalize
                total = p_peloid + p_ooid + p_broken + p_intraclast
                if total > 0:
                    probs = np.array([p_peloid / total, p_ooid / total, 
                                     p_broken / total, p_intraclast / total])
                else:
                    probs = np.array([0.25, 0.25, 0.25, 0.25])
                
                all_class_probs.append(probs)
                
                # Predicted class and confidence
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]
                
                all_predictions.append(pred_class)
                all_confidences.append(confidence)
            
            # Convert labels
            for i in range(len(images)):
                s1 = labels[i]['stage1'] if isinstance(labels, list) else labels['stage1'][i].item()
                s2 = labels[i]['stage2'] if isinstance(labels, list) else labels['stage2'][i].item()
                s3 = labels[i]['stage3'] if isinstance(labels, list) else labels['stage3'][i].item()
                
                class_idx = hierarchical_to_class(s1, s2, s3)
                all_labels.append(class_idx)
    
    return (np.array(all_predictions), 
            np.array(all_confidences),
            np.array(all_class_probs),
            np.array(all_labels))


def route_predictions(baseline_preds, baseline_confs, baseline_probs,
                      exp2_preds, exp2_confs, exp2_probs,
                      exp3_preds, exp3_confs, exp3_probs,
                      confidence_threshold=0.7):
    """
    Simple averaging of probabilities from all three models.
    Each model gets equal weight (1/3).
    """
    n_samples = len(baseline_preds)
    
    # Simple average of probabilities
    combined_probs = (baseline_probs + exp2_probs + exp3_probs) / 3.0
    
    # Predict class with highest combined probability
    final_predictions = np.argmax(combined_probs, axis=1)
    
    # Track which model had highest confidence for predicted class
    routing_decisions = []
    CLASS_NAMES = ['Peloid', 'Ooid', 'Broken', 'Intraclast']
    
    for i in range(n_samples):
        pred_class = final_predictions[i]
        
        # Which model was most confident about this prediction?
        confidences = {
            'baseline': baseline_probs[i, pred_class],
            'exp2': exp2_probs[i, pred_class],
            'exp3': exp3_probs[i, pred_class]
        }
        
        max_model = max(confidences, key=confidences.get)
        routing_decisions.append(f'{max_model}_{CLASS_NAMES[pred_class]}')
    
    return final_predictions, routing_decisions, combined_probs


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
        
        # PR-AUC
        if len(np.unique(y_true_binary)) > 1:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true_binary, y_probs[:, i])
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


def print_results(y_true, y_pred, y_probs, routing_decisions, split_name, class_names):
    """Print evaluation results."""
    accuracy = accuracy_score(y_true, y_pred)
    
    print()
    print(f"{split_name.upper()} SET RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Per-class metrics
    metrics = compute_metrics_per_class(y_true, y_pred, y_probs, class_names)
    
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name in class_names:
        m = metrics[class_name]
        print(f"{class_name:<15} {m['precision']:>11.4f} {m['recall']:>11.4f} "
              f"{m['pr_auc']:>11.4f} {m['support']:>9d}")
    
    print()
    print("Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred)
    
    # Header
    print(f"{'':>15}", end='')
    for name in class_names:
        print(f"{name:>12}", end='')
    print()
    print("-" * 80)
    
    # Rows
    for i, name in enumerate(class_names):
        print(f"{name:>15}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12d}", end='')
        print()
    
    print()
    
    # Routing statistics
    print("Routing Statistics:")
    print("-" * 80)
    routing_counts = defaultdict(int)
    for decision in routing_decisions:
        routing_counts[decision] += 1
    
    for decision, count in sorted(routing_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(routing_decisions)
        print(f"  {decision:30s}: {count:4d} ({pct:5.2f}%)")
    
    print()
    
    return {'accuracy': accuracy, 'per_class_metrics': metrics, 
            'confusion_matrix': cm.tolist(), 'routing_stats': dict(routing_counts)}


def main():
    print("=" * 80)
    print("Experiment 6: Confidence-Based Routing Ensemble")
    print("=" * 80)
    
    # Paths to checkpoints
    checkpoint_paths = {
        'baseline': project_root / 'checkpoints/new_split_v2/best_model_overall_acc_0.9337.pth',
        'exp2': project_root / 'checkpoints/exp2_augmented/best_model_overall_acc_0.9280.pth',
        'exp3': project_root / 'checkpoints/exp3_weighted/best_model_overall_acc_0.9413.pth'
    }
    
    # Check if all checkpoints exist
    for name, path in checkpoint_paths.items():
        if not path.exists():
            print(f"❌ Checkpoint not found: {path}")
            return
    
    print("✓ All checkpoints found")
    print()
    
    # Load models
    print("Loading models...")
    print("-" * 80)
    models = {}
    for name, path in checkpoint_paths.items():
        print(f"  Loading {name}...")
        models[name], device = load_model(path)
    
    print(f"\n✓ All models loaded")
    print(f"Device: {device}")
    print()
    
    # Load datasets
    val_dataset = GrainDatasetNew(
        split='val',
        patches_dir='data/processed/patches',
        use_default_transforms=False
    )
    
    test_dataset = GrainDatasetNew(
        split='test',
        patches_dir='data/processed/patches',
        use_default_transforms=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    class_names = ['Peloid', 'Ooid', 'Broken', 'Intraclast']
    
    # Evaluate on validation set (tune confidence threshold)
    print("=" * 80)
    print("VALIDATION SET - Finding Best Threshold")
    print("=" * 80)
    
    print("\nExtracting predictions from all models on validation set...")
    baseline_preds, baseline_confs, baseline_probs, val_labels = \
        get_model_predictions(models['baseline'], val_loader, device)
    exp2_preds, exp2_confs, exp2_probs, _ = \
        get_model_predictions(models['exp2'], val_loader, device)
    exp3_preds, exp3_confs, exp3_probs, _ = \
        get_model_predictions(models['exp3'], val_loader, device)
    
    print("✓ Predictions extracted")
    print()
    
    # Use simple averaging approach
    print("Using simple probability averaging (equal weights for all models)...")
    print("-" * 80)
    
    # Use fixed threshold (not used in averaging but kept for API compatibility)
    best_threshold = 0.7
    
    routed_preds, routing_decisions, combined_probs = route_predictions(
        baseline_preds, baseline_confs, baseline_probs,
        exp2_preds, exp2_confs, exp2_probs,
        exp3_preds, exp3_confs, exp3_probs,
        confidence_threshold=best_threshold
    )
    
    val_acc = accuracy_score(val_labels, routed_preds)
    print(f"Validation accuracy with averaged probabilities: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print()
    
    print(f"✓ Using equal-weight probability averaging")
    print(f"  Each model contributes 1/3 to final probability distribution")
    print()
    
    # Evaluate on validation set with weighted combination
    routed_preds, routing_decisions, combined_probs = route_predictions(
        baseline_preds, baseline_confs, baseline_probs,
        exp2_preds, exp2_confs, exp2_probs,
        exp3_preds, exp3_confs, exp3_probs,
        confidence_threshold=best_threshold
    )
    
    # Use combined probs for PR-AUC computation
    val_results = print_results(val_labels, routed_preds, combined_probs, 
                                 routing_decisions, 'validation', class_names)
    
    print("=" * 80)
    print("TEST SET - Final Evaluation")
    print("=" * 80)
    
    print(f"\nUsing simple probability averaging (equal weights)")
    print("\nExtracting predictions from all models on test set...")
    
    baseline_preds, baseline_confs, baseline_probs, test_labels = \
        get_model_predictions(models['baseline'], test_loader, device)
    exp2_preds, exp2_confs, exp2_probs, _ = \
        get_model_predictions(models['exp2'], test_loader, device)
    exp3_preds, exp3_confs, exp3_probs, _ = \
        get_model_predictions(models['exp3'], test_loader, device)
    
    print("✓ Predictions extracted")
    
    routed_preds, routing_decisions, combined_probs = route_predictions(
        baseline_preds, baseline_confs, baseline_probs,
        exp2_preds, exp2_confs, exp2_probs,
        exp3_preds, exp3_confs, exp3_probs,
        confidence_threshold=best_threshold
    )
    
    test_results = print_results(test_labels, routed_preds, combined_probs,
                                  routing_decisions, 'test', class_names)
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / 'exp6_confidence_routing.json'
    with open(output_path, 'w') as f:
        json.dump({
            'confidence_threshold': best_threshold,
            'validation': val_results,
            'test': test_results,
            'checkpoint_paths': {k: str(v) for k, v in checkpoint_paths.items()}
        }, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    print()
    print("=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
