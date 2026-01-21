"""
Experiment 5: Evaluate Meta-Classifier Ensemble

Evaluates the trained XGBoost meta-classifier on validation and test sets.
"""

import sys
import json
import pickle
from pathlib import Path

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
from src.models.efficientnet_model import EfficientNetHierarchicalClassifier
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


def load_model(checkpoint_path, model_type='resnet'):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle wrapped checkpoints
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model
    if model_type == 'resnet':
        model = HierarchicalGrainClassifier()
    elif model_type == 'efficientnet':
        model = EfficientNetHierarchicalClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device


def extract_probabilities(model, loader, device):
    """
    Extract prediction probabilities from a model.
    
    Returns:
        probs_array: (N, 4) array of class probabilities
        labels_array: (N,) array of true class labels
    """
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (hierarchical)
            predictions = model.get_predictions(outputs)
            
            # Models output single logits per stage, apply sigmoid to get probabilities
            stage1_probs = torch.sigmoid(outputs['stage1']).squeeze()  # P(non-peloid)
            stage2_probs = torch.sigmoid(outputs['stage2']).squeeze()  # P(non-ooid-like)
            stage3_probs = torch.sigmoid(outputs['stage3']).squeeze()  # P(broken)
            
            # Handle single-sample batches
            if stage1_probs.dim() == 0:
                stage1_probs = stage1_probs.unsqueeze(0)
                stage2_probs = stage2_probs.unsqueeze(0)
                stage3_probs = stage3_probs.unsqueeze(0)
            
            # Convert hierarchical predictions to class probabilities
            batch_probs = []
            for i in range(len(images)):
                s1_prob = stage1_probs[i].item()
                s2_prob = stage2_probs[i].item()
                s3_prob = stage3_probs[i].item()
                
                # Compute approximate class probabilities
                p_peloid = 1 - s1_prob
                p_ooid = s1_prob * (1 - s2_prob) * (1 - s3_prob)
                p_broken = s1_prob * (1 - s2_prob) * s3_prob
                p_intraclast = s1_prob * s2_prob
                
                # Normalize to sum to 1
                total = p_peloid + p_ooid + p_broken + p_intraclast
                if total > 0:
                    probs = [p_peloid / total, p_ooid / total, p_broken / total, p_intraclast / total]
                else:
                    probs = [0.25, 0.25, 0.25, 0.25]
                
                batch_probs.append(probs)
            
            all_probs.extend(batch_probs)
            
            # Convert hierarchical labels to class indices
            # labels is a list of dicts, one per sample in batch
            for i in range(len(images)):
                # Extract stage labels for this sample
                s1 = labels[i]['stage1'] if isinstance(labels, list) else labels['stage1'][i].item()
                s2 = labels[i]['stage2'] if isinstance(labels, list) else labels['stage2'][i].item()
                s3 = labels[i]['stage3'] if isinstance(labels, list) else labels['stage3'][i].item()
                
                class_idx = hierarchical_to_class(s1, s2, s3)
                all_labels.append(class_idx)
    
    return np.array(all_probs), np.array(all_labels)


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
            # Use probability for class i
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


def evaluate_ensemble(meta_clf, loader, checkpoint_paths, model_types, device, class_names):
    """Evaluate ensemble on a dataset."""
    # Extract features from all models
    all_features = []
    true_labels = None
    
    for name in ['baseline', 'exp2', 'exp3', 'exp4']:
        checkpoint_path = project_root / checkpoint_paths[name]
        model, device = load_model(checkpoint_path, model_types[name])
        
        probs, labels = extract_probabilities(model, loader, device)
        all_features.append(probs)
        
        if true_labels is None:
            true_labels = labels
    
    # Combine features
    X = np.concatenate(all_features, axis=1)
    y_true = true_labels
    
    # Predict with meta-classifier
    y_pred = meta_clf.predict(X)
    y_probs = meta_clf.predict_proba(X)
    
    # Compute overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute per-class metrics
    metrics = compute_metrics_per_class(y_true, y_pred, y_probs, class_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'per_class_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'true_labels': y_true.tolist()
    }


def print_results(results, split_name, class_names):
    """Print evaluation results."""
    print()
    print(f"{split_name.upper()} SET RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print()
    
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name in class_names:
        m = results['per_class_metrics'][class_name]
        print(f"{class_name:<15} {m['precision']:>11.4f} {m['recall']:>11.4f} {m['pr_auc']:>11.4f} {m['support']:>9d}")
    
    print()
    print("Confusion Matrix:")
    print("-" * 80)
    cm = np.array(results['confusion_matrix'])
    
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


def main():
    print("=" * 80)
    print("Experiment 5: Meta-Classifier Ensemble Evaluation")
    print("=" * 80)
    
    # Load metadata
    ensemble_dir = project_root / 'checkpoints' / 'exp5_ensemble'
    metadata_path = ensemble_dir / 'ensemble_metadata.json'
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        print("Run exp5_train_ensemble.py first!")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    checkpoint_paths = metadata['checkpoint_paths']
    model_types = metadata['model_types']
    class_names = metadata['class_names']
    
    print(f"✓ Loaded metadata")
    print(f"  Models: {list(checkpoint_paths.keys())}")
    print(f"  Feature dim: {metadata['feature_dim']}")
    print()
    
    # Load meta-classifier
    model_path = ensemble_dir / 'meta_classifier.pkl'
    with open(model_path, 'rb') as f:
        meta_clf = pickle.load(f)
    
    print(f"✓ Loaded meta-classifier from: {model_path}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
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
    
    # Evaluate on validation set
    print()
    print("Evaluating on validation set...")
    print("-" * 80)
    val_results = evaluate_ensemble(
        meta_clf, val_loader, checkpoint_paths, model_types, device, class_names
    )
    print_results(val_results, 'validation', class_names)
    
    # Evaluate on test set
    print()
    print("Evaluating on test set...")
    print("-" * 80)
    test_results = evaluate_ensemble(
        meta_clf, test_loader, checkpoint_paths, model_types, device, class_names
    )
    print_results(test_results, 'test', class_names)
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / 'exp5_ensemble.json'
    with open(output_path, 'w') as f:
        json.dump({
            'validation': {k: v for k, v in val_results.items() if k not in ['predictions', 'true_labels']},
            'test': {k: v for k, v in test_results.items() if k not in ['predictions', 'true_labels']},
            'metadata': metadata
        }, f, indent=2)
    
    print()
    print(f"✓ Results saved to: {output_path}")
    print()
    print("=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
