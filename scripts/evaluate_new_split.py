"""
Evaluate trained model on new split (validation and test sets).
"""
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import create_new_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier


def hierarchical_predict(logits, t1=0.5, t2=0.5, t3=0.5):
    """Apply hierarchical classification with custom thresholds."""
    batch_size = logits['stage1'].shape[0]
    final_classes = []
    all_probs = []
    
    for i in range(batch_size):
        prob_peloid = torch.sigmoid(logits['stage1'][i]).item()
        is_peloid = prob_peloid > t1
        
        probs = [0.0, 0.0, 0.0, 0.0]  # [peloid, ooid, broken, intraclast]
        
        if is_peloid:
            probs[0] = prob_peloid
            final_classes.append(0)
        else:
            prob_ooid_like = torch.sigmoid(logits['stage2'][i]).item()
            is_ooid_like = prob_ooid_like > t2
            
            if not is_ooid_like:
                probs[3] = 1.0 - prob_ooid_like
                final_classes.append(3)
            else:
                prob_whole = torch.sigmoid(logits['stage3'][i]).item()
                is_whole = prob_whole > t3
                
                if is_whole:
                    probs[1] = prob_whole * prob_ooid_like
                    final_classes.append(1)
                else:
                    probs[2] = (1.0 - prob_whole) * prob_ooid_like
                    final_classes.append(2)
        
        all_probs.append(probs)
    
    return np.array(final_classes), np.array(all_probs)


def evaluate(model, data_loader, device, t1=0.5, t2=0.5, t3=0.5):
    """Evaluate model on data loader."""
    model.eval()
    
    all_preds = []
    all_true = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels_dict, metadata in data_loader:
            images = images.to(device)
            
            logits = model(images)
            
            preds, probs = hierarchical_predict(logits, t1, t2, t3)
            
            label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
            true_labels = [label_map[l] for l in metadata['label']]
            
            all_preds.extend(preds)
            all_true.extend(true_labels)
            all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_true), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs, class_names):
    """Compute precision, recall, F1, and PR-AUC for each class."""
    results = {
        'overall': {
            'accuracy': float(np.mean(y_true == y_pred)),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        y_scores = y_probs[:, i]
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if y_true_binary.sum() > 0:
            pr_auc = average_precision_score(y_true_binary, y_scores)
        else:
            pr_auc = 0.0
        
        support = int(y_true_binary.sum())
        
        results['per_class'][class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'pr_auc': float(pr_auc),
            'support': support
        }
    
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    
    return results


def print_results(split_name, results, class_names):
    """Print evaluation results."""
    print("\n" + "="*80)
    print(f"{split_name.upper()} SET RESULTS")
    print("="*80)
    
    print("\nOverall Metrics:")
    print(f"  Accuracy:        {results['overall']['accuracy']:.4f}")
    print(f"  Macro Precision: {results['overall']['macro_precision']:.4f}")
    print(f"  Macro Recall:    {results['overall']['macro_recall']:.4f}")
    print(f"  Macro F1:        {results['overall']['macro_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'PR-AUC':<12} {'Support'}")
    print("-"*80)
    
    for class_name in class_names:
        m = results['per_class'][class_name]
        print(f"{class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['pr_auc']:<12.4f} {m['support']}")
    
    print("\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"\n{'':>15} " + " ".join(f"{name:>10}" for name in class_names))
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>15} " + " ".join(f"{cm[i,j]:>10}" for j in range(len(class_names))))


def main():
    device = 'cpu'
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    # Load best checkpoint
    checkpoint_path = 'checkpoints/new_split/best_model_overall_acc_0.9375.pth'
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded")
    
    # Create dataloaders
    _, val_loader, test_loader = create_new_dataloaders(batch_size=32, num_workers=0)
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("EVALUATING ON VALIDATION SET")
    print("="*80)
    
    y_pred_val, y_true_val, y_probs_val = evaluate(model, val_loader, device)
    val_results = compute_metrics(y_true_val, y_pred_val, y_probs_val, class_names)
    print_results("Validation", val_results, class_names)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (NEVER SEEN BEFORE)")
    print("="*80)
    
    y_pred_test, y_true_test, y_probs_test = evaluate(model, test_loader, device)
    test_results = compute_metrics(y_true_test, y_pred_test, y_probs_test, class_names)
    print_results("Test", test_results, class_names)
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'thresholds': {'t1': 0.5, 't2': 0.5, 't3': 0.5},
        'validation': val_results,
        'test': test_results
    }
    
    output_path = 'results/new_split_evaluation.json'
    Path('results').mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Comparison
    print("\n" + "="*80)
    print("VALIDATION vs TEST COMPARISON")
    print("="*80)
    
    print(f"\nOverall:")
    print(f"  Accuracy:   Val {val_results['overall']['accuracy']:.4f} | Test {test_results['overall']['accuracy']:.4f}")
    print(f"  Macro F1:   Val {val_results['overall']['macro_f1']:.4f} | Test {test_results['overall']['macro_f1']:.4f}")
    
    print(f"\nPer-class F1:")
    for class_name in class_names:
        val_f1 = val_results['per_class'][class_name]['f1']
        test_f1 = test_results['per_class'][class_name]['f1']
        print(f"  {class_name:<15}: Val {val_f1:.4f} | Test {test_f1:.4f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
