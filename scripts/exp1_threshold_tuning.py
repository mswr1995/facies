"""
Experiment 1: Threshold Tuning
Optimize T2 and T3 thresholds on validation set to improve rare class detection.
Keep T1=0.5 (peloid/non-peloid is working well).
"""
import torch
import numpy as np
from pathlib import Path
import json
from glob import glob
from sklearn.metrics import precision_score, recall_score, average_precision_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import create_new_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier


def hierarchical_predict(logits, t1=0.5, t2=0.5, t3=0.5):
    """Apply hierarchical classification with custom thresholds."""
    batch_size = logits['stage1'].shape[0]
    final_classes = []
    
    for i in range(batch_size):
        prob_peloid = torch.sigmoid(logits['stage1'][i]).item()
        is_peloid = prob_peloid > t1
        
        if is_peloid:
            final_classes.append(0)
        else:
            prob_ooid_like = torch.sigmoid(logits['stage2'][i]).item()
            is_ooid_like = prob_ooid_like > t2
            
            if not is_ooid_like:
                final_classes.append(3)  # Intraclast
            else:
                prob_whole = torch.sigmoid(logits['stage3'][i]).item()
                is_whole = prob_whole > t3
                
                if is_whole:
                    final_classes.append(1)  # Ooid
                else:
                    final_classes.append(2)  # Broken ooid
    
    return np.array(final_classes)


def get_validation_predictions(model, data_loader, device):
    """Get all predictions and labels from validation set."""
    model.eval()
    
    all_logits = {'stage1': [], 'stage2': [], 'stage3': []}
    all_true = []
    
    with torch.no_grad():
        for images, labels_dict, metadata in data_loader:
            images = images.to(device)
            
            logits = model(images)
            all_logits['stage1'].append(logits['stage1'].cpu())
            all_logits['stage2'].append(logits['stage2'].cpu())
            all_logits['stage3'].append(logits['stage3'].cpu())
            
            label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
            true_labels = [label_map[l] for l in metadata['label']]
            all_true.extend(true_labels)
    
    # Concatenate all batches
    all_logits['stage1'] = torch.cat(all_logits['stage1'], dim=0)
    all_logits['stage2'] = torch.cat(all_logits['stage2'], dim=0)
    all_logits['stage3'] = torch.cat(all_logits['stage3'], dim=0)
    all_true = np.array(all_true)
    
    return all_logits, all_true


def compute_metrics(y_true, y_pred, class_idx):
    """Compute metrics for a specific class."""
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return precision, recall


def objective_score(y_true, y_pred):
    """
    Compute objective score for threshold tuning.
    Goal: Maximize intraclast recall while maintaining overall performance.
    Weight: Intraclast recall 60%, Ooid recall 30%, Peloid recall 10%
    """
    intra_prec, intra_recall = compute_metrics(y_true, y_pred, 3)
    ooid_prec, ooid_recall = compute_metrics(y_true, y_pred, 1)
    peloid_prec, peloid_recall = compute_metrics(y_true, y_pred, 0)
    
    # Weighted score favoring intraclast improvement
    score = (0.60 * intra_recall + 0.30 * ooid_recall + 0.10 * peloid_recall)
    
    return score, {
        'intraclast_recall': intra_recall,
        'ooid_recall': ooid_recall,
        'peloid_recall': peloid_recall
    }


def tune_thresholds(all_logits, all_true, device):
    """
    Grid search over T2 and T3 to find best thresholds.
    T1 is fixed at 0.5 (working well for peloid detection).
    """
    print("\n" + "="*90)
    print("THRESHOLD TUNING: Grid Search")
    print("="*90)
    print("\nSearching T2 (ooid-like vs intraclast) and T3 (whole vs broken)")
    print("T1 fixed at 0.5 (peloid detection working well)")
    
    # Grid ranges
    t2_range = np.arange(0.3, 0.8, 0.05)  # 0.3, 0.35, 0.4, ..., 0.75
    t3_range = np.arange(0.3, 0.8, 0.05)
    
    best_score = -1
    best_t2 = 0.5
    best_t3 = 0.5
    best_metrics = {}
    results = []
    
    total = len(t2_range) * len(t3_range)
    current = 0
    
    for t2 in t2_range:
        for t3 in t3_range:
            current += 1
            
            # Create logits dict for prediction
            logits = {
                'stage1': all_logits['stage1'],
                'stage2': all_logits['stage2'],
                'stage3': all_logits['stage3']
            }
            
            # Get predictions with current thresholds
            y_pred = hierarchical_predict(logits, t1=0.5, t2=t2, t3=t3)
            
            # Compute score
            score, metrics = objective_score(all_true, y_pred)
            
            results.append({
                't2': float(t2),
                't3': float(t3),
                'score': float(score),
                'intraclast_recall': float(metrics['intraclast_recall']),
                'ooid_recall': float(metrics['ooid_recall']),
                'peloid_recall': float(metrics['peloid_recall'])
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_t2 = t2
                best_t3 = t3
                best_metrics = metrics
            
            # Progress
            if current % 25 == 0:
                print(f"  {current}/{total} combinations tested...")
    
    print(f"\n✅ Grid search complete!")
    print(f"\nBest Thresholds Found:")
    print(f"  T1 (peloid)     = 0.50 (fixed)")
    print(f"  T2 (ooid-like)  = {best_t2:.2f}")
    print(f"  T3 (whole)      = {best_t3:.2f}")
    print(f"\nMetrics with best thresholds:")
    print(f"  Intraclast Recall: {best_metrics['intraclast_recall']:.4f}")
    print(f"  Ooid Recall:       {best_metrics['ooid_recall']:.4f}")
    print(f"  Peloid Recall:     {best_metrics['peloid_recall']:.4f}")
    print(f"  Objective Score:   {best_score:.4f}")
    
    return best_t2, best_t3, results


def evaluate_test_set(model, test_loader, device, t1, t2, t3):
    """Evaluate on test set with tuned thresholds."""
    model.eval()
    
    all_logits = {'stage1': [], 'stage2': [], 'stage3': []}
    all_true = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels_dict, metadata in test_loader:
            images = images.to(device)
            
            logits = model(images)
            all_logits['stage1'].append(logits['stage1'].cpu())
            all_logits['stage2'].append(logits['stage2'].cpu())
            all_logits['stage3'].append(logits['stage3'].cpu())
            
            label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
            true_labels = [label_map[l] for l in metadata['label']]
            all_true.extend(true_labels)
    
    # Concatenate
    all_logits['stage1'] = torch.cat(all_logits['stage1'], dim=0)
    all_logits['stage2'] = torch.cat(all_logits['stage2'], dim=0)
    all_logits['stage3'] = torch.cat(all_logits['stage3'], dim=0)
    all_true = np.array(all_true)
    
    # Get predictions
    y_pred = hierarchical_predict(all_logits, t1=t1, t2=t2, t3=t3)
    
    # Compute per-class metrics
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (all_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if y_true_binary.sum() > 0:
            pr_auc = average_precision_score(y_true_binary, y_pred_binary)
        else:
            pr_auc = 0.0
        
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'pr_auc': float(pr_auc),
            'support': int(y_true_binary.sum())
        }
    
    return metrics, y_pred, all_true


def print_test_results(metrics, class_names):
    """Print test set results."""
    print("\n" + "="*90)
    print("TEST SET EVALUATION (with tuned thresholds)")
    print("="*90)
    
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Support'}")
    print("-"*90)
    
    for class_name in class_names:
        m = metrics[class_name]
        print(f"{class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['pr_auc']:<12.4f} {m['support']}")


def main():
    device = 'cpu'
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    # Load best baseline model
    checkpoint_dir = 'checkpoints/new_split_v2'
    checkpoints = glob(f'{checkpoint_dir}/best_model_*.pth')
    best_checkpoint = max(checkpoints, key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    
    print(f"\nLoading baseline model: {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    
    # Create dataloaders
    _, val_loader, test_loader = create_new_dataloaders(batch_size=32, num_workers=0)
    
    # Get validation predictions
    print("Collecting validation set predictions...")
    all_logits, all_true = get_validation_predictions(model, val_loader, device)
    print("✅ Validation predictions collected\n")
    
    # Tune thresholds on validation set
    best_t2, best_t3, search_results = tune_thresholds(all_logits, all_true, device)
    
    # Evaluate on test set with best thresholds
    print("\n" + "="*90)
    print("EVALUATING TEST SET WITH TUNED THRESHOLDS")
    print("="*90)
    
    test_metrics, test_preds, test_true = evaluate_test_set(
        model, test_loader, device, 
        t1=0.5, t2=best_t2, t3=best_t3
    )
    
    print_test_results(test_metrics, class_names)
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results_data = {
        'experiment': 'Threshold Tuning',
        'baseline_checkpoint': best_checkpoint,
        'baseline_thresholds': {'t1': 0.5, 't2': 0.5, 't3': 0.5},
        'best_thresholds': {
            't1': 0.5,
            't2': float(best_t2),
            't3': float(best_t3)
        },
        'grid_search_results': search_results,
        'test_metrics': test_metrics,
        'test_predictions': test_preds.tolist(),
        'test_true': test_true.tolist()
    }
    
    output_path = 'results/exp1_threshold_tuning.json'
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "="*90)
    
    return test_metrics


if __name__ == '__main__':
    test_metrics = main()
