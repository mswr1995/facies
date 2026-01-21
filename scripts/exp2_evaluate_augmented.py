"""
Evaluate Experiment 2 (Data Augmentation) on test set.
"""
import torch
import numpy as np
from pathlib import Path
import json
from glob import glob
from sklearn.metrics import precision_score, recall_score, average_precision_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_augmented import create_augmented_dataloaders
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
                final_classes.append(3)
            else:
                prob_whole = torch.sigmoid(logits['stage3'][i]).item()
                is_whole = prob_whole > t3
                
                if is_whole:
                    final_classes.append(1)
                else:
                    final_classes.append(2)
    
    return np.array(final_classes)


def evaluate(model, data_loader, device, t1=0.5, t2=0.5, t3=0.5):
    """Evaluate model on data loader."""
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels_dict, metadata in data_loader:
            images = images.to(device)
            
            logits = model(images)
            preds = hierarchical_predict(logits, t1, t2, t3)
            
            label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
            true_labels = [label_map[l] for l in metadata['label']]
            
            all_preds.extend(preds)
            all_true.extend(true_labels)
    
    return np.array(all_preds), np.array(all_true)


def compute_metrics(y_true, y_pred, class_names):
    """Compute per-class metrics."""
    results = {'per_class': {}}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if y_true_binary.sum() > 0:
            pr_auc = average_precision_score(y_true_binary, y_pred_binary)
        else:
            pr_auc = 0.0
        
        support = int(y_true_binary.sum())
        
        results['per_class'][class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'pr_auc': float(pr_auc),
            'support': support
        }
    
    return results


def print_results(split_name, metrics, class_names):
    """Print formatted results."""
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Support'}")
    print("-"*90)
    
    for class_name in class_names:
        m = metrics['per_class'][class_name]
        print(f"{class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['pr_auc']:<12.4f} {m['support']}")


def main():
    device = 'cpu'
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    # Find best checkpoint from exp2
    checkpoint_dir = 'checkpoints/exp2_augmented'
    checkpoints = glob(f'{checkpoint_dir}/best_model_*.pth')
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    best_checkpoint = max(checkpoints, key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    
    print(f"Loading best checkpoint: {best_checkpoint}")
    
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    
    # Create dataloaders
    _, val_loader, test_loader = create_augmented_dataloaders(
        batch_size=32,
        num_workers=0,
        strong_augmentation=False
    )
    
    # Evaluate
    print("="*90)
    print("EXPERIMENT 2: DATA AUGMENTATION EVALUATION")
    print("="*90)
    
    print("\nEvaluating validation set...")
    y_pred_val, y_true_val = evaluate(model, val_loader, device)
    val_metrics = compute_metrics(y_true_val, y_pred_val, class_names)
    
    print("\nVALIDATION SET:")
    print_results("Validation", val_metrics, class_names)
    
    print("\nEvaluating test set...")
    y_pred_test, y_true_test = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test, class_names)
    
    print("\nTEST SET:")
    print_results("Test", test_metrics, class_names)
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results = {
        'experiment': 'Data Augmentation (Strong)',
        'checkpoint': best_checkpoint,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    output_path = 'results/exp2_data_augmentation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "="*90)
    
    return test_metrics


if __name__ == '__main__':
    test_metrics = main()
