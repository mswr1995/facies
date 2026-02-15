"""
Unified evaluation script for experiments.
Outputs results in the same format as COMPARISON_TABLE.md

Usage:
  python scripts/evaluate_experiment.py --checkpoint checkpoints/exp7_staged/best_model.pth --name "Staged Training"
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from torch.utils.data import DataLoader
from src.models.hierarchical_model import HierarchicalGrainClassifier


def evaluate(model, loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            preds = model.get_predictions(logits)
            
            true_labels = [label_map[l] for l in metadata['label']]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels)
    
    return np.array(all_preds), np.array(all_labels)


def format_results(y_true, y_pred, model_name):
    """Format results for COMPARISON_TABLE.md"""
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    # Per-class accuracy
    results = {}
    for i, cls in enumerate(class_names):
        mask = y_true == i
        total = mask.sum()
        correct = (y_pred[mask] == i).sum() if total > 0 else 0
        pct = int(round(100 * correct / total)) if total > 0 else 0
        results[cls] = {'correct': int(correct), 'total': int(total), 'pct': pct}
    
    overall_acc = (y_true == y_pred).mean() * 100
    
    # Format table row
    table_row = f"| {model_name} |"
    table_row += f" {results['Peloid']['correct']}/{results['Peloid']['total']} ({results['Peloid']['pct']}%) |"
    table_row += f" {results['Ooid']['correct']}/{results['Ooid']['total']} ({results['Ooid']['pct']}%) |"
    table_row += f" {results['Broken ooid']['correct']}/{results['Broken ooid']['total']} ({results['Broken ooid']['pct']}%) |"
    table_row += f" {results['Intraclast']['correct']}/{results['Intraclast']['total']} ({results['Intraclast']['pct']}%) |"
    table_row += f" {overall_acc:.1f}% |"
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_text = f"\n### {model_name}\n```\n"
    cm_text += "              Predicted\n"
    cm_text += "Actual      Pel  Ooid  Brok  Intra\n"
    short_names = ['Peloid', 'Ooid', 'Broken', 'Intraclast']
    for i, name in enumerate(short_names):
        cm_text += f"{name:<12}"
        for j in range(4):
            cm_text += f"{cm[i,j]:>5}"
        cm_text += "\n"
    cm_text += "```"
    
    return {
        'table_row': table_row,
        'confusion_matrix': cm_text,
        'accuracy': overall_acc,
        'per_class': results
    }


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading: {args.checkpoint}")
    model = HierarchicalGrainClassifier(pretrained=False).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both direct state_dict and wrapped checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load test data
    test_dataset = GrainDatasetNew(split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluate
    y_pred, y_true = evaluate(model, test_loader, device)
    results = format_results(y_true, y_pred, args.name)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS FOR COMPARISON_TABLE.md")
    print("="*70)
    print("\nTable row (copy this to Per-Class Accuracy table):")
    print(results['table_row'])
    print("\nConfusion matrix (copy this to Confusion Matrices section):")
    print(results['confusion_matrix'])
    
    # Save JSON
    if args.output:
        output = {
            'model_name': args.name,
            'checkpoint': args.checkpoint,
            'accuracy': results['accuracy'],
            'per_class': results['per_class'],
            'table_row': results['table_row'],
            'confusion_matrix_text': results['confusion_matrix']
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--name', type=str, required=True, help='Model name for table')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    
    args = parser.parse_args()
    main(args)

