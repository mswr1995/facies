"""
Evaluation script for trained hierarchical grain classifier.

Usage:
    python scripts/evaluate.py --fold 0 --checkpoint checkpoints/fold_0/best_model.pt
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import GrainDataset
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.training.metrics import HierarchicalMetrics


def load_model(checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation metrics at checkpoint:")
    for key, value in checkpoint['metrics'].items():
        if 'acc' in key or 'f1' in key:
            print(f"  {key}: {value:.4f}")
    
    return model


def evaluate(model, data_loader, device):
    """Run evaluation on data loader."""
    all_logits = {'stage1': [], 'stage2': [], 'stage3': []}
    all_labels = {'stage1': [], 'stage2': [], 'stage3': []}
    
    metric_computer = HierarchicalMetrics()
    
    print(f"\nRunning inference on {len(data_loader.dataset)} samples...")
    
    with torch.no_grad():
        for images, labels, metadata in data_loader:
            images = images.to(device)
            
            logits = model(images)
            
            for key in logits:
                all_logits[key].append(logits[key].cpu())
                all_labels[key].append(labels[key])
    
    # Concatenate all predictions
    for key in all_logits:
        all_logits[key] = torch.cat(all_logits[key], dim=0)
        all_labels[key] = torch.cat(all_labels[key], dim=0)
    
    # Compute metrics
    metrics = metric_computer.compute_all_metrics(all_logits, all_labels, model)
    
    return metrics, all_logits, all_labels


def print_results(metrics, all_logits, all_labels, model):
    """Print detailed evaluation results."""
    metric_computer = HierarchicalMetrics()
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\nStage-wise Performance:")
    print("-" * 70)
    print(f"Stage 1 (Peloid vs Non-peloid):")
    print(f"  Accuracy: {metrics['stage1_acc']:.4f}")
    print(f"  F1 Score: {metrics['stage1_f1']:.4f}")
    print(f"  Samples:  {metrics['stage1_count']}")
    
    print(f"\nStage 2 (Ooid-like vs Intraclast):")
    print(f"  Accuracy: {metrics['stage2_acc']:.4f}")
    print(f"  F1 Score: {metrics['stage2_f1']:.4f}")
    print(f"  Samples:  {metrics['stage2_count']}")
    
    print(f"\nStage 3 (Whole vs Broken Ooid):")
    print(f"  Accuracy: {metrics['stage3_acc']:.4f}")
    print(f"  F1 Score: {metrics['stage3_f1']:.4f}")
    print(f"  Samples:  {metrics['stage3_count']}")
    
    print(f"\nOverall Hierarchical Accuracy: {metrics['overall_acc']:.4f}")
    
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = metric_computer.compute_confusion_matrix(all_logits, all_labels, model)
    
    classes = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    print("\nPredicted →")
    print("Actual ↓")
    print("\n        ", end="")
    for cls in classes:
        print(f"{cls:>13}", end="")
    print()
    print("-" * 70)
    
    for i, cls in enumerate(classes):
        print(f"{cls:>8}", end="")
        for j in range(4):
            print(f"{cm[i, j]:>13}", end="")
        print()
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    metric_computer.print_classification_report(all_logits, all_labels, model)
    
    print("\n" + "=" * 70)


def main(args):
    """Main evaluation function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Create data loader
    metadata_path = f'data/processed/fold_{args.fold}_metadata.json'
    patches_dir = 'data/processed/patches'
    
    dataset = GrainDataset(
        metadata_path=metadata_path,
        patches_dir=patches_dir,
        split='val'
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nEvaluating on validation set:")
    print(f"  Fold: {args.fold}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run evaluation
    metrics, all_logits, all_labels = evaluate(model, data_loader, device)
    
    # Print results
    print_results(metrics, all_logits, all_labels, model)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'fold': args.fold,
            'checkpoint': str(checkpoint_path),
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in metrics.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained hierarchical grain classifier')
    
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fold_0/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='results/fold_0_evaluation.json',
                       help='Output path for results')
    
    args = parser.parse_args()
    
    main(args)
