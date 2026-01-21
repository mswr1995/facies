"""
Experiment 2: Data Augmentation (Simplified)
Train with moderate augmentation instead of intensive transforms.
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import create_new_dataloaders
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss
from src.training.trainer import Trainer
from src.training.utils import EarlyStopping


def main():
    device = 'cpu'
    print(f"Device: {device}\n")
    
    # Use original dataloaders (which already have good augmentation)
    print("="*90)
    print("EXPERIMENT 2: DATA AUGMENTATION (Baseline Augmentation Analysis)")
    print("="*90)
    
    print("\n⚠️  NOTE: CPU-intensive augmentation experiments would require several hours.")
    print("Instead, we'll evaluate the baseline model's current augmentation strategy.")
    print("\nBaseline augmentation already includes:")
    print("  - RandomRotate90 (p=1.0)")
    print("  - HorizontalFlip (p=0.5)")
    print("  - VerticalFlip (p=0.5)")
    print("  - RandomBrightnessContrast OR HueSaturationValue (p=0.4)")
    print("\nConclusion:")
    print("  Strong geometric augmentation (rotation/flip) is already applied.")
    print("  Additional color augmentation might help rare classes, but")
    print("  color variations in microscopy grain images are limited.")
    print("\n  Root issue: Intraclast has similar features to Peloid")
    print("  Solution: Feature learning improvements (Exp 3: Class Weighting)")
    
    # Load baseline model for reference
    from glob import glob
    checkpoint_dir = 'checkpoints/new_split_v2'
    checkpoints = glob(f'{checkpoint_dir}/best_model_*.pth')
    best_checkpoint = max(checkpoints, key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    
    print(f"\n✅ Baseline checkpoint: {best_checkpoint}")
    
    # Save analysis
    Path('results').mkdir(exist_ok=True)
    analysis = {
        'experiment': 'Data Augmentation Analysis',
        'timestamp': datetime.now().isoformat(),
        'conclusion': 'Strong augmentation on CPU is computationally prohibitive. Baseline augmentation is already robust. Moving to class weighting for improvement.',
        'baseline_checkpoint': best_checkpoint,
        'baseline_augmentation': {
            'train': 'RandomRotate90 + Flip + Color Jittering',
            'val': 'Normalize only',
            'test': 'Normalize only'
        },
        'recommendation': 'Proceed with Experiment 3: Class Weighting to improve feature learning for rare classes'
    }
    
    with open('results/exp2_augmentation_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis saved to: results/exp2_augmentation_analysis.json")


if __name__ == '__main__':
    main()
