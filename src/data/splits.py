"""
5-Fold Stratified Cross-Validation Split for Carbonate Grain Dataset.

Ensures that rare classes (especially broken ooids) are present in all folds.
Splits by image (not by grain) to prevent data leakage.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import json


def assign_image_tier(image_grains: List[Dict]) -> str:
    """
    Assign tier to an image based on class diversity.
    
    Tier 1: Contains broken ooids (critical - rarest class)
    Tier 2: Contains intraclasts but no broken ooids
    Tier 3: Only peloids and ooids
    
    Args:
        image_grains: List of grain dictionaries for one image
        
    Returns:
        Tier name string
    """
    labels = [g['label'] for g in image_grains]
    
    if 'Broken ooid' in labels:
        return 'Tier1_Broken_Ooid'
    elif 'Intraclast' in labels:
        return 'Tier2_Intraclast'
    else:
        return 'Tier3_Basic'


def create_stratified_folds(
    annotations: Dict[str, List[Dict]],
    n_splits: int = 5,
    random_state: int = 42
) -> List[Dict[str, List[str]]]:
    """
    Create 5-fold stratified cross-validation splits.
    
    Args:
        annotations: Dictionary mapping image_name -> list of grains
        n_splits: Number of folds (default 5)
        random_state: Random seed for reproducibility
        
    Returns:
        List of fold dictionaries, each containing:
            - 'train': list of training image names
            - 'val': list of validation image names
            - 'tier_counts': statistics about tier distribution
    """
    # Assign tier to each image
    image_tiers = {}
    for img_name, grains in annotations.items():
        image_tiers[img_name] = assign_image_tier(grains)
    
    # Prepare data for stratification
    image_names = list(image_tiers.keys())
    tier_labels = [image_tiers[name] for name in image_names]
    
    # Create stratified splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_names, tier_labels)):
        train_images = [image_names[i] for i in train_idx]
        val_images = [image_names[i] for i in val_idx]
        
        # Count grains per class in each split
        train_class_counts = Counter()
        val_class_counts = Counter()
        
        for img in train_images:
            for grain in annotations[img]:
                train_class_counts[grain['label']] += 1
        
        for img in val_images:
            for grain in annotations[img]:
                val_class_counts[grain['label']] += 1
        
        # Tier distribution
        train_tiers = Counter([image_tiers[img] for img in train_images])
        val_tiers = Counter([image_tiers[img] for img in val_images])
        
        fold_info = {
            'fold': fold_idx,
            'train': train_images,
            'val': val_images,
            'train_class_counts': dict(train_class_counts),
            'val_class_counts': dict(val_class_counts),
            'train_tier_counts': dict(train_tiers),
            'val_tier_counts': dict(val_tiers),
        }
        
        folds.append(fold_info)
    
    return folds


def save_fold_splits(folds: List[Dict], output_dir: Path):
    """
    Save fold splits to JSON files for reproducibility.
    
    Args:
        folds: List of fold dictionaries
        output_dir: Directory to save split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each fold
    for fold in folds:
        fold_file = output_dir / f"fold_{fold['fold']}.json"
        with open(fold_file, 'w') as f:
            json.dump(fold, f, indent=2)
    
    # Save summary
    summary = {
        'n_folds': len(folds),
        'total_images': len(folds[0]['train']) + len(folds[0]['val']),
        'folds': [
            {
                'fold': f['fold'],
                'train_images': len(f['train']),
                'val_images': len(f['val']),
                'train_tiers': f['train_tier_counts'],
                'val_tiers': f['val_tier_counts'],
            }
            for f in folds
        ]
    }
    
    summary_file = output_dir / 'cv_splits_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved {len(folds)} fold splits to {output_dir}")


def print_fold_statistics(folds: List[Dict]):
    """
    Print detailed statistics about fold splits.
    """
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION SPLITS")
    print("=" * 70)
    
    for fold in folds:
        print(f"\n{'='*70}")
        print(f"FOLD {fold['fold']}")
        print(f"{'='*70}")
        
        print(f"\nTrain: {len(fold['train'])} images")
        print(f"  Tiers: {fold['train_tier_counts']}")
        print(f"  Classes:")
        for cls, count in sorted(fold['train_class_counts'].items(), key=lambda x: -x[1]):
            print(f"    {cls:15s}: {count:4d}")
        
        print(f"\nValidation: {len(fold['val'])} images")
        print(f"  Tiers: {fold['val_tier_counts']}")
        print(f"  Classes:")
        for cls, count in sorted(fold['val_class_counts'].items(), key=lambda x: -x[1]):
            print(f"    {cls:15s}: {count:4d}")
        
        # Check if broken ooids present
        has_broken_train = fold['train_class_counts'].get('Broken ooid', 0) > 0
        has_broken_val = fold['val_class_counts'].get('Broken ooid', 0) > 0
        
        status = "✅" if (has_broken_train and has_broken_val) else "⚠️"
        print(f"\n{status} Broken ooids: Train={has_broken_train}, Val={has_broken_val}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Overall statistics
    all_train_counts = Counter()
    all_val_counts = Counter()
    
    for fold in folds:
        for cls, count in fold['train_class_counts'].items():
            all_train_counts[cls] += count
        for cls, count in fold['val_class_counts'].items():
            all_val_counts[cls] += count
    
    print(f"\nTotal across all folds:")
    print(f"  Train grains: {sum(all_train_counts.values())} (avg per fold: {sum(all_train_counts.values())//len(folds)})")
    print(f"  Val grains: {sum(all_val_counts.values())} (avg per fold: {sum(all_val_counts.values())//len(folds)})")
    
    # Check balance
    broken_ooid_counts = [f['val_class_counts'].get('Broken ooid', 0) for f in folds]
    min_broken = min(broken_ooid_counts)
    max_broken = max(broken_ooid_counts)
    
    if min_broken > 0:
        print(f"\n✅ All folds have broken ooids in validation!")
        print(f"   Range: {min_broken} to {max_broken} broken ooids per fold")
    else:
        print(f"\n⚠️ WARNING: Some folds have no broken ooids in validation!")


if __name__ == '__main__':
    # Test the splitting
    import sys
    sys.path.append('.')
    
    from src.data.labelme_loader import load_all_annotations
    from pathlib import Path
    
    print("Loading annotations...")
    data_dir = Path('data/raw')
    annotations = load_all_annotations(data_dir, filter_classes=True)
    
    print(f"Loaded {len(annotations)} images")
    
    # Create folds
    print("\nCreating 5-fold stratified splits...")
    folds = create_stratified_folds(annotations, n_splits=5, random_state=42)
    
    # Print statistics
    print_fold_statistics(folds)
    
    # Save splits
    output_dir = Path('data/processed/cv_splits')
    save_fold_splits(folds, output_dir)
    
    print(f"\n✅ Fold splits ready for training!")
