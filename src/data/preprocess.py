"""
Preprocessing pipeline: Extract grain patches from labelme annotations.

Generates 96x96 patches for all grains using centroid-based extraction.
Organizes patches by fold for cross-validation training.
"""

import sys
sys.path.append('.')

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import argparse

from src.data.labelme_loader import (
    load_all_annotations,
    load_labelme_json,
    load_image_from_labelme,
    extract_grain_patch
)
from src.data.splits import create_stratified_folds


def preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    patch_size: int = 96,
    with_mask: bool = True,
    n_folds: int = 5,
    random_state: int = 42
):
    """
    Complete preprocessing pipeline.
    
    1. Load all annotations
    2. Create 5-fold splits
    3. Extract patches for all grains
    4. Save organized by fold
    
    Args:
        data_dir: Directory containing labelme JSON and images
        output_dir: Directory to save processed patches
        patch_size: Size of square patches (default 96)
        with_mask: Apply grain masking (default True)
        n_folds: Number of CV folds (default 5)
        random_state: Random seed
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Load annotations
    print("\n[1/4] Loading annotations...")
    annotations = load_all_annotations(data_dir, filter_classes=True)
    
    total_grains = sum(len(grains) for grains in annotations.values())
    print(f"  ✓ Loaded {len(annotations)} images")
    print(f"  ✓ Total grains: {total_grains}")
    
    # Step 2: Create folds
    print("\n[2/4] Creating 5-fold stratified splits...")
    folds = create_stratified_folds(annotations, n_splits=n_folds, random_state=random_state)
    
    # Save fold splits
    splits_dir = output_dir / 'cv_splits'
    splits_dir.mkdir(exist_ok=True)
    
    for fold in folds:
        fold_file = splits_dir / f"fold_{fold['fold']}.json"
        with open(fold_file, 'w') as f:
            json.dump(fold, f, indent=2)
    
    print(f"  ✓ Created {n_folds} folds")
    print(f"  ✓ Saved splits to {splits_dir}")
    
    # Step 3: Extract all patches
    print(f"\n[3/4] Extracting {patch_size}x{patch_size} patches...")
    print(f"  Masking: {'enabled' if with_mask else 'disabled'}")
    
    patches_dir = output_dir / 'patches'
    patches_dir.mkdir(exist_ok=True)
    
    # Create metadata for all grains
    grain_metadata = []
    grain_id = 0
    
    for image_name in tqdm(sorted(annotations.keys()), desc="Processing images"):
        # Load image
        json_path = data_dir / f"{image_name}.json"
        json_data = load_labelme_json(json_path)
        image = load_image_from_labelme(json_data, data_dir)
        
        grains = annotations[image_name]
        
        for grain_idx, grain in enumerate(grains):
            # Extract patch
            patch = extract_grain_patch(
                image,
                grain,
                patch_size=patch_size,
                with_mask=with_mask
            )
            
            # Save patch
            patch_filename = f"grain_{grain_id:06d}.png"
            patch_path = patches_dir / patch_filename
            
            # Convert RGB to BGR for cv2.imwrite
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(patch_path), patch_bgr)
            
            # Store metadata
            grain_metadata.append({
                'grain_id': grain_id,
                'image_name': image_name,
                'grain_idx': grain_idx,
                'label': grain['label'],
                'centroid': grain['centroid'],
                'patch_filename': patch_filename,
            })
            
            grain_id += 1
    
    print(f"  ✓ Extracted {grain_id} patches")
    print(f"  ✓ Saved to {patches_dir}")
    
    # Step 4: Save metadata
    print("\n[4/4] Saving metadata...")
    
    metadata_file = output_dir / 'grain_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(grain_metadata, f, indent=2)
    
    # Create fold-specific metadata
    for fold in folds:
        fold_meta = {
            'fold': fold['fold'],
            'train_grains': [],
            'val_grains': []
        }
        
        for grain_meta in grain_metadata:
            if grain_meta['image_name'] in fold['train']:
                fold_meta['train_grains'].append(grain_meta)
            elif grain_meta['image_name'] in fold['val']:
                fold_meta['val_grains'].append(grain_meta)
        
        fold_meta_file = output_dir / f"fold_{fold['fold']}_metadata.json"
        with open(fold_meta_file, 'w') as f:
            json.dump(fold_meta, f, indent=2)
    
    print(f"  ✓ Saved global metadata: {metadata_file}")
    print(f"  ✓ Saved fold-specific metadata")
    
    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"  ├── patches/              ({grain_id} PNG files)")
    print(f"  ├── cv_splits/            ({n_folds} fold splits)")
    print(f"  ├── grain_metadata.json   (all grain info)")
    print(f"  └── fold_*_metadata.json  (per-fold grain lists)")
    
    print(f"\nPatch details:")
    print(f"  Size: {patch_size}x{patch_size} px")
    print(f"  Masking: {'Yes (zeros outside grain)' if with_mask else 'No'}")
    print(f"  Format: PNG (RGB)")
    
    print(f"\n✅ Ready for PyTorch DataLoader!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess carbonate grain dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing labelme annotations and images')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed patches')
    parser.add_argument('--patch_size', type=int, default=96,
                        help='Size of square patches (default: 96)')
    parser.add_argument('--no_mask', action='store_true',
                        help='Disable grain masking')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        patch_size=args.patch_size,
        with_mask=not args.no_mask,
        n_folds=args.n_folds,
        random_state=args.random_state
    )
