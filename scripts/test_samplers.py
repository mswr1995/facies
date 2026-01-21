"""
Test script to verify class-aware samplers are working correctly.
Shows batch composition statistics.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

from src.data.dataset import GrainDataset
from src.data.samplers import HierarchicalBalancedSampler, StageWiseBatchSampler


def analyze_batch_composition(data_loader, dataset, num_batches=10):
    """Analyze class composition in batches."""
    batch_stats = []
    
    for i, (images, labels, metadata) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        # Count classes in batch
        batch_labels = [metadata['label'][j] for j in range(len(metadata['label']))]
        class_counts = Counter(batch_labels)
        
        # Count stage labels
        stage1_counts = Counter(labels['stage1'].numpy())
        stage2_valid = labels['stage2'][labels['stage2'] != -1]
        stage2_counts = Counter(stage2_valid.numpy()) if len(stage2_valid) > 0 else {}
        stage3_valid = labels['stage3'][labels['stage3'] != -1]
        stage3_counts = Counter(stage3_valid.numpy()) if len(stage3_valid) > 0 else {}
        
        batch_stats.append({
            'classes': class_counts,
            'stage1': stage1_counts,
            'stage2': stage2_counts,
            'stage3': stage3_counts,
            'total': len(batch_labels)
        })
    
    return batch_stats


def print_sampler_analysis(sampler_name, batch_stats):
    """Print analysis of batch composition."""
    print(f"\n{'='*70}")
    print(f"{sampler_name.upper()}")
    print(f"{'='*70}")
    
    # Aggregate statistics
    all_classes = Counter()
    all_stage1 = Counter()
    peloid_ratios = []
    
    for stats in batch_stats:
        all_classes.update(stats['classes'])
        all_stage1.update(stats['stage1'])
        
        total = stats['total']
        peloid_count = stats['classes'].get('Peloid', 0)
        peloid_ratio = peloid_count / total if total > 0 else 0
        peloid_ratios.append(peloid_ratio)
    
    print(f"\nOverall class distribution across {len(batch_stats)} batches:")
    for cls in ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']:
        count = all_classes.get(cls, 0)
        pct = 100 * count / sum(all_classes.values()) if sum(all_classes.values()) > 0 else 0
        print(f"  {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nStage 1 balance (across all batches):")
    peloid_total = all_stage1.get(1, 0)
    non_peloid_total = all_stage1.get(0, 0)
    total = peloid_total + non_peloid_total
    if total > 0:
        print(f"  Peloid (1):     {peloid_total:4d} ({100*peloid_total/total:5.1f}%)")
        print(f"  Non-peloid (0): {non_peloid_total:4d} ({100*non_peloid_total/total:5.1f}%)")
    
    print(f"\nPer-batch peloid ratio:")
    print(f"  Mean: {np.mean(peloid_ratios):.3f}")
    print(f"  Std:  {np.std(peloid_ratios):.3f}")
    print(f"  Min:  {np.min(peloid_ratios):.3f}")
    print(f"  Max:  {np.max(peloid_ratios):.3f}")
    
    # Print sample batch details
    print(f"\nSample batch compositions:")
    for i, stats in enumerate(batch_stats[:3]):
        print(f"\n  Batch {i+1}:")
        for cls in ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']:
            count = stats['classes'].get(cls, 0)
            print(f"    {cls:15s}: {count:2d}")


def main():
    print("="*70)
    print("CLASS-AWARE SAMPLER TEST")
    print("="*70)
    
    # Load dataset
    metadata_path = 'data/processed/fold_0_metadata.json'
    patches_dir = 'data/processed/patches'
    
    print("\nLoading dataset...")
    dataset = GrainDataset(
        metadata_path=metadata_path,
        patches_dir=patches_dir,
        split='train'
    )
    
    batch_size = 32
    
    # Test 1: Standard random sampling (baseline)
    print(f"\n{'='*70}")
    print("TEST 1: STANDARD RANDOM SAMPLING (BASELINE)")
    print(f"{'='*70}")
    
    standard_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    standard_stats = analyze_batch_composition(standard_loader, dataset, num_batches=20)
    print_sampler_analysis("Standard Random Sampling", standard_stats)
    
    # Test 2: Hierarchical Balanced Sampler
    print(f"\n{'='*70}")
    print("TEST 2: HIERARCHICAL BALANCED SAMPLER (WEIGHTED)")
    print(f"{'='*70}")
    
    balanced_sampler = HierarchicalBalancedSampler(
        dataset,
        samples_per_epoch=len(dataset) * 2,
        stage1_balance=0.5,
        stage3_broken_weight=5.0
    )
    
    balanced_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=balanced_sampler
    )
    
    balanced_stats = analyze_batch_composition(balanced_loader, dataset, num_batches=20)
    print_sampler_analysis("Hierarchical Balanced Sampler", balanced_stats)
    
    # Test 3: Stage-wise Batch Sampler
    print(f"\n{'='*70}")
    print("TEST 3: STAGE-WISE BATCH SAMPLER (BATCH-LEVEL BALANCE)")
    print(f"{'='*70}")
    
    stagewise_sampler = StageWiseBatchSampler(
        dataset,
        batch_size=batch_size,
        peloid_ratio=0.5,
        broken_ooid_oversample=3
    )
    
    stagewise_loader = DataLoader(
        dataset,
        batch_sampler=stagewise_sampler
    )
    
    stagewise_stats = analyze_batch_composition(stagewise_loader, dataset, num_batches=20)
    print_sampler_analysis("Stage-wise Batch Sampler", stagewise_stats)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    samplers = {
        'Standard': standard_stats,
        'Weighted': balanced_stats,
        'Stagewise': stagewise_stats
    }
    
    print(f"\n{'Sampler':<15} {'Peloid %':<12} {'Peloid Std':<12} {'Broken Ooid %':<15}")
    print("-" * 70)
    
    for name, stats in samplers.items():
        all_classes = Counter()
        peloid_ratios = []
        
        for s in stats:
            all_classes.update(s['classes'])
            total = s['total']
            peloid_count = s['classes'].get('Peloid', 0)
            peloid_ratio = peloid_count / total if total > 0 else 0
            peloid_ratios.append(peloid_ratio)
        
        peloid_pct = 100 * all_classes['Peloid'] / sum(all_classes.values())
        peloid_std = np.std(peloid_ratios)
        broken_pct = 100 * all_classes.get('Broken ooid', 0) / sum(all_classes.values())
        
        print(f"{name:<15} {peloid_pct:>6.1f}%     {peloid_std:>6.3f}      {broken_pct:>6.1f}%")
    
    print("\n" + "="*70)
    print("✅ Sampler test complete!")
    print("="*70)


if __name__ == '__main__':
    main()
