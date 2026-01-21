"""Create proper train/val/test split at grain level with no overlap."""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Set seed for reproducibility
np.random.seed(42)

print("="*80)
print("CREATING GRAIN-LEVEL TRAIN/VAL/TEST SPLIT")
print("="*80)

# Load all grains from grain_metadata.json
with open('data/processed/grain_metadata.json', 'r') as f:
    all_grains = json.load(f)
print(f"\nTotal grains: {len(all_grains)}")

# Check class distribution
labels = [g['label'] for g in all_grains]
label_counts = Counter(labels)
print(f"\nClass distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:15s}: {count:4d} ({100*count/len(labels):.2f}%)")

# Stratified split: 60% train, 20% val, 20% test
print(f"\nCreating splits: 60% train, 20% val, 20% test")

# First split: 60% train, 40% (val+test)
train_grains, temp_grains = train_test_split(
    all_grains,
    test_size=0.4,
    stratify=labels,
    random_state=42
)

# Second split: 50% of remaining = 20% val, 20% test
temp_labels = [g['label'] for g in temp_grains]
val_grains, test_grains = train_test_split(
    temp_grains,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_grains)} samples ({100*len(train_grains)/len(all_grains):.1f}%)")
print(f"  Val:   {len(val_grains)} samples ({100*len(val_grains)/len(all_grains):.1f}%)")
print(f"  Test:  {len(test_grains)} samples ({100*len(test_grains)/len(all_grains):.1f}%)")

# Verify no overlap
train_ids = set(g['grain_id'] for g in train_grains)
val_ids = set(g['grain_id'] for g in val_grains)
test_ids = set(g['grain_id'] for g in test_grains)

overlap_train_val = train_ids.intersection(val_ids)
overlap_train_test = train_ids.intersection(test_ids)
overlap_val_test = val_ids.intersection(test_ids)

print(f"\nOverlap check:")
print(f"  Train ∩ Val:  {len(overlap_train_val)} samples")
print(f"  Train ∩ Test: {len(overlap_train_test)} samples")
print(f"  Val ∩ Test:   {len(overlap_val_test)} samples")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("  ✅ No overlap - clean split!")
else:
    print("  ⚠️  ERROR: Overlap detected!")

# Check class distribution in each split
print(f"\nPer-split class distribution:")
for split_name, split_grains in [('Train', train_grains), ('Val', val_grains), ('Test', test_grains)]:
    split_labels = [g['label'] for g in split_grains]
    split_counts = Counter(split_labels)
    print(f"\n{split_name}:")
    for label in ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']:
        count = split_counts.get(label, 0)
        pct = 100 * count / len(split_grains)
        print(f"  {label:15s}: {count:4d} ({pct:5.2f}%)")

# Save new metadata
output_data = {
    'split_strategy': 'grain_level',
    'split_ratio': '60/20/20',
    'random_seed': 42,
    'total_grains': len(all_grains),
    'train_grains': train_grains,
    'val_grains': val_grains,
    'test_grains': test_grains
}

output_path = 'data/processed/train_val_test_split.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Split saved to: {output_path}")

# Also create separate files for convenience
for split_name, split_grains in [('train', train_grains), ('val', val_grains), ('test', test_grains)]:
    split_data = {
        'split': split_name,
        'num_samples': len(split_grains),
        'grains': split_grains
    }
    split_path = f'data/processed/{split_name}_split.json'
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"   {split_name}: {split_path}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Update dataset loader to use new split files")
print("2. Retrain model on new training set")
print("3. Tune thresholds on validation set")
print("4. Final evaluation on test set (never touched)")
print("\n" + "="*80)
