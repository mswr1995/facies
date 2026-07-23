"""
Prepare masked 96x96 patches and 60/20/20 splits for all seven labels.

Output is written to data/processed_all_classes so the original four-class
paper dataset remains unchanged.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_all_classes import ALL_CLASS_NAMES
from src.data.labelme_loader import (
    load_all_annotations,
    load_image_from_labelme,
    load_labelme_json,
    extract_grain_patch,
)


def prepare_all_classes(data_dir: Path, output_dir: Path, patch_size: int, random_state: int):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    patches_dir = output_dir / 'patches'
    patches_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARING SEVEN-CLASS DATASET")
    print("=" * 80)

    annotations = load_all_annotations(data_dir, filter_classes=False)
    metadata = []
    grain_id = 0

    for image_name in tqdm(sorted(annotations.keys()), desc="Extracting patches"):
        json_path = data_dir / f"{image_name}.json"
        json_data = load_labelme_json(json_path)
        image = load_image_from_labelme(json_data, data_dir)

        for grain_idx, grain in enumerate(annotations[image_name]):
            if grain['label'] not in ALL_CLASS_NAMES:
                raise ValueError(f"Unexpected label {grain['label']!r} in {image_name}")

            patch = extract_grain_patch(
                image,
                grain,
                patch_size=patch_size,
                with_mask=True,
            )
            patch_filename = f"grain_{grain_id:06d}.png"
            cv2.imwrite(str(patches_dir / patch_filename), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            metadata.append({
                'grain_id': grain_id,
                'image_name': image_name,
                'grain_idx': grain_idx,
                'label': grain['label'],
                'centroid': grain['centroid'],
                'patch_filename': patch_filename,
            })
            grain_id += 1

    with open(output_dir / 'grain_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    labels = [g['label'] for g in metadata]
    print("\nFull dataset:")
    for label in ALL_CLASS_NAMES:
        count = labels.count(label)
        print(f"  {label:15s}: {count:4d} ({100 * count / len(labels):5.2f}%)")

    rng = np.random.default_rng(random_state)
    train_grains = []
    val_grains = []
    test_grains = []

    for label in ALL_CLASS_NAMES:
        class_grains = [g for g in metadata if g['label'] == label]
        rng.shuffle(class_grains)
        n_total = len(class_grains)

        n_train = int(round(n_total * 0.6))
        n_val = int(round(n_total * 0.2))
        n_test = n_total - n_train - n_val

        # Keep every class represented in validation and test when possible.
        if n_total >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_test == 0:
                n_test = 1
                n_train -= 1

        train_grains.extend(class_grains[:n_train])
        val_grains.extend(class_grains[n_train:n_train + n_val])
        test_grains.extend(class_grains[n_train + n_val:])

    rng.shuffle(train_grains)
    rng.shuffle(val_grains)
    rng.shuffle(test_grains)

    for split_name, grains in [('train', train_grains), ('val', val_grains), ('test', test_grains)]:
        split_path = output_dir / f'{split_name}_split.json'
        with open(split_path, 'w') as f:
            json.dump({
                'split': split_name,
                'num_samples': len(grains),
                'grains': grains,
            }, f, indent=2)

    print("\nSplit distribution:")
    for split_name, grains in [('Train', train_grains), ('Val', val_grains), ('Test', test_grains)]:
        counts = Counter(g['label'] for g in grains)
        print(f"\n{split_name}: {len(grains)}")
        for label in ALL_CLASS_NAMES:
            print(f"  {label:15s}: {counts[label]:4d}")

    print(f"\nSaved seven-class dataset to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/processed_all_classes')
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    prepare_all_classes(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        patch_size=args.patch_size,
        random_state=args.random_state,
    )
