"""Create repeated stratified train/val/test splits for the 4-class dataset."""
import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


CLASS_NAMES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']


def load_all_grains(source_dir: Path):
    grains = []
    seen = set()
    for split in ['train', 'val', 'test']:
        with open(source_dir / f'{split}_split.json', 'r') as f:
            split_grains = json.load(f)['grains']
        for grain in split_grains:
            key = grain.get('grain_id', grain.get('patch_filename'))
            if key not in seen:
                seen.add(key)
                grains.append(grain)
    return grains


def write_split(path: Path, split_name: str, grains, seed: int) -> None:
    payload = {
        'split': split_name,
        'random_seed': seed,
        'grains': grains,
        'class_distribution': dict(Counter(g['label'] for g in grains)),
    }
    with open(path / f'{split_name}_split.json', 'w') as f:
        json.dump(payload, f, indent=2)


def make_split(grains, seed: int):
    labels = [g['label'] for g in grains]
    train_grains, temp_grains, _, temp_labels = train_test_split(
        grains,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=seed,
    )
    val_grains, test_grains = train_test_split(
        temp_grains,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_grains, val_grains, test_grains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='data/processed')
    parser.add_argument('--output_dir', default='data/repeated_splits')
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--seed_start', type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grains = load_all_grains(source_dir)
    print(f'Loaded {len(grains)} unique grains from {source_dir}')
    print('Full distribution:', dict(Counter(g['label'] for g in grains)))

    for offset in range(args.num_splits):
        seed = args.seed_start + offset
        split_dir = output_dir / f'seed_{seed:03d}'
        split_dir.mkdir(parents=True, exist_ok=True)

        train_grains, val_grains, test_grains = make_split(grains, seed)
        write_split(split_dir, 'train', train_grains, seed)
        write_split(split_dir, 'val', val_grains, seed)
        write_split(split_dir, 'test', test_grains, seed)

        print(f'\nseed_{seed:03d}')
        for name, split_grains in [('train', train_grains), ('val', val_grains), ('test', test_grains)]:
            print(f'  {name:5s}: {len(split_grains):4d} {dict(Counter(g["label"] for g in split_grains))}')


if __name__ == '__main__':
    main()
