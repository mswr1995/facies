"""Run flat vs hierarchical comparison on repeated stratified splits."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_command(cmd):
    print('\n' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def summarize(results):
    rows = []
    for item in results:
        seed = item['seed']
        flat = item['flat']
        hier = item['hierarchical']
        rows.append({
            'seed': seed,
            'flat_ba': flat['test_ba'],
            'flat_oa': flat['test_oa'],
            'hier_ba_no_tta': hier['test_without_tta']['balanced_accuracy'],
            'hier_oa_no_tta': hier['test_without_tta']['overall_accuracy'],
            'hier_ba_tta': hier['test_with_tta']['balanced_accuracy'],
            'hier_oa_tta': hier['test_with_tta']['overall_accuracy'],
            'delta_hier_tta_minus_flat_ba': hier['test_with_tta']['balanced_accuracy'] - flat['test_ba'],
            'delta_hier_tta_minus_flat_oa': hier['test_with_tta']['overall_accuracy'] - flat['test_oa'],
        })

    metrics = {}
    for key in rows[0]:
        if key == 'seed':
            continue
        values = np.array([row[key] for row in rows], dtype=float)
        metrics[key] = {
            'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', default='data/repeated_splits')
    parser.add_argument('--output_dir', default='results/repeated_comparison')
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--flat_epochs', type=int, default=50)
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=15)
    parser.add_argument('--phase3_epochs', type=int, default=15)
    parser.add_argument('--phase4_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    for offset in range(args.num_splits):
        seed = args.seed_start + offset
        split_dir = splits_dir / f'seed_{seed:03d}'
        if not split_dir.exists():
            raise FileNotFoundError(f'Missing split directory: {split_dir}')

        seed_out = output_dir / f'seed_{seed:03d}'
        seed_out.mkdir(parents=True, exist_ok=True)
        flat_out = seed_out / 'flat_results.json'
        hier_out = seed_out / 'hierarchical_results.json'
        flat_ckpt = Path('checkpoints/repeated_comparison') / f'seed_{seed:03d}' / 'flat'
        hier_ckpt = Path('checkpoints/repeated_comparison') / f'seed_{seed:03d}' / 'hierarchical'

        if args.force or not flat_out.exists():
            run_command([
                sys.executable, 'scripts/flat_baseline.py',
                '--seed', str(seed),
                '--split_dir', str(split_dir),
                '--epochs', str(args.flat_epochs),
                '--batch_size', str(args.batch_size),
                '--checkpoint_dir', str(flat_ckpt),
                '--output', str(flat_out),
            ])
        else:
            print(f'Skipping flat seed_{seed:03d}; found {flat_out}')

        if args.force or not hier_out.exists():
            run_command([
                sys.executable, 'scripts/exp_combined.py',
                '--seed', str(seed),
                '--split_dir', str(split_dir),
                '--batch_size', str(args.batch_size),
                '--checkpoint_dir', str(hier_ckpt),
                '--output', str(hier_out),
                '--phase1_epochs', str(args.phase1_epochs),
                '--phase2_epochs', str(args.phase2_epochs),
                '--phase3_epochs', str(args.phase3_epochs),
                '--phase4_epochs', str(args.phase4_epochs),
            ])
        else:
            print(f'Skipping hierarchical seed_{seed:03d}; found {hier_out}')

        completed.append({
            'seed': seed,
            'flat': load_json(flat_out),
            'hierarchical': load_json(hier_out),
        })

        rows, metrics = summarize(completed)
        summary = {
            'num_completed': len(completed),
            'seeds': [item['seed'] for item in completed],
            'rows': rows,
            'metrics': metrics,
        }
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(output_dir / 'summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print('\nCurrent summary:')
        print(f"  flat BA:      {metrics['flat_ba']['mean']:.4f} +/- {metrics['flat_ba']['std']:.4f}")
        print(f"  hier TTA BA:  {metrics['hier_ba_tta']['mean']:.4f} +/- {metrics['hier_ba_tta']['std']:.4f}")
        print(f"  delta BA:     {metrics['delta_hier_tta_minus_flat_ba']['mean']:+.4f} +/- {metrics['delta_hier_tta_minus_flat_ba']['std']:.4f}")


if __name__ == '__main__':
    main()
