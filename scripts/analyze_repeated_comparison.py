"""Print a compact summary for repeated-comparison results."""
import argparse
import json
from pathlib import Path


def fmt(metric):
    return f"{100 * metric['mean']:.2f}% +/- {100 * metric['std']:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results/repeated_comparison/summary.json')
    args = parser.parse_args()

    summary = json.load(open(Path(args.summary)))
    metrics = summary['metrics']
    print(f"Completed splits: {summary['num_completed']} {summary['seeds']}")
    print(f"Flat BA:          {fmt(metrics['flat_ba'])}")
    print(f"Hier BA no TTA:   {fmt(metrics['hier_ba_no_tta'])}")
    print(f"Hier BA TTA:      {fmt(metrics['hier_ba_tta'])}")
    print(f"Delta BA:         {fmt(metrics['delta_hier_tta_minus_flat_ba'])}")
    print(f"Flat OA:          {fmt(metrics['flat_oa'])}")
    print(f"Hier OA TTA:      {fmt(metrics['hier_oa_tta'])}")


if __name__ == '__main__':
    main()
