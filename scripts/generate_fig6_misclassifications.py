"""
Figure 6 — Peloid–Intraclast misclassification examples.

Identifies test-set grains where the final model confuses Peloid and Intraclast
in either direction, loads their raw patches, and saves a panel figure.
"""
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.models.hierarchical_model import HierarchicalGrainClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'checkpoints/exp_combined/best_model.pth'
PATCHES_DIR = Path('data/processed/patches')
OUT = Path('article/figs/fig6_misclassifications.pdf')

LABEL_MAP = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
CLASS_NAMES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']


def apply_tta(img):
    return torch.stack([
        img,
        torch.flip(img, dims=[2]),
        torch.flip(img, dims=[1]),
        torch.rot90(img, k=1, dims=[1, 2]),
        torch.rot90(img, k=2, dims=[1, 2]),
        torch.rot90(img, k=3, dims=[1, 2]),
    ])


def predict(model, img):
    aug = apply_tta(img).to(DEVICE)
    with torch.no_grad():
        logits = model(aug)
    p1 = torch.sigmoid(logits['stage1'].mean()).item()
    if p1 > 0.5:
        return 0
    p2 = torch.sigmoid(logits['stage2'].mean()).item()
    if p2 < 0.5:
        return 3
    p3 = torch.sigmoid(logits['stage3'].mean()).item()
    return 1 if p3 > 0.5 else 2


def load_model():
    model = HierarchicalGrainClassifier(pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    print('Loading model...')
    model = load_model()

    test_ds = GrainDatasetNew(split='test', use_default_transforms=True)

    pel_as_intra = []   # true=Peloid, pred=Intraclast
    intra_as_pel = []   # true=Intraclast, pred=Peloid

    print('Running inference on test set...')
    for i in range(len(test_ds)):
        img, _, meta = test_ds[i]
        true_idx = LABEL_MAP[meta['label']]
        pred_idx = predict(model, img)

        patch_filename = test_ds.samples[i]['patch_filename']
        if true_idx == 0 and pred_idx == 3:
            pel_as_intra.append(patch_filename)
        elif true_idx == 3 and pred_idx == 0:
            intra_as_pel.append(patch_filename)

    print(f'  Peloid → Intraclast: {len(pel_as_intra)} grains')
    print(f'  Intraclast → Peloid: {len(intra_as_pel)} grains')

    # Take up to 4 from the larger group, all from the smaller
    n_cols = 4
    rows = [
        ('True: Peloid  /  Predicted: Intraclast', pel_as_intra[:n_cols], '#4C72B0'),
        ('True: Intraclast  /  Predicted: Peloid', intra_as_pel[:n_cols], '#C44E52'),
    ]

    # Determine actual column count (use the max we have examples for)
    max_cols = max(min(len(pel_as_intra), n_cols), min(len(intra_as_pel), n_cols))
    if max_cols == 0:
        print('No misclassifications found — check model and thresholds.')
        return

    fig, axes = plt.subplots(2, max_cols, figsize=(max_cols * 2.4, 2 * 2.6))
    fig.patch.set_facecolor('white')

    for row_idx, (title, filenames, color) in enumerate(rows):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(filenames):
                img_path = PATCHES_DIR / filenames[col_idx]
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            else:
                ax.axis('off')

        axes[row_idx, 0].set_ylabel(title, fontsize=9.5, fontweight='bold',
                                    color=color, labelpad=8)

    fig.suptitle(
        'Peloid\u2013Intraclast confusions in the final model (test set)\n'
        'Border colour indicates the error direction',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
