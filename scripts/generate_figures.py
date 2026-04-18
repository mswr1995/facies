"""
Generate all paper figures:
  Figure 2 — Sample grain patches (4 classes × 3 examples)
  Figure 3 — Training curve across 4 phases
  Figure 4 — Per-class Precision-Recall curves
  Figure 5 — t-SNE of backbone latent space
"""
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cv2
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.models.hierarchical_model import HierarchicalGrainClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'checkpoints/exp_combined/best_model.pth'
OUT_DIR = Path('article/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
CLASS_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
LABEL_MAP    = {name: i for i, name in enumerate(CLASS_NAMES)}

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# Figure 2 — Sample grain patches
# ─────────────────────────────────────────────
def figure_sample_patches():
    print('Generating Figure 2: sample patches...')
    patches_dir = Path('data/processed/patches')

    # Load train split JSON directly (no transforms, raw images)
    with open('data/processed/train_split.json') as f:
        train_data = json.load(f)
    grains = train_data['grains']

    # Group by class
    by_class = {c: [] for c in CLASS_NAMES}
    for g in grains:
        by_class[g['label']].append(g)

    N_COLS = 3
    fig, axes = plt.subplots(4, N_COLS, figsize=(N_COLS * 2.2, 4 * 2.2))
    fig.patch.set_facecolor('white')

    for row, cls in enumerate(CLASS_NAMES):
        candidates = by_class[cls]
        chosen = random.sample(candidates, min(N_COLS, len(candidates)))

        for col, grain in enumerate(chosen):
            ax = axes[row, col]
            img_path = patches_dir / grain['patch_filename']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight='bold',
                              labelpad=6, color=CLASS_COLORS[row])
        # blank remaining cells
        for col in range(len(chosen), N_COLS):
            axes[row, col].axis('off')

    fig.suptitle('Representative masked grain patches by class', fontsize=12, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / 'fig2_sample_patches.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────
# Figure 3 — Training curve across phases
# ─────────────────────────────────────────────
def figure_training_curve():
    print('Generating Figure 3: training curve...')
    with open('checkpoints/exp_combined/training_history.json') as f:
        hist = json.load(f)['history']

    epochs    = list(range(1, len(hist) + 1))
    val_ba    = [h['val_ba'] for h in hist]
    best_ba   = max(val_ba)
    best_ep   = val_ba.index(best_ba) + 1

    # Phase boundaries (phase1=10, phase2=15, phase3=15, phase4=10 → cumulative)
    boundaries = [10, 25, 40]
    phase_labels = ['Phase 1\n(frozen backbone,\nall heads)',
                    'Phase 2\n(backbone + Head 1)',
                    'Phase 3\n(Heads 2+3 only)',
                    'Phase 4\n(full fine-tune)']
    phase_starts = [1, 11, 26, 41]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('white')

    colors_phase = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5CBA7']
    prev = 0
    for i, b in enumerate(boundaries + [50]):
        ax.axvspan(prev, b, alpha=0.18, color=colors_phase[i], zorder=0)
        mid = (prev + b) / 2
        ax.text(mid, 0.91, phase_labels[i], ha='center', va='top',
                fontsize=7.5, color='#444444',
                transform=ax.get_xaxis_transform())
        prev = b

    for b in boundaries:
        ax.axvline(b, color='#888888', lw=0.8, linestyle='--', zorder=1)

    ax.plot(epochs, val_ba, color='#2C3E50', lw=1.8, zorder=2, label='Val Balanced Accuracy')
    ax.scatter([best_ep], [best_ba], color='#E74C3C', s=60, zorder=3,
               label=f'Best BA = {best_ba:.3f} (ep {best_ep})')

    ax.set_xlabel('Training Epoch', fontsize=11)
    ax.set_ylabel('Validation Balanced Accuracy', fontsize=11)
    ax.set_title('Staged training progression across four phases', fontsize=12)
    ax.set_xlim(0.5, 50.5)
    ax.set_ylim(0.55, 0.88)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / 'fig3_training_curve.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────
# Figure 4 — Per-class Precision-Recall curves
# ─────────────────────────────────────────────
def figure_pr_curves(model):
    print('Generating Figure 4: PR curves...')
    model.eval()
    test_ds = GrainDatasetNew(split='test', use_default_transforms=True)

    # Collect sigmoid scores for each stage head and true labels
    all_s1, all_s2, all_s3, all_true = [], [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, labels, meta = test_ds[i]
            logits = model(img.unsqueeze(0).to(DEVICE))
            all_s1.append(torch.sigmoid(logits['stage1']).item())
            all_s2.append(torch.sigmoid(logits['stage2']).item())
            all_s3.append(torch.sigmoid(logits['stage3']).item())
            all_true.append(LABEL_MAP[meta['label']])

    all_true = np.array(all_true)
    s1 = np.array(all_s1)
    s2 = np.array(all_s2)
    s3 = np.array(all_s3)

    # Build per-class probability scores from hierarchical outputs:
    # P(Peloid)      = s1
    # P(Intraclast)  = (1-s1) * (1-s2)
    # P(Ooid)        = (1-s1) * s2 * s3
    # P(Broken ooid) = (1-s1) * s2 * (1-s3)
    scores = {
        'Peloid':       s1,
        'Intraclast':   (1 - s1) * (1 - s2),
        'Ooid':         (1 - s1) * s2 * s3,
        'Broken ooid':  (1 - s1) * s2 * (1 - s3),
    }

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for i, cls in enumerate(CLASS_NAMES):
        binary_true = (all_true == i).astype(int)
        score = scores[cls]
        prec, rec, _ = precision_recall_curve(binary_true, score)
        ap = average_precision_score(binary_true, score)
        n_pos = binary_true.sum()

        ax = axes[i]
        ax.plot(rec, prec, color=CLASS_COLORS[i], lw=2)
        ax.fill_between(rec, prec, alpha=0.12, color=CLASS_COLORS[i])
        ax.set_title(f'{cls}  (N={n_pos}, AP={ap:.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Recall', fontsize=9)
        ax.set_ylabel('Precision', fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        # Chance level
        chance = n_pos / len(all_true)
        ax.axhline(chance, color='#999999', lw=1, linestyle='--', label=f'Chance ({chance:.3f})')
        ax.legend(fontsize=7)

    fig.suptitle('Per-class Precision-Recall curves (test set, combined model + TTA thresholds)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / 'fig4_pr_curves.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────
# Figure 5 — t-SNE of backbone latent space
# ─────────────────────────────────────────────
def figure_tsne(model):
    print('Generating Figure 5: t-SNE (this takes a minute)...')
    model.eval()
    test_ds = GrainDatasetNew(split='test', use_default_transforms=True)

    features, labels_list = [], []

    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc='  extracting features', leave=False):
            img, _, meta = test_ds[i]
            h = model.backbone(img.unsqueeze(0).to(DEVICE))
            features.append(h.squeeze().cpu().numpy())
            labels_list.append(LABEL_MAP[meta['label']])

    features = np.array(features)
    labels_arr = np.array(labels_list)

    print('  Running t-SNE...')
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')

    markers = ['o', 's', '^', 'D']
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels_arr == i
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[i], marker=markers[i],
                   s=30 if i == 0 else 55,
                   alpha=0.55 if i == 0 else 0.85,
                   edgecolors='white' if i != 0 else 'none',
                   linewidths=0.5,
                   label=f'{cls} (n={mask.sum()})',
                   zorder=2 if i == 0 else 3)

    ax.set_title('t-SNE of ResNet-18 backbone representations\n(512-dim → 2-dim, test set)',
                 fontsize=11)
    ax.set_xlabel('t-SNE dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    ax.legend(fontsize=9, markerscale=1.2, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out = OUT_DIR / 'fig5_tsne.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    figure_sample_patches()
    figure_training_curve()

    print(f'Loading model from {CHECKPOINT}...')
    model = HierarchicalGrainClassifier(pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    figure_pr_curves(model)
    figure_tsne(model)

    print('\nAll figures saved to article/figs/')
