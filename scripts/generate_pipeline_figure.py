"""
Generate Figure 1: Data pipeline illustration.
Panel A — raw PPL micrograph crop
Panel B — same crop with LabelMe polygon annotations coloured by class
Panel C — 2×2 grid of extracted masked grain patches (one per class)
"""
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path

RAW_DIR      = Path('data/raw')
PATCHES_DIR  = Path('data/processed/patches')
META_PATH    = Path('data/processed/grain_metadata.json')
OUT_PATH     = Path('article/figs/fig1_pipeline.pdf')

IMAGE_NAME  = 'WT13-ES0023'
IMAGE_FILE  = RAW_DIR / 'WT13-ES0023.jpg'
JSON_FILE   = RAW_DIR / 'WT13-ES0023.json'

# Crop window — contains all 4 classes
CROP = (300, 250, 800, 650)   # x0, y0, x1, y1

CLASS_NAMES  = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
CLASS_COLORS = {
    'Peloid':      '#4C72B0',
    'Ooid':        '#DD8452',
    'Broken ooid': '#55A868',
    'Intraclast':  '#C44E52',
}

# ── load image ───────────────────────────────────────────────────────────────
img_bgr = cv2.imread(str(IMAGE_FILE))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

x0, y0, x1, y1 = CROP
crop_raw  = img_rgb[y0:y1, x0:x1]

# ── load annotations ─────────────────────────────────────────────────────────
with open(JSON_FILE) as f:
    ann = json.load(f)

# Collect polygons that fall (centroid) inside the crop window
poly_data = []   # (label, Nx2 array of points in crop coords)
for shape in ann['shapes']:
    pts = np.array(shape['points'])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    if x0 < cx < x1 and y0 < cy < y1:
        pts_crop = pts - np.array([x0, y0])
        poly_data.append((shape['label'], pts_crop))

# ── load good patch examples (one per class) ─────────────────────────────────
with open(META_PATH) as f:
    all_grains = json.load(f)

# Pick the best representative per class from the whole dataset
# Prefer grains from WT13-ES0023 for consistency; fall back to any grain.
chosen_patches = {}
preferred = [g for g in all_grains if IMAGE_NAME in g['image_name']]
fallback  = all_grains

for cls in CLASS_NAMES:
    cands = [g for g in preferred if g['label'] == cls]
    if not cands:
        cands = [g for g in fallback if g['label'] == cls]
    # Pick a patch that isn't mostly black (at least 20% non-zero pixels)
    np.random.seed(42)
    np.random.shuffle(cands)
    for g in cands:
        p = cv2.imread(str(PATCHES_DIR / g['patch_filename']))
        if p is None:
            continue
        nonzero = np.count_nonzero(p.sum(axis=2))
        if nonzero / (p.shape[0] * p.shape[1]) > 0.20:
            chosen_patches[cls] = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            break
    if cls not in chosen_patches:
        # last resort: just take the first valid one
        for g in cands:
            p = cv2.imread(str(PATCHES_DIR / g['patch_filename']))
            if p is not None:
                chosen_patches[cls] = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                break

# ── figure layout ─────────────────────────────────────────────────────────────
# Width ratios: raw | annotated | patches-grid
fig = plt.figure(figsize=(15, 4.5))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(
    2, 4,
    width_ratios=[2.5, 2.5, 1.1, 1.1],
    height_ratios=[1, 1],
    wspace=0.06,
    hspace=0.10,
)

ax_raw  = fig.add_subplot(gs[:, 0])   # full left column
ax_ann  = fig.add_subplot(gs[:, 1])   # full middle column
# 4 individual patch subplots in a 2×2 grid
ax_patches = [
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[1, 3]),
]

# ── Panel A: raw crop ────────────────────────────────────────────────────────
ax_raw.imshow(crop_raw)
ax_raw.axis('off')
ax_raw.text(-0.04, 1.02, 'A', transform=ax_raw.transAxes,
            fontsize=14, fontweight='bold', va='top')

# ── Panel B: annotated crop ───────────────────────────────────────────────────
ax_ann.imshow(crop_raw)

for label, pts in poly_data:
    color = CLASS_COLORS.get(label, '#888888')
    poly = plt.Polygon(pts, closed=True,
                       edgecolor=color, facecolor=color,
                       linewidth=1.2, alpha=0.30)
    ax_ann.add_patch(poly)
    outline = plt.Polygon(pts, closed=True,
                          edgecolor=color, facecolor='none',
                          linewidth=1.4, alpha=0.85)
    ax_ann.add_patch(outline)

# Legend inside Panel B
legend_handles = [
    mpatches.Patch(facecolor=CLASS_COLORS[c], edgecolor='white',
                   linewidth=0.5, label=c, alpha=0.85)
    for c in CLASS_NAMES
]
ax_ann.legend(handles=legend_handles, loc='lower right',
              fontsize=7.5, framealpha=0.75, edgecolor='#cccccc',
              borderpad=0.5, labelspacing=0.25)
ax_ann.axis('off')
ax_ann.text(-0.04, 1.02, 'B', transform=ax_ann.transAxes,
            fontsize=14, fontweight='bold', va='top')

# ── Panels C–F: 2×2 patch grid ────────────────────────────────────────────────
for i, cls in enumerate(CLASS_NAMES):
    ax = ax_patches[i]
    patch = chosen_patches.get(cls)
    if patch is not None:
        ax.imshow(patch)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(CLASS_COLORS[cls])
        spine.set_linewidth(2.2)
    ax.set_title(cls, fontsize=7.5, fontweight='bold',
                 color=CLASS_COLORS[cls], pad=2)

# Panel C label above the top-left patch
ax_patches[0].text(-0.12, 1.18, 'C', transform=ax_patches[0].transAxes,
                   fontsize=14, fontweight='bold', va='top')

# ── Flow arrows between panels (via a transparent overlay axes) ──────────────
ax_overlay = fig.add_axes([0, 0, 1, 1], facecolor='none')
ax_overlay.set_xlim(0, 1)
ax_overlay.set_ylim(0, 1)
ax_overlay.axis('off')
ax_overlay.annotate('', xy=(0.376, 0.50), xytext=(0.358, 0.50),
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8))
ax_overlay.annotate('', xy=(0.672, 0.50), xytext=(0.654, 0.50),
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8))

plt.savefig(str(OUT_PATH), bbox_inches='tight', dpi=300)
plt.close()
print(f'Saved {OUT_PATH}')
