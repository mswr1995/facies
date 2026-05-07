# FaciesNet — Project Context for Claude

## What this project is

Scientific paper submission (Computers & Geosciences → moved to **Earth Science Informatics**, Springer) on automated carbonate microfacies grain classification using hierarchical deep learning. First author: Musawer Muradi (Kocaeli University, CS dept). Co-authors: Arnaud Gallois (geology, data/annotation) and Alev Mutlu (CS, review).

## Active working file

`article/245112075_v2.tex` — this is the submission draft. Do not edit `article/245112075.tex` (old version, kept for reference only).

LaTeX class: `cas-dc` (Elsevier CAS double-column). Compile with `pdflatex`.  
Current state: **0 errors, 3 non-critical warnings** (hyperref empty anchor, float specifier, BibTeX empty pages on a URL-only ref).

## Paper structure (6 sections)

1. Introduction — XGBoost pilot mentioned, aims/contributions stated
2. Related Work — neutral tone, no "our approach" language
3. Materials & Methods — geological setting, grain definitions, dataset, architecture, training
4. Results — ablation table + confusion matrix + PR curves
5. Discussion — role of each component, flat baseline caveat, t-SNE, Peloid–Intraclast confusion (§5.4 has placeholder — waiting on Arnaud), limitations
6. Conclusion

## Dataset

- **2,642 grains** from **18 PPL micrographs** of **9 thin sections** (Mupe Member, Purbeck Limestone Group, late Jurassic–early Cretaceous, Dorset, UK)
- Annotated in LabelMe by Arnaud Gallois (single annotator)
- Raw images + JSON annotations: `data/raw/`
- Processed patches (96×96 px, masked): `data/processed/patches/`
- Metadata: `data/processed/grain_metadata.json`
- Split files: `data/processed/train_split.json`, `val_split.json`, `test_split.json`

### Class distribution

| Class       | Count | %    |
|-------------|-------|------|
| Peloid      | 2,300 | 87.1 |
| Ooid        | 210   | 7.9  |
| Intraclast  | 103   | 3.9  |
| Broken ooid | 29    | 1.1  |
| **Total**   | **2,642** | |
| Excluded (ostracod/bivalve/quartz) | 45 | — |

Split: 60/20/20 stratified random, seed 42. Test set = 529 grains (Peloid 460, Ooid 42, Intraclast 21, Broken ooid 6).

## Model

**Hierarchical ResNet-18** — shared backbone (512-dim features) + 3 independent binary MLP heads:

| Stage | Decision | α (Focal Loss) |
|-------|----------|----------------|
| 1 | Peloid vs Non-Peloid | 0.25 |
| 2 | Ooid-like vs Intraclast | 0.50 |
| 3 | Intact vs Broken Ooid | 0.75 |

All heads: γ=2.0. MLP per head: Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1).

**Training:** 4-phase staged curriculum, 50 epochs total:
- Phase 1 (ep 1–10): frozen backbone, all heads
- Phase 2 (ep 11–25): backbone + Head 1
- Phase 3 (ep 26–40): Heads 2+3 only (best checkpoint selected here)
- Phase 4 (ep 41–50): full fine-tune

**Oversampling:** 15× for broken ooids per batch.  
**Inference-time augmentation (ITA):** 6 geometric orientations, predictions averaged.  
**Checkpoint:** `checkpoints/exp_combined/best_model.pth`

## Key results (test set, 529 grains)

| Model | BA | OA |
|-------|----|----|
| Flat ResNet-18 baseline | 86.9% | 95.5% |
| **Full hierarchical model** | **83.5%** | **92.1%** |

Flat baseline advantage is unreliable (6 broken ooid test instances). Hierarchical model intraclast recall: 71.4% vs 61.9%.

Ablation broken ooid recall: base 16.7% → +oversampling 66.7% → +curriculum 83.3%.

## Figures

| File | Figure | Content |
|------|--------|---------|
| `figs/fig1_pipeline.pdf` | Fig 1 | Pipeline: raw micrograph → annotation overlay → masked patches |
| `figs/fig2_sample_patches.pdf` | Fig 2 | 4×3 grain patch grid, A/B/C/D row labels |
| `figs/fig3_training_curve.pdf` | Fig 3 | Val BA across 50 epochs, 4-phase regions |
| `figs/fig4_pr_curves.pdf` | Fig 4 | Per-class PR curves (2×2 grid) |
| `figs/fig5_tsne.pdf` | Fig 5 | t-SNE of 512-dim backbone features |
| `figs/fig6_misclassifications.pdf` | Fig 6 | Peloid↔Intraclast confusion examples |

Scripts to regenerate: `scripts/generate_figures.py` (needs model checkpoint), `scripts/generate_pipeline_figure.py` (no model needed).

## References

`article/references.bib` — key entries added this revision:
- `dunham1962`, `embry1971`, `bathurst1971` — carbonate classification classics
- `gallois2016` — Arnaud's PhD thesis (data source)
- `gallois2018` — Arnaud's Facies journal paper (data source)

## Pending items

1. **Arnaud** — magnification + microscope model for §3.1 (placeholder comment in tex)
2. **Arnaud** — geological interpretation of Peloid–Intraclast confusion for §5.4 (placeholder comment in tex)
3. Scale bars on figures — blocked on magnification from Arnaud
4. Supplementary materials — 18 micrographs with LabelMe + model overlays (needs model inference run)

## Review materials

- `article/review/mode___title (1)_Arnaud_comments.pdf` — Arnaud's annotated PDF (17 pages)
- `article/review/Text to add from Arnaud.docx` — Arnaud's geological content (grain definitions, geological setting, sample table)

Both have been incorporated into v2.tex.

## Language rules

- No AI-sounding phrasing (no "it is worth noting", "in conclusion", "leveraging")
- No "our approach" / "we propose" in Related Work — neutral descriptions only
- Grain size units in μm (not mm) for microscopic grains
- Single annotator = Arnaud Gallois (A.G.)
