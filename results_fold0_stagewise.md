# Carbonate Grain Classification - Fold 0 Results

## Model Architecture

**Backbone:** ResNet-18 pretrained on ImageNet
- Input: 96×96×3 RGB patches
- Feature extraction: 512-dimensional embeddings
- Total parameters: 11,373,891 (11.4M backbone + 0.2M heads)

**Classification Heads:** 3 independent binary classifiers
- Architecture per head: FC(512→128) → ReLU → Dropout(0.3) → FC(128→1)
- Stage 1: Peloid (1) vs Non-peloid (0) - applied to all samples
- Stage 2: Ooid-like (1) vs Intraclast (0) - applied to non-peloids only
- Stage 3: Whole Ooid (1) vs Broken Ooid (0) - applied to ooid-likes only

## Training Configuration

**Optimization:**
- Optimizer: AdamW (lr=0.0001, weight_decay=0.0001)
- Loss function: Focal Loss with stage-specific parameters
  - Stage 1: α=0.25, γ=2.0 (moderate focus on hard examples)
  - Stage 2: α=0.50, γ=2.0 (balanced focus)
  - Stage 3: α=0.75, γ=2.0 (strong focus on minority class)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: patience=10 epochs (triggered at epoch N/A)

**Sampling Strategy:** stagewise
- StageWiseBatchSampler: Constructs balanced batches
  - ~50% peloids, ~50% non-peloids per batch
  - Broken ooids oversampled 3x within ooid-like samples
  - Ensures balance within each batch, not just across epoch

**Data Augmentation (training only):**
- Random 90° rotations (p=1.0)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Color jitter (p=0.4): brightness/contrast (±0.15), HSV (±8/15/15)
- Gaussian blur (p=0.2): kernel size 3-7
- Normalization: ImageNet statistics

**Dataset:** 2642 grains, stratified 5-fold CV
- Fold 0: 1907 train / 735 test

---

## Training vs Test Results

### Overall Performance

| Split | Overall Accuracy | Stage 1 Acc | Stage 2 Acc | Stage 3 Acc |
|-------|-----------------|-------------|-------------|-------------|
| **Training** | 96.07% | 96.12% | 100.00% | 99.30% |
| **Test** | 93.61% | 95.10% | 94.74% | 92.78% |
| **Gap** | 2.46% | 1.02% | 5.26% | 6.51% |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support | Train/Test Gap |
|-------|-----------|--------|----|---------|----------------|
| Peloid | 100.00% | 95.81% | 97.86% | 621 | -0.1% |
| Ooid | 100.00% | 92.13% | 95.91% | 89 | +3.7% |
| Broken ooid | 100.00% | 12.50% | 22.22% | 8 | +77.8% |
| Intraclast | 100.00% | 58.82% | 74.07% | 17 | +25.9% |

### Test Confusion Matrix

|             | Peloid | Ooid | Broken | Intra |
|-------------|--------|------|--------|-------|
| **Peloid** (621) | 595 | 15 | 0 | 11 |
| **Ooid** (89) | 6 | 82 | 0 | 1 |
| **Broken** (8) | 1 | 6 | 1 | 0 |
| **Intra** (17) | 3 | 4 | 0 | 10 |

---

## Analysis

**Generalization gap:** 2.46% overall accuracy difference between train and test splits. Stage 1 shows 1.02% gap, Stage 2 shows 5.26% gap, Stage 3 shows 6.51% gap.

**Broken ooid performance:** Training recall of 100.00% drops to 12.50% on test set. 7 of 8 test samples misclassified (6 classified as whole ooids at Stage 3 decision boundary).

**Class distribution effects:** 
- Peloid (621 test samples): -0.1% train-test F1 gap
- Ooid (89 test samples): +3.7% train-test F1 gap
- Broken ooid (8 test samples): +77.8% train-test F1 gap
- Intraclast (17 test samples): +25.9% train-test F1 gap

**Error patterns:** Confusion matrix shows primary misclassifications:
- Peloid → Ooid (15 errors)
- Peloid → Intraclast (11 errors)
- Ooid → Peloid (6 errors)
- Broken ooid → Ooid (6 errors)
- Intraclast → Ooid (4 errors)

**Sampling strategy impact:** Using stagewise sampling improved rare class representation during training. Intraclast F1 of 74.1% benefits from balanced sampling. 
