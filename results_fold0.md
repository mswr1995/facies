# Carbonate Grain Classification - Fold 0 Results

## Model Architecture

**Backbone:** ResNet-18 pretrained on ImageNet
- Input: 96×96×3 RGB patches
- Feature extraction: 512-dimensional embeddings
- Total parameters: 11,373,891 (11.2M backbone + 3×66K heads)

**Classification Heads:** 3 independent binary classifiers
- Architecture per head: FC(512→128) → ReLU → Dropout(0.3) → FC(128→1)
- Stage 1: Peloid (1) vs Non-peloid (0) - applied to all samples
- Stage 2: Ooid-like (1) vs Intraclast (0) - applied to non-peloids only
- Stage 3: Whole Ooid (1) vs Broken Ooid (0) - applied to ooid-likes only

## Training Configuration

**Optimization:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Loss function: Focal Loss with stage-specific parameters
  - Stage 1: α=0.25, γ=2.0 (moderate focus on hard examples)
  - Stage 2: α=0.50, γ=2.0 (balanced focus)
  - Stage 3: α=0.75, γ=2.0 (strong focus on minority class)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: patience=10 epochs (triggered at epoch 33)

**Data Augmentation (training only):**
- Random 90° rotations (p=1.0)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Color jitter (p=0.4): brightness/contrast (±0.15), HSV (±8/15/15)
- Gaussian blur (p=0.2): kernel size 3-7
- Normalization: ImageNet statistics

**Dataset:** 2,642 grains, stratified 5-fold CV
- Fold 0: 1,907 train / 735 test
- Class distribution: Peloid (87%), Ooid (8%), Intraclast (4%), Broken ooid (1.1%)

---

## Training vs Test Results

### Overall Performance

| Split | Overall Accuracy | Stage 1 Acc | Stage 2 Acc | Stage 3 Acc |
|-------|-----------------|-------------|-------------|-------------|
| **Training** | 96.22% | 96.59% | 96.93% | 99.30% |
| **Test** | 93.33% | 94.69% | 96.49% | 92.78% |
| **Gap** | 2.89% | 1.90% | 0.44% | 6.52% |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support | Train/Test Gap |
|-------|-----------|--------|----|---------|----------------|
| Peloid | 96.78% | 96.94% | 96.86% | 621 | -1.1% |
| Ooid | 81.11% | 82.02% | 81.56% | 89 | -10.0% |
| Broken ooid | 100.00% | 12.50% | 22.22% | 8 | -86.8% |
| Intraclast | 45.45% | 58.82% | 51.28% | 17 | -28.1% |

### Test Confusion Matrix

|             | Peloid | Ooid | Broken | Intra |
|-------------|--------|------|--------|-------|
| **Peloid** (621) | 602 | 8 | 0 | 11 |
| **Ooid** (89) | 15 | 73 | 0 | 1 |
| **Broken** (8) | 1 | 6 | 1 | 0 |
| **Intra** (17) | 4 | 3 | 0 | 10 |

---

## Analysis

**Generalization gap:** 2.89% overall accuracy difference between train and test splits. Stage 1 and Stage 2 show <2% gaps, while Stage 3 shows 6.52% gap.

**Broken ooid performance:** Training accuracy of 99.30% drops to 92.78% at stage level, with final recall of 12.5% (1/8 correct). 6 of 8 test samples misclassified as whole ooids at Stage 3 decision boundary.

**Class distribution effects:** 
- Peloid (621 test samples): 1.1% train-test gap
- Ooid (89 test samples): 10.0% train-test gap  
- Broken ooid (8 test samples): 86.8% train-test gap
- Intraclast (17 test samples): 28.1% train-test gap

**Error patterns:** Confusion matrix shows primary misclassifications:
- Peloid → Intraclast (11 errors)
- Ooid → Peloid (15 errors)
- Broken ooid → Ooid (6 errors)
- Intraclast → Peloid (4 errors)
