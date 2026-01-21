# Codebase and Model Architecture Analysis

## Project Structure

```
src/
├── data/
│   ├── dataset.py (old fold-based CV)
│   ├── dataset_new.py (NEW: clean train/val/test split)
│   ├── labelme_loader.py
│   ├── preprocess.py
│   └── splits.py
├── models/
│   ├── focal_loss.py
│   └── hierarchical_model.py
├── training/
│   ├── trainer.py (main training class)
│   ├── metrics.py
│   └── utils.py
└── utils/

scripts/
├── train_new_split.py (NEW: training with clean split)
├── evaluate_new_split.py (NEW: evaluation with no leakage)
└── create_grain_split.py (NEW: creates 60/20/20 split)

data/processed/
├── grain_metadata.json (all 2,642 grains)
├── patches/ (96x96 grain images)
├── train_split.json (1,585 samples)
├── val_split.json (528 samples)
└── test_split.json (529 samples - never touched)
```

---

## Model Architecture

### HierarchicalGrainClassifier

**Backbone:** ResNet-18 (ImageNet pretrained)
- Input: 96×96 RGB images
- Output: 512-dim features (after global average pooling)

**Three Independent Binary Heads:**

```
Input (96×96)
     ↓
ResNet-18 Backbone (513.2M → 11.1M parameters)
     ↓
512-dim Features
     ├→ Head Stage 1: 512 → 128 → 1 (Peloid vs Non-peloid)
     ├→ Head Stage 2: 512 → 128 → 1 (Ooid-like vs Intraclast)
     └→ Head Stage 3: 512 → 128 → 1 (Whole vs Broken Ooid)
```

**Each Head Architecture:**
- Input: 512-dim features
- FC Layer 1: 512 → 128
- ReLU Activation
- Dropout(0.3)
- FC Layer 2: 128 → 1 (logit)

**Total Parameters:** 11,373,891
- Backbone: 11,176,512 (trainable)
- Stage 1 Head: 65,793
- Stage 2 Head: 65,793
- Stage 3 Head: 65,793

---

## Hierarchical Classification Logic

```
Stage 1: Is it Peloid?
  ├─ YES (P(peloid) > T1=0.5) → CLASS: Peloid
  └─ NO  → Go to Stage 2
     │
     Stage 2: Is it Ooid-like?
     ├─ YES (P(ooid_like) > T2=0.5) → Go to Stage 3
     │  │
     │  Stage 3: Is it Whole Ooid?
     │  ├─ YES (P(whole) > T3=0.5) → CLASS: Ooid
     │  └─ NO  → CLASS: Broken Ooid
     │
     └─ NO  (P(ooid_like) ≤ T2) → CLASS: Intraclast
```

---

## Training Pipeline

### Data Loading (dataset_new.py)
- **Train:** 1,585 samples (87.1% peloid, 8.0% ooid, 1.1% broken, 3.9% intraclast)
- **Val:** 528 samples (87.1% peloid, 8.0% ooid, 1.1% broken, 3.8% intraclast)
- **Test:** 529 samples (87.0% peloid, 7.9% ooid, 1.1% broken, 4.0% intraclast)

**Data Augmentation (train only):**
- Random 90° rotations (p=1.0)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Color jitter: brightness/contrast (±0.15), HSV (±8/15/15) with p=0.4

### Loss Functions (focal_loss.py)

**Focal Loss Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Per-Stage Configuration:**
- Stage 1 (Peloid vs Non-peloid): α=0.25, γ=2.0 (moderate focus on hard examples)
- Stage 2 (Ooid-like vs Intraclast): α=0.50, γ=2.0 (balanced focus)
- Stage 3 (Whole vs Broken Ooid): α=0.75, γ=2.0 (strong focus on minority class)

**Why Focal Loss:**
- Suppresses easy examples (majority class/easy negatives)
- Forces learning from rare and hard examples
- Better than weighted BCE for extreme class imbalance

### Optimizer & Scheduler (trainer.py)

**Optimizer:** AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-4 (L2 regularization)

**Scheduler:** ReduceLROnPlateau
- Monitor: Validation accuracy
- Mode: max (maximize accuracy)
- Factor: 0.5 (multiply LR by 0.5)
- Patience: 5 epochs (wait 5 epochs before reducing LR)

**Early Stopping:**
- Monitor: Validation accuracy
- Patience: 10 epochs (stop if no improvement for 10 epochs)
- Mode: max

### Training Loop (trainer.py::train_epoch)

1. **Forward Pass:**
   - Input images → ResNet backbone → 512-dim features
   - Feed to all 3 heads simultaneously
   - Get logits for each stage

2. **Loss Computation:**
   - Stage 1: FocalLoss(logit_s1, label_s1)
   - Stage 2: FocalLoss(logit_s2, label_s2) [only for non-peloids]
   - Stage 3: FocalLoss(logit_s3, label_s3) [only for ooid-likes]
   - Total Loss = Loss_s1 + Loss_s2 + Loss_s3

3. **Backward Pass & Optimization:**
   - Backpropagate total loss
   - AdamW update all parameters
   - Accumulate metrics

### Validation Loop (trainer.py::validate)

1. Evaluate on validation set (no augmentation)
2. Compute hierarchical metrics:
   - Stage-wise accuracy/F1
   - Overall accuracy (final class predictions)
3. Save best checkpoint if validation accuracy improves
4. Return metrics for scheduler & early stopping

---

## Key Features of Current Implementation

✅ **Clean Data Split:**
- 60/20/20 train/val/test with no grain overlap
- Stratified by class (maintains distribution)
- Test set completely held out

✅ **Hierarchical Training:**
- All 3 heads trained simultaneously
- Shared backbone learns grain features
- Independent heads for each decision

✅ **Imbalance Handling:**
- Focal Loss with stage-specific α (0.25 → 0.75 for rarer classes)
- Stratified sampling preserves class ratios
- Early stopping prevents overfitting

⚠️ **Known Limitations:**
- Rare classes (broken ooid 1.1%, intraclast 3.9%) still hard to detect
- Intraclast precision low (confusion with peloids and ooids)
- Broken ooid recall limited by small training set (17 samples)

---

## Previous Training Results

**Model trained on new 60/20/20 split:**
- Best validation accuracy: 93.75% (epoch 14)
- Stopped at epoch 24 (early stopping triggered)

**Key Metrics:**
- Peloid: 96.84% F1, 99.39% PR-AUC ✓ Excellent
- Ooid: 75.56% F1, 72.45% PR-AUC ✓ Good
- Broken ooid: 66.67% F1, 77.41% PR-AUC ⚠️ Moderate
- Intraclast: 68.75% F1, 54.78% PR-AUC ⚠️ Weak

---

## Recommendations for Improvement

1. **Threshold Tuning:**
   - Current: T1=T2=T3=0.5 (default)
   - Could optimize T2, T3 on validation set to improve rare classes

2. **Data Augmentation:**
   - Currently only train split gets augmentation
   - Consider stronger augmentation for rare classes

3. **Class Weighting:**
   - Could add sample weights to training
   - Dynamically weight rare class samples higher

4. **Alternative Architectures:**
   - Try EfficientNet instead of ResNet-18
   - Add attention mechanisms for rare classes

5. **Post-Processing:**
   - Use confidence scores for predictions
   - Ensemble with XGBoost on rare classes
