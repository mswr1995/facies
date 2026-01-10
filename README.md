# Hierarchical Classification Pipeline for Carbonate Grain Microfacies

This document is the **final, implementation-ready design** for a robust grain classification system tailored to **carbonate rock microfacies** with extreme class imbalance.

It is written to be:
- Directly usable in **VS Code**
- Clear enough to translate into code without ambiguity
- Scientifically defensible for a thesis or paper

---

## 1. Problem Definition

### Objective
Classify individual carbonate grains into:

- Peloid  
- Ooid  
- Broken ooid  
- Intraclast  

Other rare classes (bivalve, ostracod, quartz grain) are **filtered out during preprocessing**.

### Dataset Overview
- **18 images** with labelme polygon annotations
- **2,642 grains** after filtering (from 2,687 total)
- Extreme imbalance: 87% peloids, 1% broken ooids

---

## 2. Key Challenges

- Extreme class imbalance (≈85% peloids, ≈1% broken ooids)
- Subtle morphological differences
- Ambiguous class boundaries (especially broken ooids)
- Limited dataset size
- Pixel-level texture insufficient for grain identity

---

## 3. Core Design Principles

1. **Object-level classification**, not pixel-level
2. **Hierarchical binary decisions**, not flat multi-class softmax
3. **Shared feature learning**, specialized decision heads
4. **Class-aware sampling**, not random sampling
5. **Probabilistic outputs**, not forced hard labels
6. **Geological reasoning encoded into model structure**

---

## 4. Data Representation

### 4.1 Unit of Learning
Each training sample corresponds to **one annotated grain**.

---

### 4.2 Grain-Centered Patch Extraction

For each grain:

- Crop a square patch centered on grain centroid
- Patch size: **96 × 96 pixels**
- Input channels:
  - **RGB (3 channels)** - use as-is from microscopy images

---

### 4.3 Grain Masking (Critical)

Each grain has a binary mask:

- 1 = target grain
- 0 = everything else

**Default masking strategy**:
- Multiply image by mask
- Outside-grain pixels set to zero

This ensures:
- No interference from neighboring grains
- Full internal morphology preserved
- Broken edges remain visible

---

## 4.4 Data Augmentation

**Training only** (no augmentation for validation/test):

- Random rotation (360°) - grains have no preferred orientation
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random brightness/contrast adjustment (p=0.3)
- Gaussian blur (slight, p=0.1)
- Color jitter (saturation/hue, p=0.2)

Purpose: Increase effective dataset size and improve robustness

---

## 5. Dataset Splitting

### Rule (Non-negotiable)
**Split by image, not by grain**

### Strategy: 5-Fold Cross-Validation
- 5-fold stratified CV (not simple random split)
- Each fold: ~14 train / ~4 validation images
- **Stratification by image tier**:
  - Tier 1: Images with broken ooids (critical - must be in all folds)
  - Tier 2: Images with intraclasts
  - Tier 3: Basic peloid + ooid images

Goal:
- Prevent fabric / illumination leakage
- Ensure broken ooids (rarest class) present in all validation folds
- More robust evaluation for thesis/paper
- Better use of small dataset (18 images)

---

## 6. Model Architecture

### 6.1 Shared Backbone

Use **one** of the following:

- ResNet-18 (recommended default)
- EfficientNet-B0

Reasons:
- Sufficient capacity
- Stable on small datasets
- Well-studied behavior

Output:
- Feature embedding (e.g., 512-D vector)

---

### 6.2 Hierarchical Binary Heads

#### Head 1: Peloid vs Non-peloid
- Binary sigmoid output
- Most important classifier
- Trained first

#### Head 2: Ooid-like vs Intraclast
- Applied only if Head 1 = Non-peloid
- Ooid-like = ooid + broken ooid

#### Head 3: Whole vs Broken Ooid
- Applied only if Head 2 = Ooid-like
- Expected to be uncertain and noisy

Each head:
- Small MLP (e.g., 512 → 128 → 1)
- Independent loss
- Independent decision threshold

---

## 7. Sampling Strategy (Essential)

Binary classification **does not remove imbalance**.

### Stage-wise Sampling

#### Stage 1 (Peloid vs Non-peloid)
- Downsample peloids
- Oversample non-peloids
- Target batch ratio ≈ 50 / 50

#### Stage 2 (Ooid-like vs Intraclast)
- Balanced sampling

#### Stage 3 (Whole vs Broken Ooid)
- Heavy oversampling of broken ooids
- Expect noisy gradients

> Sampling affects **feature learning**, not just class balance.

---

## 8. Loss Functions

### Default Choice
**Focal Loss** for all heads.

Recommended parameters:
- Gamma (γ) = 2.0
- Alpha (α) tuned per stage

Why:
- Suppresses easy majority examples
- Forces learning from rare grains
- More robust than weighted BCE alone

---

## 9. Training Schedule

### Phase 1: Feature Learning
- Train backbone + Head 1
- Until validation performance stabilizes

### Phase 2: Intermediate Discrimination
- Freeze early backbone layers
- Train Head 2

### Phase 3: Rare-Class Fine-Tuning
- Light backbone fine-tuning
- Train Head 3 last

Purpose:
- Prevent rare broken ooids from corrupting early features

---

## 10. Inference Logic

For each grain:

```text
if P(peloid) > T1:
    label = Peloid
else:
    if P(ooid_like) > T2:
        if P(broken) > T3:
            label = Broken Ooid
        else:
            label = Whole Ooid
    else:
        label = Intraclast