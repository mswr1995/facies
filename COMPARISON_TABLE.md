# Test Set Performance Comparison

## Overall Accuracy

| Model | Test Accuracy |
|-------|--------------|
| Baseline | 94.90% |
| Exp 1 (Threshold) | 94.90% |
| Exp 2 (Augmented) | 92.25% |
| Exp 3 (Weighted) | 92.64% |
| Exp 4 (EfficientNet) | 89.42% |
| Exp 5 (Ensemble) | 93.01% |

---

## Precision (Test Set)

| Model | Peloid | Ooid | Broken | Intraclast |
|-------|--------|------|--------|------------|
| Baseline | 96.79% | 86.84% | 83.33% | 66.67% |
| Exp 1 | 96.79% | 86.84% | 83.33% | 66.67% |
| Exp 2 | 96.53% | 71.15% | 33.33% | 69.23% |
| Exp 3 | 97.79% | 77.78% | 83.33% | 52.00% |
| Exp 4 | 95.64% | 68.63% | 66.67% | 31.25% |
| Exp 5 | 96.52% | 82.50% | 80.00% | 45.83% |

---

## Recall (Test Set)

| Model | Peloid | Ooid | Broken | Intraclast |
|-------|--------|------|--------|------------|
| Baseline | 98.26% | 78.57% | 83.33% | 57.14% |
| Exp 1 | 98.26% | 78.57% | 83.33% | 57.14% |
| Exp 2 | 96.74% | 88.10% | 16.67% | 42.86% |
| Exp 3 | 96.30% | 83.33% | 83.33% | 61.90% |
| Exp 4 | 95.43% | 83.33% | 33.33% | 23.81% |
| Exp 5 | 96.52% | 78.57% | 66.67% | 52.38% |

---

## PR-AUC (Test Set)

| Model | Peloid | Ooid | Broken | Intraclast |
|-------|--------|------|--------|------------|
| Baseline | 0.9940 | 0.7430 | 0.7324 | 0.3758 |
| Exp 1 | 0.9940 | 0.7430 | 0.7324 | 0.3758 |
| Exp 2 | 0.9622 | 0.6363 | 0.0650 | 0.3194 |
| Exp 3 | 0.9739 | 0.6614 | 0.6963 | 0.3370 |
| Exp 4 | 0.9913 | 0.8227 | 0.5937 | 0.2469 |
| Exp 5 | 0.9933 | 0.9080 | 0.7871 | 0.4530 |

---

## Confusion Matrices (Test Set)

### Baseline
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            452     3       0           5
Ooid                8    33       1           0
Broken              0     0       5           1
Intraclast          7     2       0          12
```

### Exp 1 (Threshold Tuning)
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            452     3       0           5
Ooid                8    33       1           0
Broken              0     0       5           1
Intraclast          7     2       0          12
```

### Exp 2 (Data Augmentation)
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            445     7       2           6
Ooid                5    37       0           0
Broken              1     0       1           4
Intraclast          9     9       0           3
```

### Exp 3 (Class Weighting)
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            443     5       1          11
Ooid                7    35       0           0
Broken              1     0       5           0
Intraclast          7     1       0          13
```

### Exp 4 (EfficientNet)
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            439    13       0           8
Ooid                7    35       0           0
Broken              4     0       2           0
Intraclast         16     4       0           1
```

### Exp 5 (XGBoost Ensemble)
```
True\Pred      Peloid  Ooid  Broken  Intraclast
Peloid            444     3       0          13
Ooid                8    33       1           0
Broken              1     1       4           0
Intraclast          7     3       0          11
```

---

## Class Distribution (Test Set)

| Class | Count | Percentage |
|-------|-------|------------|
| Peloid | 460 | 86.96% |
| Ooid | 42 | 7.94% |
| Broken | 6 | 1.13% |
| Intraclast | 21 | 3.97% |

**Total Test Samples:** 529
