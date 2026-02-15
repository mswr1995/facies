# Test Set Results

## Per-Class Accuracy (Recall)

How many of each class were correctly identified:

| Model | Peloid | Ooid | Broken | Intraclast | Overall |
|-------|--------|------|--------|------------|---------|
| ResNet18-Hierarchical (default) | 452/460 (98%) | 33/42 (79%) | 5/6 (83%) | 12/21 (57%) | 94.9% |
| + Tuned Thresholds | 452/460 (98%) | 33/42 (79%) | 5/6 (83%) | 12/21 (57%) | 94.9% |
| + Strong Augmentation | 445/460 (97%) | 37/42 (88%) | 1/6 (17%) | 9/21 (43%) | 92.3% |
| + Class Weights | 443/460 (96%) | 35/42 (83%) | 5/6 (83%) | 13/21 (62%) | 92.6% |
| EfficientNet-B0 + Attention | 439/460 (95%) | 35/42 (83%) | 2/6 (33%) | 5/21 (24%) | 89.4% |
| XGBoost Ensemble (4 models) | 444/460 (97%) | 33/42 (79%) | 4/6 (67%) | 11/21 (52%) | 93.0% |
| + Staged Training (4-phase) | 447/460 (97%) | 35/42 (83%) | 5/6 (83%) | 10/21 (48%) | 94.0% |
| + Heavy Oversample (15x) | 450/460 (98%) | 37/42 (88%) | 5/6 (83%) | 16/21 (76%) | 96.0% |
| + Heavy Oversample (15x) + TTA | 451/460 (98%) | 39/42 (93%) | 5/6 (83%) | 14/21 (67%) | 96.2% |
| + Test-Time Augmentation (TTA) | 454/460 (99%) | 34/42 (81%) | 5/6 (83%) | 13/21 (62%) | 95.7% |
| SupCon Pre-train + Fine-tune | 436/460 (95%) | 38/42 (90%) | 4/6 (67%) | 16/21 (76%) | 93.4% |

## Confusion Matrices

### ResNet18-Hierarchical (default)
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      452     3     0      5
Ooid          8    33     1      0
Broken        0     0     5      1
Intraclast    7     2     0     12
```

### + Strong Augmentation
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      445     7     2      6
Ooid          5    37     0      0
Broken        1     0     1      4
Intraclast    9     9     0      3
```

### + Class Weights
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      443     5     1     11
Ooid          7    35     0      0
Broken        1     0     5      0
Intraclast    7     1     0     13
```

### EfficientNet-B0 + Attention
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      439    13     0      8
Ooid          7    35     0      0
Broken        4     0     2      0
Intraclast   16     4     0      1
```

### XGBoost Ensemble
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      444     3     0     13
Ooid          8    33     1      0
Broken        1     1     4      0
Intraclast    7     3     0     11
```

### + Staged Training (4-phase)
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      447     6     1      6
Ooid          3    35     4      0
Broken        0     0     5      1
Intraclast    9     2     0     10
```

### + Heavy Oversample (15x)
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      450     4     0      6
Ooid          3    37     2      0
Broken        0     0     5      1
Intraclast    4     1     0     16
```

### + Heavy Oversample (15x) + TTA
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      451     3     0      6
Ooid          2    39     1      0
Broken        0     0     5      1
Intraclast    5     2     0     14
```

### + Test-Time Augmentation (TTA)
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      454     2     0      4
Ooid          7    34     1      0
Broken        0     0     5      1
Intraclast    5     3     0     13
```

### SupCon Pre-train + Fine-tune
```
              Predicted
Actual      Pel  Ooid  Brok  Intra
Peloid      436     7     1     16
Ooid          3    38     1      0
Broken        0     1     4      1
Intraclast    4     1     0     16
```

## Test Set Class Distribution

| Class | Count | % |
|-------|-------|---|
| Peloid | 460 | 87% |
| Ooid | 42 | 8% |
| Intraclast | 21 | 4% |
| Broken | 6 | 1% |

Total: 529 samples
