# Experiment Status

## Current Progress

### ✅ Completed
- **Baseline**: ResNet-18 + 3 heads, Test Acc 94.90%, Intraclast Recall 57.14%
- **Exp 1 - Threshold Tuning**: Grid search found T2=0.60, T3=0.30, no improvement in test performance
  - Conclusion: Issue is feature representation, not decision thresholds

### ⏳ In Progress (Background Training)
- **Exp 2 - Data Augmentation**: Strong augmentation on CPU
  - Status: Epoch ~2-5/100 (started recently)
  - Transforms: Rotate, Flip, Elastic, Perspective, Color, Noise, Blur
  - Expected: ~50-70 hours total training time
  - Checkpoint: `checkpoints/exp2_augmented/`
  - Log: `training_exp2_augmented.log`

### 📋 Ready to Run (Next)
- **Exp 3 - Class Weighting**: 
  - Custom weights: Peloid=1.0, Ooid=2.0, Broken=5.0, Intraclast=3.0
  - Script: `scripts/exp3_train_weighted.py`
  - Can start after Exp 2 completes OR in parallel
  
### 🔮 Planned
- **Exp 4 - Alternative Architecture**: EfficientNet + attention
- **Exp 5 - Post-Processing**: Confidence + XGBoost ensemble

## Key Insights So Far

1. **Exp 1 Result**: Threshold tuning doesn't help because model learned decision boundaries are already optimal. The bottleneck is **feature quality**, not decision rules.

2. **Bottleneck Analysis**:
   - Peloid: 98.26% recall ✅ (working well)
   - Ooid: 78.57% recall ✅ (acceptable)
   - Broken Ooid: 83.33% recall ✅ (surprisingly good)
   - **Intraclast: 57.14% recall ❌** (main bottleneck)
   - PR-AUC for intraclast only 0.376 (vs 0.994 for peloid)

3. **Next Steps**:
   - Exp 2: Will test if diverse augmentations help model learn better intraclast features
   - Exp 3: Will test if class weighting during training improves minority class learning
   - Exp 4: Will test if more powerful architecture (EfficientNet) + attention helps

## Monitoring

Check training progress:
```bash
tail -f training_exp2_augmented.log
```

Check checkpoints:
```bash
ls -lh checkpoints/exp2_augmented/
```

Check results when ready:
```bash
python scripts/exp2_evaluate_augmented.py
python scripts/exp3_evaluate_weighted.py
```
