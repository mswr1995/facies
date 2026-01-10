# Facies Classification - Setup Guide

## Project Structure Created ✅

```
facies/
├── data/
│   ├── raw/                    # Your labelme annotations
│   └── processed/              # Will contain preprocessed patches
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   └── labelme_loader.py  # Core utilities (ready to use)
│   ├── models/                 # Model architectures (next step)
│   ├── training/               # Training loops and losses (next step)
│   └── utils/                  # Visualization and metrics (next step)
├── configs/                    # Configuration files
├── notebooks/
│   └── 01_data_exploration.ipynb  # 📊 START HERE
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```powershell
# Using uv (fast Python package installer)
# Install uv if you haven't: pip install uv

# Create virtual environment and install dependencies
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# Or in one command:
uv pip sync requirements.txt
```

### 2. Explore Data

Open and run the notebook:
```
notebooks/01_data_exploration.ipynb
```

This will:
- ✅ Load all 18 images with labelme annotations
- ✅ Show class distribution (87% peloid, 1% broken ooid)
- ✅ Visualize images with polygon overlays
- ✅ Extract and display sample grain patches (96×96)
- ✅ Analyze image tiers for 5-fold CV stratification

### 3. Next Steps (after visualization)

We'll implement:
1. **Preprocessing pipeline**: Labelme → patches with masks
2. **5-fold stratified splitting**: Ensure broken ooids in all folds
3. **PyTorch Dataset**: With augmentation for training
4. **Hierarchical model**: ResNet-18 backbone + 3 binary heads
5. **Training pipeline**: Focal loss + stage-wise sampling

## Key Decisions Made ✅

- ✅ RGB input (3 channels, use as-is)
- ✅ 96×96 patches centered on grain centroids
- ✅ Zero-masking (multiply by binary mask)
- ✅ 5-fold cross-validation (not simple train/val/test)
- ✅ Filter out: Bivalve, Ostracod, Quartz grain
- ✅ Augmentation: Rotation, flip, brightness, blur, color jitter

## Troubleshooting

If you encounter import errors in the notebook:
```python
import sys
sys.path.append('../src')  # Already included in notebook
```

If labelme package is missing:
```powershell
uv pip install labelme
```

---

**Ready to visualize!** Open `notebooks/01_data_exploration.ipynb` in Jupyter or VS Code.
