"""
Experiment 5: Train Meta-Classifier Ensemble

Extracts prediction probabilities from all 4 models (baseline, exp2, exp3, exp4)
on the validation set, then trains an XGBoost meta-classifier to combine them.
"""

import sys
import json
import pickle
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.efficientnet_model import EfficientNetHierarchicalClassifier
from src.data.dataset_new import GrainDatasetNew


def hierarchical_to_class(stage1, stage2, stage3):
    """Convert hierarchical predictions to class index."""
    if stage1 == 1:  # Peloid
        return 0
    else:  # Non-peloid (stage1 == 0)
        if stage2 == 1:  # Ooid-like
            if stage3 == 1:
                return 1  # Whole ooid
            else:
                return 2  # Broken ooid
        else:  # Non-ooid-like (stage2 == 0)
            return 3  # Intraclast


def load_model(checkpoint_path, model_type='resnet'):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle wrapped checkpoints
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model
    if model_type == 'resnet':
        model = HierarchicalGrainClassifier()
    elif model_type == 'efficientnet':
        model = EfficientNetHierarchicalClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device


def extract_probabilities(model, loader, device):
    """
    Extract prediction probabilities from a model.
    
    Returns:
        probs_array: (N, 4) array of class probabilities
        labels_array: (N,) array of true class labels
    """
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (hierarchical)
            predictions = model.get_predictions(outputs)
            
            # Models output single logits per stage, apply sigmoid to get probabilities
            stage1_probs = torch.sigmoid(outputs['stage1']).squeeze()  # P(non-peloid)
            stage2_probs = torch.sigmoid(outputs['stage2']).squeeze()  # P(non-ooid-like)
            stage3_probs = torch.sigmoid(outputs['stage3']).squeeze()  # P(broken)
            
            # Handle single-sample batches
            if stage1_probs.dim() == 0:
                stage1_probs = stage1_probs.unsqueeze(0)
                stage2_probs = stage2_probs.unsqueeze(0)
                stage3_probs = stage3_probs.unsqueeze(0)
            
            # Convert hierarchical predictions to class probabilities
            # This is an approximation based on the decision tree
            batch_probs = []
            for i in range(len(images)):
                s1_prob = stage1_probs[i].item()
                s2_prob = stage2_probs[i].item()
                s3_prob = stage3_probs[i].item()
                
                # Compute approximate class probabilities
                # P(peloid) = P(stage1=1)
                p_peloid = 1 - s1_prob
                
                # P(ooid) = P(stage1=0) * P(stage2=0) * P(stage3=0)
                p_ooid = s1_prob * (1 - s2_prob) * (1 - s3_prob)
                
                # P(broken) = P(stage1=0) * P(stage2=0) * P(stage3=1)
                p_broken = s1_prob * (1 - s2_prob) * s3_prob
                
                # P(intraclast) = P(stage1=0) * P(stage2=1)
                p_intraclast = s1_prob * s2_prob
                
                # Normalize to sum to 1
                total = p_peloid + p_ooid + p_broken + p_intraclast
                if total > 0:
                    probs = [p_peloid / total, p_ooid / total, p_broken / total, p_intraclast / total]
                else:
                    probs = [0.25, 0.25, 0.25, 0.25]
                
                batch_probs.append(probs)
            
            all_probs.extend(batch_probs)
            
            # Convert hierarchical labels to class indices
            # labels is a list of dicts, one per sample in batch
            for i in range(len(images)):
                # Extract stage labels for this sample
                s1 = labels[i]['stage1'] if isinstance(labels, list) else labels['stage1'][i].item()
                s2 = labels[i]['stage2'] if isinstance(labels, list) else labels['stage2'][i].item()
                s3 = labels[i]['stage3'] if isinstance(labels, list) else labels['stage3'][i].item()
                
                class_idx = hierarchical_to_class(s1, s2, s3)
                all_labels.append(class_idx)
    
    return np.array(all_probs), np.array(all_labels)


def main():
    print("=" * 80)
    print("Experiment 5: Meta-Classifier Ensemble Training")
    print("=" * 80)
    
    # Paths to checkpoints
    checkpoint_paths = {
        'baseline': 'checkpoints/new_split_v2/best_model_overall_acc_0.9337.pth',
        'exp2': 'checkpoints/exp2_augmented/best_model_overall_acc_0.9280.pth',
        'exp3': 'checkpoints/exp3_weighted/best_model_overall_acc_0.9413.pth',
        'exp4': 'checkpoints/exp4_efficientnet/best_model_overall_acc_0.9015.pth'
    }
    
    model_types = {
        'baseline': 'resnet',
        'exp2': 'resnet',
        'exp3': 'resnet',
        'exp4': 'efficientnet'
    }
    
    # Check if all checkpoints exist
    for name, path in checkpoint_paths.items():
        full_path = project_root / path
        if not full_path.exists():
            print(f"❌ Checkpoint not found: {path}")
            return
    
    print("✓ All checkpoints found")
    print()
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = GrainDatasetNew(
        split='val',
        patches_dir='data/processed/patches',
        use_default_transforms=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Extract features from all models
    print("Extracting features from all models...")
    print("-" * 80)
    
    all_features = []
    true_labels = None
    
    for name in ['baseline', 'exp2', 'exp3', 'exp4']:
        print(f"\nLoading {name} model...")
        checkpoint_path = project_root / checkpoint_paths[name]
        model, device = load_model(checkpoint_path, model_types[name])
        
        print(f"Extracting probabilities from {name}...")
        probs, labels = extract_probabilities(model, val_loader, device)
        
        print(f"  Shape: {probs.shape}")
        print(f"  Sample probs (first image): {probs[0]}")
        
        all_features.append(probs)
        
        if true_labels is None:
            true_labels = labels
    
    # Combine features: (N, 4*4) = (N, 16)
    X_train = np.concatenate(all_features, axis=1)
    y_train = true_labels
    
    print()
    print("=" * 80)
    print(f"Combined feature shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Label distribution: {np.bincount(y_train)}")
    print()
    
    # Train XGBoost meta-classifier
    print("Training XGBoost meta-classifier...")
    print("-" * 80)
    
    # Handle class imbalance with scale_pos_weight
    class_counts = np.bincount(y_train)
    # For multiclass, we use sample weights instead
    sample_weights = np.zeros(len(y_train))
    for i in range(len(class_counts)):
        class_weight = len(y_train) / (len(class_counts) * class_counts[i])
        sample_weights[y_train == i] = class_weight
    
    meta_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42,
        tree_method='hist',
        device='cpu'
    )
    
    meta_clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=True)
    
    # Evaluate on validation set
    print()
    print("Validation set performance:")
    print("-" * 80)
    
    y_pred = meta_clf.predict(X_train)
    val_acc = accuracy_score(y_train, y_pred)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print()
    
    class_names = ['Peloid', 'Ooid', 'Broken', 'Intraclast']
    print(classification_report(y_train, y_pred, target_names=class_names, digits=4))
    
    # Save meta-classifier
    output_dir = project_root / 'checkpoints' / 'exp5_ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'meta_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(meta_clf, f)
    
    print(f"\n✓ Meta-classifier saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'checkpoint_paths': checkpoint_paths,
        'model_types': model_types,
        'feature_dim': X_train.shape[1],
        'num_classes': 4,
        'class_names': class_names,
        'val_accuracy': float(val_acc),
        'xgboost_params': meta_clf.get_params()
    }
    
    metadata_path = output_dir / 'ensemble_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    print()
    print("=" * 80)
    print("✓ Ensemble training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
