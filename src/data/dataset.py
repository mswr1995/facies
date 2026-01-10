"""
PyTorch Dataset for grain patch classification.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GrainDataset(Dataset):
    """
    Dataset for loading grain patches with augmentation.
    
    Supports hierarchical labels:
    - Stage 1: Peloid (1) vs Non-peloid (0)
    - Stage 2: Ooid-like (1) vs Intraclast (0) [for non-peloids]
    - Stage 3: Whole Ooid (1) vs Broken Ooid (0) [for ooid-likes]
    """
    
    # Class definitions
    VALID_CLASSES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    def __init__(
        self,
        metadata_path: str,
        patches_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        use_default_transforms: bool = True
    ):
        """
        Args:
            metadata_path: Path to fold_*_metadata.json file
            patches_dir: Directory containing patch images
            split: 'train' or 'val'
            transform: Custom albumentations transform (overrides default)
            use_default_transforms: Use default transforms if transform=None
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get appropriate split
        if split == 'train':
            self.samples = metadata['train_grains']
        elif split == 'val':
            self.samples = metadata['val_grains']
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        elif use_default_transforms:
            self.transform = self._get_default_transform()
        else:
            self.transform = self._get_minimal_transform()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation pipeline."""
        if self.split == 'train':
            return A.Compose([
                A.RandomRotate90(p=1.0),  # Random 0/90/180/270 rotation
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=15,
                        val_shift_limit=15,
                        p=1.0
                    ),
                ], p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:  # validation
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _get_minimal_transform(self) -> A.Compose:
        """Minimal transform (normalize + tensor conversion only)."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _print_class_distribution(self):
        """Print class distribution for current split."""
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"  Class distribution:")
        for label in self.VALID_CLASSES:
            count = label_counts.get(label, 0)
            pct = 100.0 * count / len(self.samples) if len(self.samples) > 0 else 0
            print(f"    {label:15s}: {count:4d} ({pct:5.2f}%)")
    
    def get_hierarchical_labels(self, label: str) -> Dict[str, int]:
        """
        Convert class label to hierarchical binary labels.
        
        Returns:
            Dictionary with keys:
            - 'stage1': Peloid (1) vs Non-peloid (0)
            - 'stage2': Ooid-like (1) vs Intraclast (0) [for non-peloids only]
            - 'stage3': Whole Ooid (1) vs Broken Ooid (0) [for ooid-likes only]
        """
        labels = {}
        
        # Stage 1: Peloid vs Non-peloid
        if label == 'Peloid':
            labels['stage1'] = 1
            labels['stage2'] = -1  # Not applicable
            labels['stage3'] = -1  # Not applicable
        else:
            labels['stage1'] = 0
            
            # Stage 2: Ooid-like vs Intraclast
            if label == 'Intraclast':
                labels['stage2'] = 0
                labels['stage3'] = -1  # Not applicable
            else:  # Ooid or Broken ooid
                labels['stage2'] = 1
                
                # Stage 3: Whole Ooid vs Broken Ooid
                if label == 'Ooid':
                    labels['stage3'] = 1
                else:  # Broken ooid
                    labels['stage3'] = 0
        
        return labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, any]]:
        """
        Get a sample.
        
        Returns:
            image: Tensor of shape (3, H, W)
            labels: Dictionary with hierarchical labels
            metadata: Dictionary with grain metadata
        """
        sample = self.samples[idx]
        
        # Load image
        patch_path = self.patches_dir / sample['patch_filename']
        image = cv2.imread(str(patch_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get hierarchical labels
        labels = self.get_hierarchical_labels(sample['label'])
        
        # Prepare metadata
        metadata = {
            'grain_id': sample['grain_id'],
            'image_name': sample['image_name'],
            'label': sample['label'],
            'patch_filename': sample['patch_filename']
        }
        
        return image, labels, metadata


def create_dataloaders(
    fold: int,
    data_dir: str = 'data/processed',
    batch_size: int = 32,
    num_workers: int = 4,
    use_default_transforms: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for a specific fold.
    
    Args:
        fold: Fold number (0-4)
        data_dir: Directory containing processed data
        batch_size: Batch size
        num_workers: Number of worker processes
        use_default_transforms: Use default augmentation pipeline
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / f'fold_{fold}_metadata.json'
    patches_dir = data_dir / 'patches'
    
    # Create datasets
    train_dataset = GrainDataset(
        metadata_path=str(metadata_path),
        patches_dir=str(patches_dir),
        split='train',
        use_default_transforms=use_default_transforms
    )
    
    val_dataset = GrainDataset(
        metadata_path=str(metadata_path),
        patches_dir=str(patches_dir),
        split='val',
        use_default_transforms=use_default_transforms
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    """Test the dataset."""
    print("=" * 60)
    print("Testing GrainDataset")
    print("=" * 60)
    
    # Test loading fold 0
    train_loader, val_loader = create_dataloaders(
        fold=0,
        batch_size=8,
        num_workers=0  # For testing
    )
    
    print("\n" + "=" * 60)
    print("Testing train loader")
    print("=" * 60)
    
    # Get one batch
    images, labels, metadata = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels keys: {labels.keys()}")
    print(f"  Stage 1 labels: {labels['stage1'].tolist()}")
    print(f"  Stage 2 labels: {labels['stage2'].tolist()}")
    print(f"  Stage 3 labels: {labels['stage3'].tolist()}")
    
    print(f"\nMetadata keys: {metadata.keys()}")
    print(f"  Sample labels: {metadata['label'][:3]}")
    print(f"  Sample patch files: {metadata['patch_filename'][:3]}")
    
    print("\n" + "=" * 60)
    print("Testing validation loader")
    print("=" * 60)
    
    images_val, labels_val, metadata_val = next(iter(val_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images_val.shape}")
    print(f"  Labels keys: {labels_val.keys()}")
    
    # Test class distribution
    print("\n" + "=" * 60)
    print("Hierarchical label statistics (validation)")
    print("=" * 60)
    
    stage1_pos = (labels_val['stage1'] == 1).sum().item()
    stage1_neg = (labels_val['stage1'] == 0).sum().item()
    print(f"\nStage 1 (Peloid vs Non-peloid):")
    print(f"  Peloid: {stage1_pos}")
    print(f"  Non-peloid: {stage1_neg}")
    
    # Count stage 2 (only for non-peloids)
    stage2_mask = labels_val['stage2'] != -1
    if stage2_mask.sum() > 0:
        stage2_labels = labels_val['stage2'][stage2_mask]
        stage2_ooid = (stage2_labels == 1).sum().item()
        stage2_intra = (stage2_labels == 0).sum().item()
        print(f"\nStage 2 (Ooid-like vs Intraclast) [Non-peloids only]:")
        print(f"  Ooid-like: {stage2_ooid}")
        print(f"  Intraclast: {stage2_intra}")
        
        # Count stage 3 (only for ooid-likes)
        stage3_mask = labels_val['stage3'] != -1
        if stage3_mask.sum() > 0:
            stage3_labels = labels_val['stage3'][stage3_mask]
            stage3_whole = (stage3_labels == 1).sum().item()
            stage3_broken = (stage3_labels == 0).sum().item()
            print(f"\nStage 3 (Whole vs Broken Ooid) [Ooid-likes only]:")
            print(f"  Whole Ooid: {stage3_whole}")
            print(f"  Broken Ooid: {stage3_broken}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset test completed successfully!")
    print("=" * 60)
