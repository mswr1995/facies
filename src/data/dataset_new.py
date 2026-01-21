"""
PyTorch Dataset for grain patch classification with new train/val/test split.
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


class GrainDatasetNew(Dataset):
    """
    Dataset for loading grain patches with new non-overlapping train/val/test split.
    
    Supports hierarchical labels:
    - Stage 1: Peloid (1) vs Non-peloid (0)
    - Stage 2: Ooid-like (1) vs Intraclast (0) [for non-peloids]
    - Stage 3: Whole Ooid (1) vs Broken Ooid (0) [for ooid-likes]
    """
    
    # Class definitions
    VALID_CLASSES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    
    def __init__(
        self,
        split: str = 'train',
        patches_dir: str = 'data/processed/patches',
        transform: Optional[A.Compose] = None,
        use_default_transforms: bool = True
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            patches_dir: Directory containing patch images
            transform: Custom albumentations transform (overrides default)
            use_default_transforms: Use default transforms if transform=None
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        
        # Load split data
        split_path = f'data/processed/{split}_split.json'
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        self.samples = split_data['grains']
        
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
                A.RandomRotate90(p=1.0),
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
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
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
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int], Dict]:
        sample = self.samples[idx]
        
        # Load image
        image_path = self.patches_dir / sample['patch_filename']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get hierarchical labels
        labels = self.get_hierarchical_labels(sample['label'])
        
        # Metadata
        metadata = {
            'grain_id': sample['grain_id'],
            'image_name': sample['image_name'],
            'label': sample['label']
        }
        
        return image, labels, metadata
    
    def get_hierarchical_labels(self, label: str) -> Dict[str, int]:
        """Convert class label to hierarchical binary labels."""
        labels = {}
        
        # Stage 1: Peloid vs Non-peloid
        if label == 'Peloid':
            labels['stage1'] = 1
            labels['stage2'] = -1
            labels['stage3'] = -1
        else:
            labels['stage1'] = 0
            
            # Stage 2: Ooid-like vs Intraclast
            if label == 'Intraclast':
                labels['stage2'] = 0
                labels['stage3'] = -1
            else:
                labels['stage2'] = 1
                
                # Stage 3: Whole Ooid vs Broken Ooid
                if label == 'Ooid':
                    labels['stage3'] = 1
                else:
                    labels['stage3'] = 0
        
        return labels


def create_new_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    use_default_transforms: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders with new split.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = GrainDatasetNew(
        split='train',
        use_default_transforms=use_default_transforms
    )
    
    val_dataset = GrainDatasetNew(
        split='val',
        use_default_transforms=use_default_transforms
    )
    
    test_dataset = GrainDatasetNew(
        split='test',
        use_default_transforms=use_default_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
