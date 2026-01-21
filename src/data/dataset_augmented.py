"""
Enhanced dataset loader with stronger augmentation for rare classes.
Experiment 2: Data Augmentation
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


class GrainDatasetAugmented(Dataset):
    """
    Enhanced dataset with aggressive augmentation for training on rare classes.
    
    Augmentation strategy:
    - Rotation, flips, elastic deformations
    - Color jittering (brightness, contrast, hue, saturation)
    - Random crops and resizing
    - Gaussian noise and blur
    """
    
    VALID_CLASSES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    RARE_CLASSES = {'Broken ooid', 'Intraclast'}  # Classes with < 5% support
    
    def __init__(
        self,
        split: str = 'train',
        patches_dir: str = 'data/processed/patches',
        transform: Optional[A.Compose] = None,
        use_default_transforms: bool = True,
        strong_augmentation: bool = True
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            patches_dir: Directory containing patch images
            transform: Custom albumentations transform (overrides default)
            use_default_transforms: Use default transforms if transform=None
            strong_augmentation: Use aggressive augmentation for train split
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        self.strong_augmentation = strong_augmentation
        
        # Load split data
        split_path = f'data/processed/{split}_split.json'
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        self.samples = split_data['grains']
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        elif use_default_transforms:
            self.transform = self._get_augmentation_pipeline()
        else:
            self.transform = self._get_minimal_transform()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _get_augmentation_pipeline(self) -> A.Compose:
        """Get augmentation pipeline with strong transforms for train split."""
        if self.split == 'train' and self.strong_augmentation:
            return A.Compose([
                # Geometric transformations
                A.Rotate(limit=45, p=0.8),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.3),
                
                # Elastic deformation
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=0.3
                ),
                
                # Spatial transforms
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                
                # Color/intensity transforms
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=25,
                        p=1.0
                    ),
                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=1.0
                    ),
                ], p=0.7),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Minimal augmentation for val/test
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
            image = self.transform(image=image)['image']
        
        # Create hierarchical labels
        label_name = sample['label']
        labels = self._create_hierarchical_labels(label_name)
        
        # Metadata
        metadata = {
            'label': label_name,
            'image_id': sample.get('image_id', ''),
            'grain_id': sample.get('grain_id', '')
        }
        
        return image, labels, metadata
    
    def _create_hierarchical_labels(self, label: str) -> Dict[str, int]:
        """Create hierarchical labels from grain class."""
        labels = {
            'stage1': -1,
            'stage2': -1,
            'stage3': -1
        }
        
        if label == 'Peloid':
            labels['stage1'] = 1  # Peloid
        elif label in ['Ooid', 'Broken ooid']:
            labels['stage1'] = 0  # Non-peloid
            labels['stage2'] = 1  # Ooid-like
            
            if label == 'Ooid':
                labels['stage3'] = 1  # Whole
            else:
                labels['stage3'] = 0  # Broken
        elif label == 'Intraclast':
            labels['stage1'] = 0  # Non-peloid
            labels['stage2'] = 0  # Intraclast
        
        return labels


def create_augmented_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    strong_augmentation: bool = True
):
    """Create dataloaders with augmented training data."""
    train_dataset = GrainDatasetAugmented(
        split='train',
        strong_augmentation=strong_augmentation
    )
    val_dataset = GrainDatasetAugmented(
        split='val',
        strong_augmentation=False
    )
    test_dataset = GrainDatasetAugmented(
        split='test',
        strong_augmentation=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader
