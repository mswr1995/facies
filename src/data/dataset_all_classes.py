"""
PyTorch dataset for seven-class grain/component classification.

This is separate from dataset_new.py, which encodes the four-class
hierarchical labels used by the manuscript model.
"""
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


ALL_CLASS_NAMES = [
    'Peloid',
    'Ooid',
    'Broken ooid',
    'Intraclast',
    'Ostracod',
    'Bivalve',
    'Quartz grain',
]
ALL_LABEL_MAP = {name: i for i, name in enumerate(ALL_CLASS_NAMES)}


class AllClassGrainDataset(Dataset):
    """Dataset for flat seven-class classification from masked grain patches."""

    def __init__(
        self,
        split: str = 'train',
        processed_dir: str = 'data/processed_all_classes',
        transform: Optional[A.Compose] = None,
        use_default_transforms: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.patches_dir = self.processed_dir / 'patches'
        self.split = split

        split_path = self.processed_dir / f'{split}_split.json'
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        self.samples = split_data['grains']

        if transform is not None:
            self.transform = transform
        elif use_default_transforms:
            self.transform = self._get_default_transform()
        else:
            self.transform = self._get_minimal_transform()

        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()

    def _get_default_transform(self) -> A.Compose:
        if self.split == 'train':
            return A.Compose([
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=15,
                        val_shift_limit=15,
                        p=1.0,
                    ),
                ], p=0.4),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])

        return self._get_minimal_transform()

    def _get_minimal_transform(self) -> A.Compose:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _print_class_distribution(self):
        counts = {name: 0 for name in ALL_CLASS_NAMES}
        for sample in self.samples:
            counts[sample['label']] += 1

        print("  Class distribution:")
        for label in ALL_CLASS_NAMES:
            count = counts[label]
            pct = 100.0 * count / len(self.samples) if self.samples else 0.0
            print(f"    {label:15s}: {count:4d} ({pct:5.2f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        image_path = self.patches_dir / sample['patch_filename']

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Patch not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        target = torch.tensor(ALL_LABEL_MAP[sample['label']], dtype=torch.long)
        metadata = {
            'grain_id': sample['grain_id'],
            'image_name': sample['image_name'],
            'label': sample['label'],
        }
        return image, target, metadata
