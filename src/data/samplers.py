"""
Class-aware samplers for hierarchical grain classification.

Implements stage-wise balanced sampling to handle extreme class imbalance:
- Stage 1: Balance peloid vs non-peloid
- Stage 2: Balance ooid-like vs intraclast (from non-peloids)
- Stage 3: Heavily oversample broken ooids (from ooid-likes)
"""
import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict
from collections import defaultdict


class HierarchicalBalancedSampler(Sampler):
    """
    Balanced sampler for hierarchical classification.
    
    Ensures each batch has balanced representation at each stage:
    - Stage 1: ~50/50 peloid vs non-peloid
    - Stage 2: Balanced ooid-like vs intraclast
    - Stage 3: Heavy oversampling of broken ooids
    
    Args:
        dataset: GrainDataset instance
        samples_per_epoch: Total samples per epoch (default: 2x dataset size)
        stage1_balance: Target ratio for peloid vs non-peloid (default: 0.5)
        stage3_broken_weight: Oversampling weight for broken ooids (default: 5.0)
        shuffle: Whether to shuffle indices (default: True)
    """
    
    def __init__(
        self,
        dataset,
        samples_per_epoch: int = None,
        stage1_balance: float = 0.5,
        stage3_broken_weight: float = 5.0,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.stage1_balance = stage1_balance
        self.stage3_broken_weight = stage3_broken_weight
        
        # Default to 2x dataset size for oversampling effect
        if samples_per_epoch is None:
            samples_per_epoch = len(dataset) * 2
        self.samples_per_epoch = samples_per_epoch
        
        # Organize indices by class
        self._organize_indices()
        
        # Compute sampling weights
        self._compute_weights()
        
        print(f"\nHierarchical Balanced Sampler initialized:")
        print(f"  Samples per epoch: {self.samples_per_epoch}")
        print(f"  Stage 1 balance (peloid): {self.stage1_balance:.2f}")
        print(f"  Stage 3 broken ooid weight: {self.stage3_broken_weight:.1f}x")
        print(f"  Class distribution:")
        for cls in ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']:
            count = len(self.class_indices.get(cls, []))
            print(f"    {cls:15s}: {count:4d} samples")
    
    def _organize_indices(self):
        """Organize dataset indices by class label."""
        self.class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.samples[idx]
            label = sample['label']
            self.class_indices[label].append(idx)
        
        # Convert to regular dict with lists
        self.class_indices = dict(self.class_indices)
        
        # Create hierarchical groupings
        self.peloid_indices = self.class_indices.get('Peloid', [])
        self.non_peloid_indices = []
        for cls in ['Ooid', 'Broken ooid', 'Intraclast']:
            self.non_peloid_indices.extend(self.class_indices.get(cls, []))
        
        # Stage 2 groupings (non-peloids only)
        self.intraclast_indices = self.class_indices.get('Intraclast', [])
        self.ooid_like_indices = []
        for cls in ['Ooid', 'Broken ooid']:
            self.ooid_like_indices.extend(self.class_indices.get(cls, []))
        
        # Stage 3 groupings (ooid-likes only)
        self.whole_ooid_indices = self.class_indices.get('Ooid', [])
        self.broken_ooid_indices = self.class_indices.get('Broken ooid', [])
    
    def _compute_weights(self):
        """Compute sampling weights for each sample."""
        self.weights = np.zeros(len(self.dataset))
        
        # Stage 1: Balance peloid vs non-peloid
        n_peloid = len(self.peloid_indices)
        n_non_peloid = len(self.non_peloid_indices)
        
        if n_peloid > 0:
            peloid_weight = self.stage1_balance / n_peloid
            for idx in self.peloid_indices:
                self.weights[idx] = peloid_weight
        
        if n_non_peloid > 0:
            non_peloid_weight = (1.0 - self.stage1_balance) / n_non_peloid
            for idx in self.non_peloid_indices:
                self.weights[idx] = non_peloid_weight
        
        # Stage 3: Boost broken ooids heavily
        n_broken = len(self.broken_ooid_indices)
        if n_broken > 0:
            for idx in self.broken_ooid_indices:
                self.weights[idx] *= self.stage3_broken_weight
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
    
    def __iter__(self):
        """Generate indices for one epoch."""
        # Sample with replacement using computed weights
        indices = np.random.choice(
            len(self.dataset),
            size=self.samples_per_epoch,
            replace=True,
            p=self.weights
        )
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self):
        return self.samples_per_epoch


class StageWiseBatchSampler(Sampler):
    """
    Advanced sampler that constructs batches with stage-wise balance.
    
    Each batch contains:
    - ~50% peloids, ~50% non-peloids (Stage 1 balance)
    - Among non-peloids: balanced ooid-like vs intraclast
    - Among ooid-likes: oversampled broken ooids
    
    This is more sophisticated than simple weighted sampling as it
    ensures balance within each batch, not just across the epoch.
    
    Args:
        dataset: GrainDataset instance
        batch_size: Batch size
        peloid_ratio: Ratio of peloids per batch (default: 0.5)
        broken_ooid_oversample: How many times to repeat broken ooids (default: 3)
        drop_last: Whether to drop incomplete batches (default: True)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        peloid_ratio: float = 0.5,
        broken_ooid_oversample: int = 3,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.peloid_ratio = peloid_ratio
        self.broken_ooid_oversample = broken_ooid_oversample
        self.drop_last = drop_last
        
        # Organize indices
        self._organize_indices()
        
        # Compute batch composition
        self.n_peloid_per_batch = int(batch_size * peloid_ratio)
        self.n_non_peloid_per_batch = batch_size - self.n_peloid_per_batch
        
        # Among non-peloids, aim for 50/50 split
        self.n_intraclast_per_batch = self.n_non_peloid_per_batch // 2
        self.n_ooid_like_per_batch = self.n_non_peloid_per_batch - self.n_intraclast_per_batch
        
        # Estimate number of batches (based on minority class)
        min_class_size = min(
            len(self.peloid_indices),
            len(self.non_peloid_indices)
        )
        self.num_batches = min_class_size // (batch_size // 2)
        
        print(f"\nStage-wise Batch Sampler initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {self.num_batches}")
        print(f"  Samples per epoch: {self.num_batches * batch_size}")
        print(f"  Batch composition:")
        print(f"    Peloids: {self.n_peloid_per_batch} ({peloid_ratio*100:.0f}%)")
        print(f"    Non-peloids: {self.n_non_peloid_per_batch}")
        print(f"      ├─ Intraclasts: {self.n_intraclast_per_batch}")
        print(f"      └─ Ooid-likes: {self.n_ooid_like_per_batch}")
        print(f"  Broken ooid oversample: {broken_ooid_oversample}x")
    
    def _organize_indices(self):
        """Organize dataset indices by class label."""
        self.class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.samples[idx]
            label = sample['label']
            self.class_indices[label].append(idx)
        
        # Convert to regular dict
        self.class_indices = dict(self.class_indices)
        
        # Hierarchical groupings
        self.peloid_indices = np.array(self.class_indices.get('Peloid', []))
        
        self.intraclast_indices = np.array(self.class_indices.get('Intraclast', []))
        
        # Ooid-likes with broken ooid oversampling
        whole_ooid_indices = np.array(self.class_indices.get('Ooid', []))
        broken_ooid_indices = np.array(self.class_indices.get('Broken ooid', []))
        
        # Repeat broken ooids for oversampling
        broken_ooid_repeated = np.repeat(
            broken_ooid_indices,
            self.broken_ooid_oversample
        )
        
        self.ooid_like_indices = np.concatenate([
            whole_ooid_indices,
            broken_ooid_repeated
        ])
        
        # Combined non-peloids
        self.non_peloid_indices = np.concatenate([
            self.intraclast_indices,
            self.ooid_like_indices
        ])
    
    def __iter__(self):
        """Generate batches with stage-wise balance."""
        # Shuffle indices for each class
        peloid_shuffled = np.random.permutation(self.peloid_indices)
        intraclast_shuffled = np.random.permutation(self.intraclast_indices)
        ooid_like_shuffled = np.random.permutation(self.ooid_like_indices)
        
        batches = []
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Sample peloids (with replacement if needed)
            peloid_sample = np.random.choice(
                peloid_shuffled,
                size=self.n_peloid_per_batch,
                replace=len(peloid_shuffled) < self.n_peloid_per_batch
            )
            batch.extend(peloid_sample.tolist())
            
            # Sample intraclasts
            if len(intraclast_shuffled) > 0:
                intraclast_sample = np.random.choice(
                    intraclast_shuffled,
                    size=self.n_intraclast_per_batch,
                    replace=len(intraclast_shuffled) < self.n_intraclast_per_batch
                )
                batch.extend(intraclast_sample.tolist())
            
            # Sample ooid-likes (includes oversampled broken ooids)
            if len(ooid_like_shuffled) > 0:
                ooid_like_sample = np.random.choice(
                    ooid_like_shuffled,
                    size=self.n_ooid_like_per_batch,
                    replace=len(ooid_like_shuffled) < self.n_ooid_like_per_batch
                )
                batch.extend(ooid_like_sample.tolist())
            
            # Shuffle within batch
            np.random.shuffle(batch)
            batches.append(batch)
        
        # Shuffle batch order
        np.random.shuffle(batches)
        
        # Flatten and return
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.num_batches
