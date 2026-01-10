"""
Evaluation metrics for hierarchical classification.
"""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class HierarchicalMetrics:
    """
    Compute metrics for hierarchical classification.
    
    Tracks metrics at each stage and overall classification accuracy.
    """
    
    def __init__(self, class_names=None):
        """
        Args:
            class_names: List of class names for final predictions
        """
        if class_names is None:
            self.class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
        else:
            self.class_names = class_names
    
    def compute_stage_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a single stage.
        
        Args:
            predictions: Binary predictions (0 or 1)
            targets: Ground truth labels (0 or 1)
            mask: Optional mask for valid samples (targets != -1)
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        # Apply mask if provided
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Filter out invalid targets (-1)
        valid_mask = targets != -1
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(targets) == 0:
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'count': 0
            }
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(target_np, pred_np)
        f1 = f1_score(target_np, pred_np, average='binary', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'count': len(targets)
        }
    
    def compute_all_metrics(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        model
    ) -> Dict[str, float]:
        """
        Compute metrics for all stages and overall accuracy.
        
        Args:
            logits: Dictionary with stage1, stage2, stage3 logits
            labels: Dictionary with stage1, stage2, stage3 ground truth
            model: Model instance (for get_predictions method)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Stage 1: Peloid vs Non-peloid
        stage1_pred = (torch.sigmoid(logits['stage1']).squeeze() > 0.5).long()
        stage1_metrics = self.compute_stage_metrics(stage1_pred, labels['stage1'])
        metrics['stage1_acc'] = stage1_metrics['accuracy']
        metrics['stage1_f1'] = stage1_metrics['f1']
        metrics['stage1_count'] = stage1_metrics['count']
        
        # Stage 2: Ooid-like vs Intraclast (only non-peloids)
        stage2_mask = labels['stage2'] != -1
        if stage2_mask.sum() > 0:
            stage2_pred = (torch.sigmoid(logits['stage2']).squeeze() > 0.5).long()
            stage2_metrics = self.compute_stage_metrics(
                stage2_pred, labels['stage2'], mask=stage2_mask
            )
            metrics['stage2_acc'] = stage2_metrics['accuracy']
            metrics['stage2_f1'] = stage2_metrics['f1']
            metrics['stage2_count'] = stage2_metrics['count']
        else:
            metrics['stage2_acc'] = 0.0
            metrics['stage2_f1'] = 0.0
            metrics['stage2_count'] = 0
        
        # Stage 3: Whole vs Broken Ooid (only ooid-likes)
        stage3_mask = labels['stage3'] != -1
        if stage3_mask.sum() > 0:
            stage3_pred = (torch.sigmoid(logits['stage3']).squeeze() > 0.5).long()
            stage3_metrics = self.compute_stage_metrics(
                stage3_pred, labels['stage3'], mask=stage3_mask
            )
            metrics['stage3_acc'] = stage3_metrics['accuracy']
            metrics['stage3_f1'] = stage3_metrics['f1']
            metrics['stage3_count'] = stage3_metrics['count']
        else:
            metrics['stage3_acc'] = 0.0
            metrics['stage3_f1'] = 0.0
            metrics['stage3_count'] = 0
        
        # Overall accuracy: hierarchical prediction vs ground truth
        final_pred = model.get_predictions(logits)
        ground_truth = self._labels_to_classes(labels)
        
        overall_acc = accuracy_score(ground_truth.cpu().numpy(), final_pred.cpu().numpy())
        metrics['overall_acc'] = float(overall_acc)
        
        return metrics
    
    def _labels_to_classes(self, labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert hierarchical labels to final class indices.
        
        Args:
            labels: Dictionary with stage1, stage2, stage3 labels
            
        Returns:
            Class indices (0=Peloid, 1=Ooid, 2=Broken ooid, 3=Intraclast)
        """
        batch_size = labels['stage1'].size(0)
        classes = torch.zeros(batch_size, dtype=torch.long, device=labels['stage1'].device)
        
        # Stage 1: Peloid vs Non-peloid
        stage1 = labels['stage1']
        peloid_mask = stage1 == 1
        classes[peloid_mask] = 0  # Peloid
        
        # Stage 2: Ooid-like vs Intraclast (for non-peloids)
        stage2 = labels['stage2']
        intraclast_mask = (stage1 == 0) & (stage2 == 0)
        classes[intraclast_mask] = 3  # Intraclast
        
        # Stage 3: Whole vs Broken (for ooid-likes)
        stage3 = labels['stage3']
        ooid_mask = (stage1 == 0) & (stage2 == 1) & (stage3 == 1)
        broken_mask = (stage1 == 0) & (stage2 == 1) & (stage3 == 0)
        
        classes[ooid_mask] = 1  # Ooid
        classes[broken_mask] = 2  # Broken ooid
        
        return classes
    
    def compute_confusion_matrix(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        model
    ) -> np.ndarray:
        """
        Compute confusion matrix for final predictions.
        
        Args:
            logits: Dictionary with stage logits
            labels: Dictionary with stage labels
            model: Model instance
            
        Returns:
            Confusion matrix (4x4)
        """
        final_pred = model.get_predictions(logits)
        ground_truth = self._labels_to_classes(labels)
        
        cm = confusion_matrix(
            ground_truth.cpu().numpy(),
            final_pred.cpu().numpy(),
            labels=[0, 1, 2, 3]
        )
        
        return cm
    
    def print_classification_report(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        model
    ):
        """
        Print detailed classification report.
        
        Args:
            logits: Dictionary with stage logits
            labels: Dictionary with stage labels
            model: Model instance
        """
        final_pred = model.get_predictions(logits)
        ground_truth = self._labels_to_classes(labels)
        
        report = classification_report(
            ground_truth.cpu().numpy(),
            final_pred.cpu().numpy(),
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        
        print("\nClassification Report:")
        print("=" * 60)
        print(report)
    
    def print_confusion_matrix(self, cm: np.ndarray):
        """
        Print formatted confusion matrix.
        
        Args:
            cm: Confusion matrix (4x4)
        """
        print("\nConfusion Matrix:")
        print("=" * 60)
        
        # Print header
        header = "        " + "  ".join([f"{name[:8]:>8s}" for name in self.class_names])
        print(header)
        print("-" * 60)
        
        # Print rows
        for i, name in enumerate(self.class_names):
            row = f"{name[:8]:8s}"
            for j in range(len(self.class_names)):
                row += f"{cm[i, j]:>10d}"
            print(row)
        
        print("=" * 60)
