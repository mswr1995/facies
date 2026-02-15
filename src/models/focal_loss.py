"""
Focal Loss for handling class imbalance.

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
https://arxiv.org/abs/1708.02002
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This loss applies a modulating term (1 - p_t)^gamma to the cross entropy loss,
    which focuses training on hard examples and down-weights easy examples.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples.
               Default: 0.25 (works well for most cases)
        gamma: Focusing parameter for modulating loss. Higher gamma increases
               the focus on hard examples. Default: 2.0
        reduction: 'mean', 'sum' or 'none'. Default: 'mean'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size,) or (batch_size, 1)
            targets: Ground truth labels (batch_size,) with values 0 or 1
            
        Returns:
            Focal loss value
        """
        # Ensure inputs are 1D (but keep at least 1 dim for single samples)
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        if inputs.dim() == 0:
            inputs = inputs.unsqueeze(0)

        # Ensure targets are float and 1D
        targets = targets.float()
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self):
        return f'FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction})'


if __name__ == '__main__':
    """Test focal loss."""
    print("Testing Focal Loss")
    print("=" * 60)
    
    # Create sample data: 90% class 0, 10% class 1 (imbalanced)
    torch.manual_seed(42)
    batch_size = 100
    
    # Simulate predictions (logits)
    logits = torch.randn(batch_size)
    
    # Imbalanced targets: 90 negatives, 10 positives
    targets = torch.zeros(batch_size)
    targets[:10] = 1.0
    
    # Compare with standard BCE loss
    bce_loss_fn = nn.BCEWithLogitsLoss()
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    bce_loss = bce_loss_fn(logits, targets)
    focal_loss = focal_loss_fn(logits, targets)
    
    print(f"Standard BCE Loss: {bce_loss.item():.4f}")
    print(f"Focal Loss (gamma=2.0): {focal_loss.item():.4f}")
    
    # Test with different gamma values
    print("\nFocal Loss with different gamma values:")
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        fl = FocalLoss(alpha=0.25, gamma=gamma)
        loss = fl(logits, targets)
        print(f"  gamma={gamma}: {loss.item():.4f}")
    
    print("\nFocal Loss with different alpha values (gamma=2.0):")
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        fl = FocalLoss(alpha=alpha, gamma=2.0)
        loss = fl(logits, targets)
        print(f"  alpha={alpha}: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Focal Loss test completed successfully!")
