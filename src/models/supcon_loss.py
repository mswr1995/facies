"""
Supervised Contrastive Loss for pre-training the backbone.

Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
https://arxiv.org/abs/2004.11362

The key idea: pull together embeddings of same-class samples, push apart
embeddings of different-class samples. This learns a feature space where
minority classes (Intraclast, Broken ooid) are well-separated from the
majority (Peloid) BEFORE any classification head is attached.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    For each anchor, all same-class samples in the batch are positives
    and all different-class samples are negatives. The loss encourages
    same-class samples to cluster tightly in embedding space.
    
    Args:
        temperature: Scaling factor for logits (default: 0.07).
                     Lower temperature = sharper distribution = harder contrastive task.
        base_temperature: Base temperature for normalization (default: 0.07).
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Normalized projected features (batch_size, proj_dim).
                      MUST be L2-normalized before passing in.
            labels: Ground truth class labels (batch_size,) with integer values
                    (0=Peloid, 1=Ooid, 2=Broken ooid, 3=Intraclast).
        
        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        labels = labels.contiguous().view(-1, 1)
        
        # Mask: 1 where labels match (same class), 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute pairwise cosine similarities (features are already L2-normalized)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability, subtract max from each row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Compute log_prob: log(exp(sim_i,j) / sum_k≠i(exp(sim_i,k)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positives
        # For each anchor, average the log_prob over its positive pairs
        num_positives = mask.sum(1)
        
        # Avoid division by zero for samples with no positives in batch
        num_positives = torch.clamp(num_positives, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / num_positives
        
        # Loss (with temperature scaling)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps backbone features (512-D) to a lower-dimensional space (128-D)
    where the contrastive loss is applied. This head is discarded after
    pre-training — only the backbone is kept.
    
    Architecture: 512 -> 256 -> 128 (with ReLU and BN)
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Project and L2-normalize features.
        
        Args:
            x: Backbone features (batch_size, 512)
            
        Returns:
            L2-normalized projections (batch_size, 128)
        """
        z = self.net(x)
        return F.normalize(z, dim=1)


if __name__ == '__main__':
    """Test SupCon loss and projection head."""
    print("Testing Supervised Contrastive Loss")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Simulate a batch with 4 classes
    batch_size = 16
    feat_dim = 512
    proj_dim = 128
    
    # Create projection head
    proj = ProjectionHead(input_dim=feat_dim, output_dim=proj_dim)
    
    # Simulate backbone features
    features = torch.randn(batch_size, feat_dim)
    
    # Simulate balanced labels (4 per class)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
    # Project features
    projections = proj(features)
    print(f"Input features shape: {features.shape}")
    print(f"Projected features shape: {projections.shape}")
    print(f"L2 norm check (should be ~1.0): {projections.norm(dim=1).mean():.4f}")
    
    # Compute loss
    criterion = SupConLoss(temperature=0.07)
    loss = criterion(projections, labels)
    print(f"\nSupCon Loss: {loss.item():.4f}")
    
    # Verify gradient flows
    loss.backward()
    print(f"Gradient flows: {proj.net[0].weight.grad is not None}")
    
    print("\n" + "=" * 60)
    print("SupCon loss test passed!")
