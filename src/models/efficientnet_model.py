"""
EfficientNet-B0 based hierarchical grain classifier with attention mechanisms.

Architecture improvements over baseline ResNet-18:
1. EfficientNet-B0: More efficient feature extraction with compound scaling
2. Spatial Attention: Learn which grain regions are most informative
3. Channel Attention: Emphasize important feature channels
4. Deeper feature extraction for fine-grained morphology discrimination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important grain regions.
    
    Applies both average and max pooling across channels, then learns
    spatial attention weights via a 7x7 convolution.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply conv
        pooled = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation).
    
    Learns to emphasize important feature channels via adaptive
    global pooling and two FC layers.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Both average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class AttentionBlock(nn.Module):
    """Combined channel and spatial attention."""
    def __init__(self, in_channels, reduction=16):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EfficientNetHierarchicalClassifier(nn.Module):
    """
    EfficientNet-B0 based hierarchical grain classifier.
    
    Architecture:
    - EfficientNet-B0 backbone (ImageNet pretrained)
    - Attention blocks after feature extraction
    - Three independent binary classification heads (hierarchical)
    
    Hierarchy:
        Stage 1: Peloid (1) vs Ooid-like (0)
        Stage 2: Ooid (1) vs Intraclast (0)  [if stage1=0]
        Stage 3: Whole ooid (1) vs Broken (0) [if stage1=0 & stage2=1]
    """
    
    def __init__(self, pretrained=True, dropout=0.3):
        super(EfficientNetHierarchicalClassifier, self).__init__()
        
        # EfficientNet-B0 backbone (5.3M params vs ResNet-18 11.7M)
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Extract feature extractor (everything except classifier)
        self.features = efficientnet.features
        
        # EfficientNet-B0 outputs 1280 channels
        self.feature_channels = 1280
        
        # Attention mechanism
        self.attention = AttentionBlock(self.feature_channels, reduction=16)
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Three independent binary classifiers
        # Stage 1: Peloid vs Ooid-like
        self.classifier_stage1 = nn.Sequential(
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # Stage 2: Ooid vs Intraclast (among ooid-like)
        self.classifier_stage2 = nn.Sequential(
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # Stage 3: Whole vs Broken (among ooids)
        self.classifier_stage3 = nn.Sequential(
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier layer weights."""
        for m in [self.classifier_stage1, self.classifier_stage2, self.classifier_stage3]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through hierarchical classifier.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary with stage logits:
                stage1: [B, 1] - Peloid vs Ooid-like
                stage2: [B, 1] - Ooid vs Intraclast
                stage3: [B, 1] - Whole vs Broken
        """
        # Feature extraction
        features = self.features(x)  # [B, 1280, H', W']
        
        # Apply attention
        features = self.attention(features)  # [B, 1280, H', W']
        
        # Global average pooling
        features = self.avgpool(features)  # [B, 1280, 1, 1]
        features = torch.flatten(features, 1)  # [B, 1280]
        
        # Apply dropout
        features = self.dropout(features)
        
        # Three independent predictions
        stage1_logits = self.classifier_stage1(features)  # [B, 1]
        stage2_logits = self.classifier_stage2(features)  # [B, 1]
        stage3_logits = self.classifier_stage3(features)  # [B, 1]
        
        return {
            'stage1': stage1_logits,
            'stage2': stage2_logits,
            'stage3': stage3_logits
        }
    
    def predict(self, x, thresholds=(0.5, 0.5, 0.5)):
        """
        Predict grain class following hierarchical decision tree.
        
        Args:
            x: Input tensor [B, 3, H, W]
            thresholds: (t1, t2, t3) decision thresholds
        
        Returns:
            predictions: [B] - class indices (0=peloid, 1=ooid, 2=broken, 3=intraclast)
            probs: dict of probabilities for each stage
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            stage1_logits = logits['stage1']
            stage2_logits = logits['stage2']
            stage3_logits = logits['stage3']
            
            # Convert logits to probabilities
            stage1_probs = torch.sigmoid(stage1_logits).squeeze()  # [B]
            stage2_probs = torch.sigmoid(stage2_logits).squeeze()
            stage3_probs = torch.sigmoid(stage3_logits).squeeze()
            
            # Hierarchical decision
            t1, t2, t3 = thresholds
            batch_size = x.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
            for i in range(batch_size):
                if stage1_probs[i] >= t1:
                    # Peloid
                    predictions[i] = 0
                else:
                    # Ooid-like
                    if stage2_probs[i] >= t2:
                        # Ooid
                        if stage3_probs[i] >= t3:
                            # Whole ooid
                            predictions[i] = 1
                        else:
                            # Broken ooid
                            predictions[i] = 2
                    else:
                        # Intraclast
                        predictions[i] = 3
            
            probs = {
                'stage1': stage1_probs,
                'stage2': stage2_probs,
                'stage3': stage3_probs
            }
            
            return predictions, probs
    
    def get_predictions(self, logits):
        """
        Convert hierarchical logits to final class predictions (compatible with trainer).
        
        Follows the decision tree:
          Stage 1: If peloid (1) -> Peloid (0)
          Stage 1: If non-peloid (0):
              Stage 2: If intraclast (0) -> Intraclast (3)
              Stage 2: If ooid-like (1):
                  Stage 3: If whole (1) -> Ooid (1)
                  Stage 3: If broken (0) -> Broken ooid (2)
        
        Args:
            logits: Dictionary with stage1, stage2, stage3 logits
            
        Returns:
            Class predictions (batch_size,) with values:
                0: Peloid
                1: Ooid
                2: Broken ooid
                3: Intraclast
        """
        batch_size = logits['stage1'].size(0)
        predictions = torch.zeros(batch_size, dtype=torch.long, device=logits['stage1'].device)
        
        # Stage 1: Peloid vs Non-peloid
        stage1_pred = (torch.sigmoid(logits['stage1']).squeeze() > 0.5).long()
        
        # Stage 2: Ooid-like vs Intraclast (for non-peloids)
        stage2_pred = (torch.sigmoid(logits['stage2']).squeeze() > 0.5).long()
        
        # Stage 3: Whole vs Broken (for ooid-likes)
        stage3_pred = (torch.sigmoid(logits['stage3']).squeeze() > 0.5).long()
        
        # Apply hierarchical decision tree
        peloid_mask = stage1_pred == 1
        non_peloid_mask = stage1_pred == 0
        
        # Peloids
        predictions[peloid_mask] = 0
        
        # Non-peloids: separate into intraclasts and ooid-likes
        intraclast_mask = non_peloid_mask & (stage2_pred == 0)
        ooid_like_mask = non_peloid_mask & (stage2_pred == 1)
        
        # Intraclasts
        predictions[intraclast_mask] = 3
        
        # Ooid-likes: separate into whole ooids and broken ooids
        whole_ooid_mask = ooid_like_mask & (stage3_pred == 1)
        broken_ooid_mask = ooid_like_mask & (stage3_pred == 0)
        
        # Ooids
        predictions[whole_ooid_mask] = 1
        
        # Broken ooids
        predictions[broken_ooid_mask] = 2
        
        return predictions


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test model instantiation
    print("Testing EfficientNetHierarchicalClassifier...")
    
    model = EfficientNetHierarchicalClassifier(pretrained=False)
    total, trainable = count_parameters(model)
    
    print(f"\nModel: EfficientNetHierarchicalClassifier")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 96, 96)
    
    logits = model(x)
    
    print(f"\nForward pass test (batch_size={batch_size}):")
    print(f"  Input shape:  {x.shape}")
    print(f"  Stage 1 (Peloid vs Ooid-like): {logits['stage1'].shape}")
    print(f"  Stage 2 (Ooid vs Intraclast):  {logits['stage2'].shape}")
    print(f"  Stage 3 (Whole vs Broken):     {logits['stage3'].shape}")
    
    # Test prediction
    preds, probs = model.predict(x)
    print(f"\nPrediction test:")
    print(f"  Predictions: {preds}")
    print(f"  Stage1 probs shape: {probs['stage1'].shape}")
    
    print("\n✅ Model test passed!")
