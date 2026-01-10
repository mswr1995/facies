"""
Hierarchical model architecture for carbonate grain classification.

Three-stage binary classification:
  Stage 1: Peloid vs Non-peloid
  Stage 2: Ooid-like vs Intraclast (for non-peloids)
  Stage 3: Whole Ooid vs Broken Ooid (for ooid-likes)
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple, Optional


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head with MLP architecture.
    
    Architecture: input_dim -> hidden_dim -> 1 (logit)
    Includes ReLU activation and Dropout for regularization.
    """
    
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.3):
        super(BinaryClassificationHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Logit for binary classification (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class HierarchicalGrainClassifier(nn.Module):
    """
    Hierarchical classifier for carbonate grain classification.
    
    Uses a shared ResNet-18 backbone with three independent binary classification heads.
    Each head corresponds to one stage in the hierarchical decision tree.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights for ResNet-18
        dropout: Dropout rate for classification heads
        freeze_backbone: Whether to freeze backbone weights initially
    """
    
    def __init__(self, pretrained=True, dropout=0.3, freeze_backbone=False):
        super(HierarchicalGrainClassifier, self).__init__()
        
        # Load pretrained ResNet-18 and remove final FC layer
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension after global average pooling
        feature_dim = 512
        
        # Three independent classification heads
        self.head_stage1 = BinaryClassificationHead(
            input_dim=feature_dim,
            hidden_dim=128,
            dropout=dropout
        )
        
        self.head_stage2 = BinaryClassificationHead(
            input_dim=feature_dim,
            hidden_dim=128,
            dropout=dropout
        )
        
        self.head_stage3 = BinaryClassificationHead(
            input_dim=feature_dim,
            hidden_dim=128,
            dropout=dropout
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input images (batch_size, 3, 96, 96)
            
        Returns:
            Dictionary containing logits for each stage:
                'stage1': Peloid vs Non-peloid logits (batch_size, 1)
                'stage2': Ooid-like vs Intraclast logits (batch_size, 1)
                'stage3': Whole vs Broken Ooid logits (batch_size, 1)
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Get predictions from each head
        logits_stage1 = self.head_stage1(features)
        logits_stage2 = self.head_stage2(features)
        logits_stage3 = self.head_stage3(features)
        
        return {
            'stage1': logits_stage1,
            'stage2': logits_stage2,
            'stage3': logits_stage3
        }
    
    def freeze_backbone(self):
        """Freeze all backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_early_layers(self, num_layers=6):
        """
        Freeze early layers of ResNet for gradual unfreezing strategy.
        
        Args:
            num_layers: Number of ResNet blocks to freeze (0-8)
        """
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def get_predictions(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert hierarchical logits to final class predictions.
        
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
    
    def count_parameters(self, trainable_only=False):
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def print_parameter_summary(self):
        """Print summary of model parameters."""
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        frozen_params = total_params - trainable_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        head1_params = sum(p.numel() for p in self.head_stage1.parameters())
        head2_params = sum(p.numel() for p in self.head_stage2.parameters())
        head3_params = sum(p.numel() for p in self.head_stage3.parameters())
        
        print("Model Parameter Summary:")
        print("=" * 60)
        print(f"Total parameters:        {total_params:,}")
        print(f"Trainable parameters:    {trainable_params:,}")
        print(f"Frozen parameters:       {frozen_params:,}")
        print()
        print(f"Backbone (ResNet-18):    {backbone_params:,} ({backbone_trainable:,} trainable)")
        print(f"Stage 1 Head:            {head1_params:,}")
        print(f"Stage 2 Head:            {head2_params:,}")
        print(f"Stage 3 Head:            {head3_params:,}")
        print("=" * 60)


if __name__ == '__main__':
    """Test the model architecture."""
    print("Testing Hierarchical Grain Classifier")
    print("=" * 60)
    
    # Create model
    model = HierarchicalGrainClassifier(pretrained=False, freeze_backbone=False)
    
    # Print parameter summary
    model.print_parameter_summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 96, 96)
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Stage 1 logits shape: {logits['stage1'].shape}")
    print(f"Stage 2 logits shape: {logits['stage2'].shape}")
    print(f"Stage 3 logits shape: {logits['stage3'].shape}")
    
    # Test prediction conversion
    predictions = model.get_predictions(logits)
    print(f"Final predictions shape: {predictions.shape}")
    print(f"Final predictions: {predictions}")
    
    # Test freezing/unfreezing
    print("\nTesting freeze/unfreeze functionality...")
    print(f"Initial trainable params: {model.count_parameters(trainable_only=True):,}")
    
    model.freeze_backbone()
    print(f"After freezing backbone: {model.count_parameters(trainable_only=True):,}")
    
    model.unfreeze_backbone()
    print(f"After unfreezing backbone: {model.count_parameters(trainable_only=True):,}")
    
    print("\n" + "=" * 60)
    print("Model test completed successfully!")
