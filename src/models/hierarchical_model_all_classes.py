"""
Seven-class hierarchical ResNet-18 model.

Hierarchy:
  Stage 1: Peloid vs non-peloid
  Stage 2: Carbonate target classes vs excluded classes
  Stage 3: Ooid-like vs intraclast
  Stage 4: Whole ooid vs broken ooid
  Stage 5: Fossil fragment vs quartz grain
  Stage 6: Ostracod vs bivalve
"""
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SevenClassHierarchicalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                resnet = models.resnet18(weights=weights)
            except AttributeError:
                resnet = models.resnet18(pretrained=True)
        else:
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head_stage1 = BinaryClassificationHead(dropout=dropout)
        self.head_stage2 = BinaryClassificationHead(dropout=dropout)
        self.head_stage3 = BinaryClassificationHead(dropout=dropout)
        self.head_stage4 = BinaryClassificationHead(dropout=dropout)
        self.head_stage5 = BinaryClassificationHead(dropout=dropout)
        self.head_stage6 = BinaryClassificationHead(dropout=dropout)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return {
            'stage1': self.head_stage1(features),
            'stage2': self.head_stage2(features),
            'stage3': self.head_stage3(features),
            'stage4': self.head_stage4(features),
            'stage5': self.head_stage5(features),
            'stage6': self.head_stage6(features),
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_predictions(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = logits['stage1'].shape[0]
        predictions = torch.zeros(batch_size, dtype=torch.long, device=logits['stage1'].device)

        s1 = (torch.sigmoid(logits['stage1']).squeeze(-1) > 0.5)
        s2 = (torch.sigmoid(logits['stage2']).squeeze(-1) > 0.5)
        s3 = (torch.sigmoid(logits['stage3']).squeeze(-1) > 0.5)
        s4 = (torch.sigmoid(logits['stage4']).squeeze(-1) > 0.5)
        s5 = (torch.sigmoid(logits['stage5']).squeeze(-1) > 0.5)
        s6 = (torch.sigmoid(logits['stage6']).squeeze(-1) > 0.5)

        peloid = s1
        non_peloid = ~s1
        carbonate = non_peloid & s2
        other = non_peloid & ~s2

        predictions[peloid] = 0
        predictions[carbonate & ~s3] = 3
        predictions[carbonate & s3 & s4] = 1
        predictions[carbonate & s3 & ~s4] = 2
        predictions[other & ~s5] = 6
        predictions[other & s5 & s6] = 4
        predictions[other & s5 & ~s6] = 5

        return predictions

    def get_class_scores(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Approximate seven-class probabilities from routed binary probabilities."""
        p1 = torch.sigmoid(logits['stage1']).squeeze(-1)
        p2 = torch.sigmoid(logits['stage2']).squeeze(-1)
        p3 = torch.sigmoid(logits['stage3']).squeeze(-1)
        p4 = torch.sigmoid(logits['stage4']).squeeze(-1)
        p5 = torch.sigmoid(logits['stage5']).squeeze(-1)
        p6 = torch.sigmoid(logits['stage6']).squeeze(-1)

        scores = torch.stack([
            p1,
            (1 - p1) * p2 * p3 * p4,
            (1 - p1) * p2 * p3 * (1 - p4),
            (1 - p1) * p2 * (1 - p3),
            (1 - p1) * (1 - p2) * p5 * p6,
            (1 - p1) * (1 - p2) * p5 * (1 - p6),
            (1 - p1) * (1 - p2) * (1 - p5),
        ], dim=1)
        return scores
