"""ResNet-50 used as R_teacher (NEW_BENCH Step 2).

Trained via the existing supervised task with `student=resnet50`. The classifier
head is the trainer's `_SupervisedHead`, which takes pooled features.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor


class ResNet50Backbone(nn.Module):
    """timm ResNet-50 with global-avg-pooled features, no classifier head."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feature_dim: int = int(self.backbone.num_features)  # 2048

    def forward_features(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def make_resnet50(pretrained: bool = False) -> ResNet50Backbone:
    return ResNet50Backbone(pretrained=pretrained)
