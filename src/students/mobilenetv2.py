"""MobileNetV2 student wrapper around timm.

Builds `mobilenetv2_100` with `num_classes=0` and global average pooling, so
forward returns a (B, 1280) embedding. We expose this under the project's
uniform `forward_features(x) -> Tensor` API.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor

DEFAULT_TIMM_NAME = "mobilenetv2_100"


class MobileNetV2Student(nn.Module):
    """Thin wrapper around timm's mobilenetv2_100 with no classifier head."""

    def __init__(
        self,
        pretrained: bool = False,
        timm_name: str = DEFAULT_TIMM_NAME,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feature_dim: int = int(self.backbone.num_features)

    def forward_features(self, x: Tensor) -> Tensor:
        """Return pooled backbone features of shape (B, feature_dim)."""
        return self.backbone(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def make_mobilenetv2_student(pretrained: bool = False) -> MobileNetV2Student:
    return MobileNetV2Student(pretrained=pretrained)
