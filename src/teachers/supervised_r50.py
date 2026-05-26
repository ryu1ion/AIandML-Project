"""Supervised ResNet-50 teacher (R_teacher) loader for R5/R6.

Loads a checkpoint produced by training with `task=supervised, student=resnet50`
and exposes a unified API for both Hinton KD (R5) and FitNet (R6):

    teacher = load_supervised_resnet50(ckpt_path, num_classes=100)
    logits  = teacher.classify(x)        # (B, num_classes)
    mid     = teacher.mid_features(x)    # (B, 1024, 14, 14)  - layer3 output

Both calls run in eval mode under `torch.no_grad()`. The classifier head
weights come from `_SupervisedHead` (single nn.Linear) saved in the checkpoint
under `head_state_dict`.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.feature_hooks import MidFeatureGrabber, get_resnet50_mid_module
from src.students.resnet50 import ResNet50Backbone


class SupervisedResNet50Teacher(nn.Module):
    """Frozen, eval-mode supervised ResNet-50 with classifier + layer3 access."""

    def __init__(self, backbone: ResNet50Backbone, classifier: nn.Linear) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._mid_hook = MidFeatureGrabber(get_resnet50_mid_module(self.backbone))

    def train(self, mode: bool = True) -> "SupervisedResNet50Teacher":  # type: ignore[override]
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def classify(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)               # (B, 2048) (also fires layer3 hook)
        return self.classifier(feats)

    @torch.no_grad()
    def mid_features(self, x: Tensor) -> Tensor:
        _ = self.backbone(x)                   # populates layer3 hook
        assert self._mid_hook.feat is not None
        return self._mid_hook.feat

    @torch.no_grad()
    def classify_and_mid(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Single forward pass returning (logits, layer3-feature)."""
        feats = self.backbone(x)
        logits = self.classifier(feats)
        assert self._mid_hook.feat is not None
        return logits, self._mid_hook.feat


def load_supervised_resnet50(
    checkpoint_path: str | Path,
    num_classes: int = 100,
    device: torch.device | str | None = None,
) -> SupervisedResNet50Teacher:
    """Load a `task=supervised, student=resnet50` checkpoint as a frozen teacher."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = ResNet50Backbone(pretrained=False)
    backbone.load_state_dict(ck["student_state_dict"])
    head_sd = ck.get("head_state_dict")
    if head_sd is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no head_state_dict; "
            "expected a supervised R-50 trained with task=supervised."
        )
    # _SupervisedHead stores its Linear under `fc.*`.
    weight = head_sd["fc.weight"]
    bias = head_sd["fc.bias"]
    classifier = nn.Linear(weight.shape[1], weight.shape[0])
    classifier.weight.data.copy_(weight)
    classifier.bias.data.copy_(bias)
    if weight.shape[0] != num_classes:
        raise ValueError(
            f"Checkpoint classifier has {weight.shape[0]} classes but "
            f"num_classes={num_classes} was requested."
        )
    teacher = SupervisedResNet50Teacher(backbone, classifier)
    if device is not None:
        teacher = teacher.to(device)
    return teacher
