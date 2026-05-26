"""Forward-hook helpers to capture intermediate feature maps for FitNet.

NEW_BENCH R6 layer choice: ResNet-50 `layer3` (1024ch, 14x14 @ 224x224) <->
timm MobileNetV2 `blocks[4]` (96ch, 14x14 @ 224x224). Note: NEW_BENCH text
references torchvision's `features[14]`; the equivalent in timm's
`mobilenetv2_100` (an `EfficientNet`-style module tree) is `blocks[4]`.
Both produce the same 14x14 spatial map at 224x224 input.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MidFeatureGrabber:
    """Forward-hook wrapper that exposes the most recent activation of a module.

    Usage:
        grab = MidFeatureGrabber(model.blocks[4])
        _ = model(x)                # any forward pass through the parent model
        feat = grab.feat            # (B, C, H, W)
        grab.close()                # unregister when done

    The hook stores the raw tensor (no clone), so `feat` becomes stale after
    the next forward pass. Re-read it after each forward.
    """

    def __init__(self, module: nn.Module) -> None:
        self.feat: Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs, output: Tensor) -> None:
        self.feat = output

    def close(self) -> None:
        self._handle.remove()


def get_mobilenetv2_mid_module(student_backbone: nn.Module) -> nn.Module:
    """Return the timm MNV2 block whose output is the 14x14 / ~96-channel stage.

    Accepts either the timm model directly or a `MobileNetV2Student` wrapper
    (whose underlying timm model lives at `.backbone`).
    """
    m = getattr(student_backbone, "backbone", student_backbone)
    if hasattr(m, "blocks"):
        return m.blocks[4]
    raise AttributeError(
        "Expected a timm MobileNetV2 (with .blocks); got "
        f"{type(student_backbone).__name__}"
    )


def get_resnet50_mid_module(teacher: nn.Module) -> nn.Module:
    """Return the ResNet-50 `layer3` module (1024ch, 14x14 @ 224)."""
    m = getattr(teacher, "backbone", teacher)
    if hasattr(m, "layer3"):
        return m.layer3
    raise AttributeError(
        f"Expected a ResNet-style model with .layer3; got {type(teacher).__name__}"
    )
