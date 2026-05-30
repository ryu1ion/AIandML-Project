"""Forward-hook helper to capture the pre-pool spatial feature map of MobileNetV2.

The proposed method's Local Structural Distillation reads the
pre-global-pool spatial feature map ``(B, 1280, 7, 7)`` produced by
MobileNetV2's ``bn2`` layer; it is then upsampled to match the teacher's
14×14 patch grid and used to build the patch-to-patch similarity matrix.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MidFeatureGrabber:
    """Forward-hook wrapper that exposes the most recent activation of a module.

    Usage::

        grab = MidFeatureGrabber(model.bn2)
        _ = model(x)        # any forward pass through the parent model
        feat = grab.feat    # (B, C, H, W)
        grab.close()        # unregister when done

    The hook stores the raw tensor (no clone), so ``feat`` becomes stale after
    the next forward pass. Re-read it after each forward.
    """

    def __init__(self, module: nn.Module) -> None:
        self.feat: Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs, output: Tensor) -> None:
        self.feat = output

    def close(self) -> None:
        self._handle.remove()


def get_mobilenetv2_spatial_module(student_backbone: nn.Module) -> nn.Module:
    """Return the timm MNV2 ``bn2`` module whose output is (B, 1280, 7, 7) at 224."""
    m = getattr(student_backbone, "backbone", student_backbone)
    if hasattr(m, "bn2"):
        return m.bn2
    raise AttributeError(
        "Expected a timm MobileNetV2 (with .bn2); got "
        f"{type(student_backbone).__name__}"
    )
