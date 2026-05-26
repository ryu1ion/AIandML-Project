"""FitNet (Romero 2015) intermediate-feature distillation.

- 1x1 conv `FitNetAdapter` maps student mid-feature channels to teacher channels.
- `FitNetLoss` = beta * ||adapter(s_mid) - t_mid||^2 + CE(student_logits, y)

NEW_BENCH R6: ResNet-50 `layer3` (1024ch, 14x14 @ 224) <-> MobileNetV2
`features[14]` (~96ch, 14x14 @ 224). The 1x1 adapter handles the channel diff.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FitNetAdapter(nn.Module):
    """1x1 conv with He init to project student mid-feats to teacher dim."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class FitNetLoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(
        self,
        student_mid: Tensor,        # (B, C_s_out, H, W) after adapter
        teacher_mid: Tensor,        # (B, C_t,    H, W)
        student_logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if student_mid.shape != teacher_mid.shape:
            raise ValueError(
                f"student_mid {tuple(student_mid.shape)} != teacher_mid "
                f"{tuple(teacher_mid.shape)} -- check adapter / spatial alignment"
            )
        # Mean over batch of per-sample sum-squared-difference; element-mean is
        # simpler and equivalent up to a constant for fixed C*H*W.
        l_hint = F.mse_loss(student_mid, teacher_mid.detach(), reduction="mean")
        l_ce = F.cross_entropy(student_logits, labels)
        loss = self.beta * l_hint + l_ce
        return loss, {"L_hint": l_hint.detach(), "L_ce": l_ce.detach()}
