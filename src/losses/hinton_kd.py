"""Hinton (2015) knowledge-distillation loss.

L_kd = T^2 * KL( softmax(t/T) || softmax(s/T) )
L_ce = CE(s, y)
L    = alpha * L_kd + (1 - alpha) * L_ce

Standard defaults: T=4, alpha=0.9 (NEW_BENCH R5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HintonKDLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.9) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.T = float(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        T = self.T
        # KL(teacher || student) in nats, batchmean. Multiply by T^2 to keep
        # gradient magnitudes comparable to L_ce (Hinton 2015).
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits.detach() / T, dim=-1)
        l_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        l_ce = F.cross_entropy(student_logits, labels)
        loss = self.alpha * l_kd + (1.0 - self.alpha) * l_ce
        return loss, {"L_kd": l_kd.detach(), "L_ce": l_ce.detach()}
