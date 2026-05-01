"""SimSiam projector + predictor + symmetric negative-cosine loss.

Reference: Chen & He, "Exploring Simple Siamese Representation Learning", CVPR 2021.

Projector: 3-layer MLP with BN, no affine on the final BN.
Predictor: 2-layer MLP with bottleneck (hidden 512, output 2048).
Loss: D(p1, sg(z2))/2 + D(p2, sg(z1))/2 with D = -cos.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimSiamProjector(nn.Module):
    """3-layer MLP with BN (final BN has no affine), per the SimSiam paper."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SimSiamPredictor(nn.Module):
    """2-layer MLP with bottleneck (default in=2048, hidden=512, out=2048)."""

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        out_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def simsiam_loss(p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
    """Symmetric negative-cosine loss with stop-gradient on z."""
    z1 = z1.detach()
    z2 = z2.detach()
    d1 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    d2 = -F.cosine_similarity(p2, z1, dim=-1).mean()
    return 0.5 * (d1 + d2)
