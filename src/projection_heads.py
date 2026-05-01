"""Projection heads mapping student features to teacher feature space."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MLPProjectionHead(nn.Module):
    """2-layer MLP: Linear(in -> hidden) -> BN -> GELU -> Linear(hidden -> out).

    Hidden dim defaults to 2 * out_dim (the project default).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else 2 * out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = h
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
