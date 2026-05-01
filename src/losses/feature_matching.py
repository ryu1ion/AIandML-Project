"""Feature-matching distillation losses: L2 (on L2-normalized features) and cosine.

These are equivalent up to a factor of 2 when both inputs are L2-normalized:
  ||s - t||^2 = 2 - 2 cos(s, t)  for ||s|| = ||t|| = 1
so `l2_normalized_mse_loss == 2 * cosine_distance_loss` for unit-norm inputs.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def l2_normalized_mse_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Mean over batch of squared L2 distance between L2-normalized features.

    For each sample: || normalize(s) - normalize(t) ||^2  (sum over feature dim).
    Returns a scalar tensor.
    """
    s = F.normalize(student, dim=-1, p=2)
    t = F.normalize(teacher, dim=-1, p=2)
    return ((s - t) ** 2).sum(dim=-1).mean()


def cosine_distance_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Mean over batch of (1 - cosine_similarity)."""
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
