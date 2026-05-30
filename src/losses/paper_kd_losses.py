"""Paper-backed distillation losses added on top of the base normalized-MSE.

Two losses are implemented here:

1. SP-KD: Similarity-Preserving Knowledge Distillation
   Tung & Mori, ICCV 2019. https://arxiv.org/abs/1907.09682

2. RKD-D: Relational Knowledge Distillation (distance-wise)
   Park et al., CVPR 2019. https://arxiv.org/abs/1904.05068

Both are *additive* terms; the base loss is unchanged. See
``distill_total_loss`` for the combined objective:

    L_total = L_base + lambda_sp * L_sp + lambda_rkd * L_rkd

The functions deliberately take pre-computed (s, t) feature tensors so they
are reusable for either single-view or per-view (averaged) computation; the
trainer wraps them per-view.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-12


def normalized_mse_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Re-export of the base L2-on-L2-normalized loss (same as
    ``src.losses.feature_matching.l2_normalized_mse_loss``), kept here so the
    "distill_total_loss" helper has all components in one module."""
    s = F.normalize(student, dim=-1, p=2)
    t = F.normalize(teacher, dim=-1, p=2)
    return ((s - t) ** 2).sum(dim=-1).mean()


def similarity_preserving_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Similarity-Preserving Knowledge Distillation, ICCV 2019.

    Computes the row-normalized batch-similarity (Gram) matrices for student
    and teacher features and matches them with MSE. The base loss aligns each
    student sample to its corresponding teacher sample independently; SP-KD
    additionally preserves the teacher's batch-level semantic geometry.

    Args:
        student: (B, D) student feature batch (with gradient).
        teacher: (B, D) teacher feature batch (will be detached internally).

    Returns:
        Scalar loss tensor.
    """
    t = teacher.detach()
    s_norm = F.normalize(student, dim=-1)
    t_norm = F.normalize(t, dim=-1)
    G_s = s_norm @ s_norm.T
    G_t = t_norm @ t_norm.T
    return F.mse_loss(G_s, G_t)


def rkd_distance_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Relational Knowledge Distillation, distance-wise, CVPR 2019.

    Matches the (mean-normalized) pairwise Euclidean-distance matrix between
    student and teacher feature batches with Smooth L1. Distance-wise RKD is
    the simpler / more stable of the two RKD variants (the angle-wise one is
    not implemented here).

    Args:
        student: (B, D) student feature batch (with gradient).
        teacher: (B, D) teacher feature batch (will be detached internally).

    Returns:
        Scalar loss tensor. Returns zero (safely) if B < 2.
    """
    B = student.shape[0]
    if B < 2:
        return student.sum() * 0.0  # safe zero with gradient path

    t = teacher.detach()
    d_s = torch.cdist(student, student, p=2)
    d_t = torch.cdist(t, t, p=2)

    # Normalize by the mean of strictly-positive (off-diagonal) entries.
    # Use a mask instead of (d > 0) thresholding to be robust to tiny
    # off-diagonals in low-precision arithmetic.
    eye = torch.eye(B, dtype=torch.bool, device=student.device)
    mean_s = d_s[~eye].mean().clamp_min(_EPS)
    mean_t = d_t[~eye].mean().clamp_min(_EPS)
    d_s = d_s / mean_s
    d_t = d_t / mean_t

    return F.smooth_l1_loss(d_s, d_t)


def distill_total_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    lambda_sp: float = 1.0,
    lambda_rkd: float = 0.5,
) -> dict[str, Tensor]:
    """Compute base + SP-KD + RKD-D for a single view.

    Returns a dict with the individual loss terms (for logging) and the
    weighted total under "loss_total". Each component is a scalar tensor.
    """
    loss_base = normalized_mse_loss(student, teacher)
    loss_sp = similarity_preserving_loss(student, teacher) if lambda_sp > 0 \
        else torch.zeros((), device=student.device, dtype=student.dtype)
    loss_rkd = rkd_distance_loss(student, teacher) if lambda_rkd > 0 \
        else torch.zeros((), device=student.device, dtype=student.dtype)
    loss_total = loss_base + lambda_sp * loss_sp + lambda_rkd * loss_rkd
    return {
        "loss_base": loss_base,
        "loss_sp": loss_sp,
        "loss_rkd_distance": loss_rkd,
        "loss_total": loss_total,
    }
