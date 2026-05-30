"""Unit tests for the proposed structural distillation losses and the
paper-backed reference losses (SP-KD, RKD-distance).

Run from repo root:  python -m pytest tests/test_losses.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.losses.feature_matching import l2_normalized_mse_loss
from src.losses.ours_distill_loss import (
    CrossViewInvariantLoss,
    GlobalSemanticLoss,
    LocalStructuralLoss,
    OursDistillLoss,
)
from src.losses.paper_kd_losses import (
    distill_total_loss,
    normalized_mse_loss,
    rkd_distance_loss,
    similarity_preserving_loss,
)


# ---- Proposed structural distillation losses ----


def test_local_mse_zero_when_identical() -> None:
    torch.manual_seed(10)
    patches = torch.randn(4, 49, 128)
    loss_fn = LocalStructuralLoss(mode="mse")
    loss = loss_fn(patches.clone(), patches.clone())
    assert loss.abs().item() < 1e-5


def test_local_kl_zero_when_identical() -> None:
    torch.manual_seed(11)
    patches = torch.randn(4, 49, 128)
    loss_fn = LocalStructuralLoss(mode="kl", temperature=0.1)
    loss = loss_fn(patches.clone(), patches.clone())
    assert loss.abs().item() < 1e-4


def test_local_different_dims_ok() -> None:
    torch.manual_seed(12)
    s = torch.randn(4, 49, 1280)
    t = torch.randn(4, 49, 384)
    loss_fn = LocalStructuralLoss(mode="mse")
    loss = loss_fn(s, t)
    assert torch.isfinite(loss)


def test_global_kl_zero_when_identical() -> None:
    torch.manual_seed(13)
    feats = torch.randn(8, 128)
    loss_fn = GlobalSemanticLoss(mode="kl", temperature=0.1)
    loss = loss_fn(feats.clone(), feats.clone())
    assert loss.abs().item() < 1e-4


def test_global_mse_zero_when_identical() -> None:
    torch.manual_seed(14)
    feats = torch.randn(8, 128)
    loss_fn = GlobalSemanticLoss(mode="mse")
    loss = loss_fn(feats.clone(), feats.clone())
    assert loss.abs().item() < 1e-5


def test_crossview_kl_zero_when_identical() -> None:
    torch.manual_seed(15)
    v1 = torch.randn(8, 128)
    v2 = torch.randn(8, 128)
    loss_fn = CrossViewInvariantLoss(mode="kl", temperature=0.1)
    loss = loss_fn(v1.clone(), v2.clone(), v1.clone(), v2.clone())
    assert loss.abs().item() < 1e-4


def test_ours_no_grad_to_teacher_features() -> None:
    torch.manual_seed(16)
    s_g = torch.randn(4, 128, requires_grad=True)
    t_g = torch.randn(4, 128, requires_grad=True)
    loss_fn = OursDistillLoss(lambda_local=0.0, lambda_global=0.5, lambda_view=0.5)
    comps = loss_fn(
        s_global_v1=s_g,
        s_global_v2=s_g.clone().requires_grad_(True),
        t_global_v1=t_g.detach(),
        t_global_v2=t_g.detach(),
        step=0,
    )
    comps["loss_ours"].backward()
    assert s_g.grad is not None and s_g.grad.abs().sum().item() > 0
    assert t_g.grad is None or t_g.grad.abs().sum().item() == 0


def test_ours_all_losses_finite() -> None:
    torch.manual_seed(17)
    B, N, Cs, Ct = 4, 49, 1280, 384
    loss_fn = OursDistillLoss()
    comps = loss_fn(
        s_patch_v1=torch.randn(B, N, Cs),
        s_patch_v2=torch.randn(B, N, Cs),
        t_patch_v1=torch.randn(B, N, Ct),
        t_patch_v2=torch.randn(B, N, Ct),
        s_global_v1=torch.randn(B, Cs),
        s_global_v2=torch.randn(B, Cs),
        t_global_v1=torch.randn(B, Ct),
        t_global_v2=torch.randn(B, Ct),
        step=0,
    )
    for k, v in comps.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"


def test_ours_warmup() -> None:
    torch.manual_seed(18)
    B, Cs, Ct = 4, 128, 64
    loss_fn = OursDistillLoss(warmup_steps=100, lambda_global=1.0, lambda_local=0.0, lambda_view=0.0)
    comps_0 = loss_fn(
        s_global_v1=torch.randn(B, Cs), s_global_v2=torch.randn(B, Cs),
        t_global_v1=torch.randn(B, Ct), t_global_v2=torch.randn(B, Ct), step=0,
    )
    comps_50 = loss_fn(
        s_global_v1=torch.randn(B, Cs), s_global_v2=torch.randn(B, Cs),
        t_global_v1=torch.randn(B, Ct), t_global_v2=torch.randn(B, Ct), step=50,
    )
    assert comps_0["warmup_factor"].item() == 0.0
    assert abs(comps_50["warmup_factor"].item() - 0.5) < 1e-5


# ---- Paper-backed KD losses (SP-KD, RKD-D) ----


def test_sp_zero_when_student_equals_teacher() -> None:
    torch.manual_seed(20)
    feats = torch.randn(8, 384)
    loss = similarity_preserving_loss(feats.clone(), feats.clone())
    assert loss.abs().item() < 1e-6


def test_sp_finite_random() -> None:
    torch.manual_seed(21)
    s = torch.randn(8, 384)
    t = torch.randn(8, 384)
    loss = similarity_preserving_loss(s, t)
    assert torch.isfinite(loss)
    assert loss.dim() == 0  # scalar


def test_sp_no_teacher_grad() -> None:
    torch.manual_seed(22)
    s = torch.randn(8, 384, requires_grad=True)
    t = torch.randn(8, 384, requires_grad=True)
    loss = similarity_preserving_loss(s, t)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0
    assert t.grad is None or t.grad.abs().sum().item() == 0


def test_rkd_zero_when_student_equals_teacher() -> None:
    torch.manual_seed(23)
    feats = torch.randn(8, 384)
    loss = rkd_distance_loss(feats.clone(), feats.clone())
    assert loss.abs().item() < 1e-6


def test_rkd_finite_random() -> None:
    torch.manual_seed(24)
    s = torch.randn(8, 384)
    t = torch.randn(8, 384)
    loss = rkd_distance_loss(s, t)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_rkd_no_teacher_grad() -> None:
    torch.manual_seed(25)
    s = torch.randn(8, 384, requires_grad=True)
    t = torch.randn(8, 384, requires_grad=True)
    loss = rkd_distance_loss(s, t)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0
    assert t.grad is None or t.grad.abs().sum().item() == 0


def test_rkd_safe_when_batch_size_below_two() -> None:
    """A single-sample "batch" has no pairwise distances; loss should be 0."""
    torch.manual_seed(26)
    s = torch.randn(1, 384, requires_grad=True)
    t = torch.randn(1, 384)
    loss = rkd_distance_loss(s, t)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_distill_total_lambda_zero_reproduces_base() -> None:
    """L_total at lambda_sp=lambda_rkd=0 must equal the base normalized-MSE."""
    torch.manual_seed(27)
    s = torch.randn(8, 384)
    t = torch.randn(8, 384)
    components = distill_total_loss(s, t, lambda_sp=0.0, lambda_rkd=0.0)
    base_ref = l2_normalized_mse_loss(s, t)
    assert torch.allclose(components["loss_total"], base_ref, atol=1e-7)
    assert torch.allclose(components["loss_base"], base_ref, atol=1e-7)
    assert components["loss_sp"].item() == 0.0
    assert components["loss_rkd_distance"].item() == 0.0


def test_normalized_mse_matches_existing_base() -> None:
    """paper_kd_losses.normalized_mse_loss must match the project's existing
    l2_normalized_mse_loss bit-for-bit."""
    torch.manual_seed(28)
    s = torch.randn(8, 384)
    t = torch.randn(8, 384)
    a = normalized_mse_loss(s, t)
    b = l2_normalized_mse_loss(s, t)
    assert torch.allclose(a, b, atol=1e-7)


if __name__ == "__main__":
    import subprocess

    raise SystemExit(
        subprocess.call(["python", "-m", "pytest", __file__, "-v", "-s"])
    )
