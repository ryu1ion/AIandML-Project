"""Debug script: verify the 'ours' distillation pipeline end-to-end.

Runs a few iterations on synthetic data and checks:
- Forward pass produces finite outputs
- All three auxiliary losses are finite
- Gradients flow into the student (and head/patch_proj)
- No gradients flow into the teacher
- Feature shapes match expectations

Usage:
  python scripts/debug_ours.py
  python scripts/debug_ours.py --use-patch-proj
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.losses.feature_hooks import MidFeatureGrabber, get_mobilenetv2_spatial_module
from src.losses.feature_matching import l2_normalized_mse_loss
from src.losses.ours_distill_loss import OursDistillLoss
from src.projection_heads import MLPProjectionHead, PatchProjectionHead
from src.students import get_student
from src.teachers import get_teacher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--use-patch-proj", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    student = get_student("mobilenetv2").to(device)
    teacher = get_teacher("dino_vits16").to(device).eval()
    head = MLPProjectionHead(student.feature_dim, out_dim=384, hidden_dim=768).to(device)

    spatial_hook = MidFeatureGrabber(get_mobilenetv2_spatial_module(student))

    patch_proj = None
    if args.use_patch_proj:
        patch_proj = PatchProjectionHead(student.feature_dim, out_dim=384).to(device)
        print("Using patch projection head")

    loss_fn = OursDistillLoss(
        lambda_local=1.0,
        lambda_global=0.5,
        lambda_view=0.5,
        local_temperature=0.1,
        global_temperature=0.1,
        view_temperature=0.1,
        local_mode="mse",
        global_mode="kl",
        view_mode="kl",
    )

    params = list(student.parameters()) + list(head.parameters())
    if patch_proj is not None:
        params += list(patch_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    B = args.batch_size
    print(f"\n=== Shape sanity check (B={B}) ===")
    x1 = torch.randn(B, 3, 224, 224, device=device)
    x2 = torch.randn(B, 3, 224, 224, device=device)

    with torch.no_grad():
        t_cls_v1, t_patch_v1 = teacher.forward_patch_features(x1)
        print(f"Teacher CLS:   {t_cls_v1.shape}")
        print(f"Teacher patch: {t_patch_v1.shape}")
        assert t_cls_v1.shape == (B, 384), f"Expected (B, 384), got {t_cls_v1.shape}"
        assert t_patch_v1.shape == (B, 196, 384), f"Expected (B, 196, 384), got {t_patch_v1.shape}"

    s_pooled = student(x1)
    s_spatial = spatial_hook.feat
    print(f"Student pooled:  {s_pooled.shape}")
    print(f"Student spatial: {s_spatial.shape}")
    assert s_pooled.shape == (B, 1280), f"Expected (B, 1280), got {s_pooled.shape}"
    assert s_spatial.shape == (B, 1280, 7, 7), f"Expected (B, 1280, 7, 7), got {s_spatial.shape}"

    s_up = F.interpolate(s_spatial, size=(14, 14), mode="bilinear", align_corners=False)
    s_tokens = s_up.flatten(2).transpose(1, 2)
    print(f"Student tokens (after interp+flatten): {s_tokens.shape}")
    assert s_tokens.shape == (B, 196, 1280)

    if patch_proj is not None:
        s_tokens = patch_proj(s_tokens)
        print(f"Student tokens (after proj): {s_tokens.shape}")
        assert s_tokens.shape == (B, 196, 384)

    # Teacher grad check
    for p_t in teacher.parameters():
        assert not p_t.requires_grad, "Teacher params should be frozen!"
    print("\nTeacher is frozen: OK")

    print(f"\n=== Training loop ({args.steps} steps) ===")
    student.train()
    head.train()
    if patch_proj is not None:
        patch_proj.train()

    for step in range(args.steps):
        x1 = torch.randn(B, 3, 224, 224, device=device)
        x2 = torch.randn(B, 3, 224, 224, device=device)

        optimizer.zero_grad()

        s_pooled_v1 = student(x1)
        s_spatial_v1 = spatial_hook.feat.clone()
        s_pooled_v2 = student(x2)
        s_spatial_v2 = spatial_hook.feat

        s_global_v1 = head(s_pooled_v1)
        s_global_v2 = head(s_pooled_v2)

        with torch.no_grad():
            t_cls_v1, t_patch_v1 = teacher.forward_patch_features(x1)
            t_cls_v2, t_patch_v2 = teacher.forward_patch_features(x2)

        loss_base = 0.5 * (
            l2_normalized_mse_loss(s_global_v1, t_cls_v1)
            + l2_normalized_mse_loss(s_global_v2, t_cls_v2)
        )

        s_up_v1 = F.interpolate(s_spatial_v1, size=(14, 14), mode="bilinear", align_corners=False)
        s_up_v2 = F.interpolate(s_spatial_v2, size=(14, 14), mode="bilinear", align_corners=False)
        s_patch_v1 = s_up_v1.flatten(2).transpose(1, 2)
        s_patch_v2 = s_up_v2.flatten(2).transpose(1, 2)

        if patch_proj is not None:
            s_patch_v1 = patch_proj(s_patch_v1)
            s_patch_v2 = patch_proj(s_patch_v2)

        comps = loss_fn(
            s_patch_v1=s_patch_v1,
            s_patch_v2=s_patch_v2,
            t_patch_v1=t_patch_v1,
            t_patch_v2=t_patch_v2,
            s_global_v1=s_global_v1,
            s_global_v2=s_global_v2,
            t_global_v1=t_cls_v1,
            t_global_v2=t_cls_v2,
            step=step,
        )

        loss = loss_base + comps["loss_ours"]
        loss.backward()

        # Check finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        for name in ("loss_local", "loss_global", "loss_view", "loss_ours"):
            v = comps[name]
            assert torch.isfinite(v), f"{name} is not finite: {v.item()}"

        # Check student grads exist
        student_grad_norm = sum(
            p_s.grad.norm().item() for p_s in student.parameters() if p_s.grad is not None
        )
        assert student_grad_norm > 0, "No gradients in student!"

        head_grad_norm = sum(
            p_h.grad.norm().item() for p_h in head.parameters() if p_h.grad is not None
        )
        assert head_grad_norm > 0, "No gradients in head!"

        # Check teacher has no grads
        teacher_grad_any = any(
            p_t.grad is not None and p_t.grad.norm().item() > 0
            for p_t in teacher.parameters()
        )
        assert not teacher_grad_any, "Teacher has gradients!"

        optimizer.step()

        print(
            f"  step {step}: "
            f"base={loss_base.item():.4f} "
            f"local={comps['loss_local'].item():.4f} "
            f"global={comps['loss_global'].item():.4f} "
            f"view={comps['loss_view'].item():.4f} "
            f"total={loss.item():.4f} "
            f"student_grad={student_grad_norm:.4f} "
            f"head_grad={head_grad_norm:.4f}"
        )

    spatial_hook.close()
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
