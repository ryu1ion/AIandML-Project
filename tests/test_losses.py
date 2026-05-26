"""Unit tests for Hinton KD + FitNet (NEW_BENCH Step 1).

Run from repo root:  python -m pytest tests/test_losses.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import timm
import torch
import torch.nn as nn

from src.losses.feature_hooks import (
    MidFeatureGrabber,
    get_mobilenetv2_mid_module,
    get_resnet50_mid_module,
)
from src.losses.fitnet import FitNetAdapter, FitNetLoss
from src.losses.hinton_kd import HintonKDLoss


def test_hinton_zero_when_student_equals_teacher() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 100)
    labels = torch.randint(0, 100, (8,))
    loss_fn = HintonKDLoss(temperature=4.0, alpha=1.0)  # alpha=1 -> only L_kd
    loss, comps = loss_fn(logits.clone(), logits.clone(), labels)
    assert comps["L_kd"].abs().item() < 1e-5, comps["L_kd"]


def test_hinton_components_are_nonneg() -> None:
    torch.manual_seed(1)
    s = torch.randn(4, 10)
    t = torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    loss, comps = HintonKDLoss()(s, t, y)
    assert comps["L_kd"].item() >= 0.0
    assert comps["L_ce"].item() >= 0.0
    assert loss.item() >= 0.0


def test_hinton_gradient_does_not_flow_to_teacher() -> None:
    torch.manual_seed(2)
    s = torch.randn(4, 10, requires_grad=True)
    t = torch.randn(4, 10, requires_grad=True)
    y = torch.randint(0, 10, (4,))
    loss, _ = HintonKDLoss()(s, t, y)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0
    # Teacher logits are detached inside the loss -> no grad accumulates.
    assert t.grad is None or t.grad.abs().sum().item() == 0


def test_fitnet_hint_zero_when_features_equal() -> None:
    torch.manual_seed(3)
    feat = torch.randn(2, 1024, 14, 14)
    logits = torch.randn(2, 100)
    y = torch.randint(0, 100, (2,))
    loss, comps = FitNetLoss(beta=1.0)(feat.clone(), feat.clone(), logits, y)
    assert comps["L_hint"].abs().item() < 1e-6


def test_fitnet_adapter_shape() -> None:
    a = FitNetAdapter(in_channels=96, out_channels=1024)
    x = torch.randn(2, 96, 14, 14)
    y = a(x)
    assert y.shape == (2, 1024, 14, 14)


def test_fitnet_gradient_does_not_flow_to_teacher() -> None:
    torch.manual_seed(4)
    s = torch.randn(2, 1024, 14, 14, requires_grad=True)
    t = torch.randn(2, 1024, 14, 14, requires_grad=True)
    logits = torch.randn(2, 100, requires_grad=True)
    y = torch.randint(0, 100, (2,))
    loss, _ = FitNetLoss(beta=1.0)(s, t, logits, y)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum().item() > 0
    assert logits.grad is not None
    assert t.grad is None or t.grad.abs().sum().item() == 0


def test_mid_feature_shape_match_at_224() -> None:
    """NEW_BENCH §"Layer matching choice": both hooks 14x14 @ 224x224 input."""
    student = timm.create_model(
        "mobilenetv2_100", pretrained=False, num_classes=0, global_pool=""
    ).eval()
    teacher = timm.create_model("resnet50", pretrained=False, num_classes=0).eval()
    s_hook = MidFeatureGrabber(get_mobilenetv2_mid_module(student))
    t_hook = MidFeatureGrabber(get_resnet50_mid_module(teacher))
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        _ = student(x)
        _ = teacher(x)
    s_feat, t_feat = s_hook.feat, t_hook.feat
    assert s_feat is not None and t_feat is not None
    print(f"\n  student blocks[4]: {tuple(s_feat.shape)}")
    print(f"  teacher layer3:    {tuple(t_feat.shape)}")
    assert s_feat.shape[-2:] == (14, 14), s_feat.shape
    assert t_feat.shape[-2:] == (14, 14), t_feat.shape
    assert s_feat.shape[1] == 96, s_feat.shape
    assert t_feat.shape[1] == 1024, t_feat.shape
    s_hook.close()
    t_hook.close()


if __name__ == "__main__":
    import subprocess

    raise SystemExit(
        subprocess.call(["python", "-m", "pytest", __file__, "-v", "-s"])
    )
