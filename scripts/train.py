"""Train MobileNetV2 ← DINO ViT-S/16 unlabeled distillation.

Three task variants:
  --task distill   base method (L2 on normalized features)
  --task ours      base + structural local distillation (and optional global /
                   cross-view auxiliaries)
  --task paperkd   base + SP-KD (Tung & Mori 2019) + RKD-distance (Park 2019)

Driven by a YAML config and/or CLI overrides. CLI overrides YAML.

Examples:
  python scripts/train.py --task distill --dataset cifar100 \\
      --epochs 100 --batch-size 256 --out-dir checkpoints/base
  python scripts/train.py --config configs/ours_distill_cifar100.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.trainer import TrainConfig, train  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--task",
        choices=["distill", "ours", "paperkd"],
        default=None,
    )
    # core training knobs
    p.add_argument("--student", type=str, default=None)
    p.add_argument("--teacher", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, help="cifar100 | imagenet100")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None)
    p.add_argument("--schedule", type=str, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--resume-from", type=str, default=None,
                   help="path to a final.pt to resume student/head/predictor weights from")
    p.add_argument("--lr-scale-rule", type=str, default=None, help="none | linear")
    p.add_argument("--sync-bn", type=int, default=None, help="0 or 1")
    p.add_argument("--two-view-aug", type=str, default=None, help="in100 | mild")
    p.add_argument("--bn-recalib-on-save", type=int, default=None, help="0 or 1")
    p.add_argument("--bn-recalib-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--bf16", type=int, default=None, help="0 or 1")
    p.add_argument("--proj-hidden", type=int, default=None)
    p.add_argument("--proj-out", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="debug/smoke only: cap iters/epoch (0=no limit)")
    p.add_argument("--wandb", type=int, default=None, help="0 or 1")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default=None, help="offline | online | disabled")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--student-pretrained", type=int, default=None, help="0 or 1")
    # "ours" method (L_base + lambda_local*L_local + lambda_global*L_global + lambda_cross*L_cross)
    p.add_argument("--use-local-structural-loss", type=int, default=None, help="0 or 1")
    p.add_argument("--use-global-semantic-loss", type=int, default=None, help="0 or 1")
    p.add_argument("--use-cross-view-invariant-loss", type=int, default=None, help="0 or 1")
    p.add_argument("--lambda-local", type=float, default=None)
    p.add_argument("--lambda-global", type=float, default=None)
    p.add_argument("--lambda-cross", type=float, default=None)
    p.add_argument("--lambda-view", type=float, default=None, help="alias for --lambda-cross")
    p.add_argument("--relation-temperature-teacher", type=float, default=None)
    p.add_argument("--relation-temperature-student", type=float, default=None)
    p.add_argument("--patch-relation-loss-type", type=str, default=None, help="mse | kl")
    p.add_argument("--global-relation-loss-type", type=str, default=None, help="mse | kl")
    p.add_argument("--cross-view-relation-loss-type", type=str, default=None, help="mse | kl")
    p.add_argument("--local-relation-mode", type=str, default=None, help="full | sample")
    p.add_argument("--local-max-tokens", type=int, default=None)
    p.add_argument("--global-mask-diagonal", type=int, default=None, help="0 or 1")
    p.add_argument("--ours-warmup-frac", type=float, default=None)
    p.add_argument("--ours-use-patch-proj", type=int, default=None, help="0 or 1")
    p.add_argument("--ours-patch-proj-dim", type=int, default=None)
    # legacy YAML-compat aliases for the older lambda_view / local_mode etc. fields
    p.add_argument("--local-temperature", type=float, default=None)
    p.add_argument("--global-temperature", type=float, default=None)
    p.add_argument("--view-temperature", type=float, default=None)
    p.add_argument("--local-mode", type=str, default=None)
    p.add_argument("--global-mode", type=str, default=None)
    p.add_argument("--view-mode", type=str, default=None)
    # "paperkd" method (SP-KD + RKD-distance)
    p.add_argument("--lambda-sp", type=float, default=None)
    p.add_argument("--lambda-rkd", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base: dict = {}
    if args.config is not None:
        with open(args.config) as f:
            base = yaml.safe_load(f) or {}

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    for bkey in ("bf16", "wandb", "sync_bn", "bn_recalib_on_save", "student_pretrained",
                  "global_mask_diagonal", "ours_use_patch_proj",
                  "use_local_structural_loss", "use_global_semantic_loss",
                  "use_cross_view_invariant_loss"):
        if bkey in overrides:
            overrides[bkey] = bool(overrides[bkey])
    base.update(overrides)

    if "task" not in base:
        raise SystemExit("--task is required (provide via --config or --task)")

    cfg = TrainConfig(**base)
    if int(os.environ.get("RANK", "0")) == 0:
        print("==== Train config ====")
        print(json.dumps(asdict(cfg), indent=2, default=str))
        print("======================", flush=True)

    result = train(cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        hist = result.get("history") or []
        if hist:
            print(f"\nFinal epoch loss: {hist[-1]['loss']:.4f}")
        print(f"Output dir: {result['out_dir']}")


if __name__ == "__main__":
    main()
