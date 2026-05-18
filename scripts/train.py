"""Train one of the four preliminary runs (supervised / simsiam / distill).

Driven by a YAML config and/or CLI overrides. CLI overrides YAML.

Examples:
  python scripts/train.py --task distill --epochs 3 --out-dir checkpoints/preliminary/r4_smoke
  python scripts/train.py --config configs/preliminary/r4_distill.yaml
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--task", choices=["supervised", "simsiam", "distill"], default=None)
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base: dict = {}
    if args.config is not None:
        with open(args.config) as f:
            base = yaml.safe_load(f) or {}

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if "bf16" in overrides:
        overrides["bf16"] = bool(overrides["bf16"])
    if "wandb" in overrides:
        overrides["wandb"] = bool(overrides["wandb"])
    base.update(overrides)

    if "task" not in base:
        raise SystemExit("--task is required (provide via --config or --task)")

    cfg = TrainConfig(**base)
    print("==== Train config ====")
    print(json.dumps(asdict(cfg), indent=2, default=str))
    print("======================", flush=True)

    result = train(cfg)
    final_loss = result["history"][-1]["loss"]
    print(f"\nFinal epoch loss: {final_loss:.4f}")
    print(f"Output dir: {result['out_dir']}")


if __name__ == "__main__":
    main()
