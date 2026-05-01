"""Run linear probe (CIFAR-100), kNN (CIFAR-100), and 5-shot transfer (STL-10)
on a single MobileNetV2 checkpoint.

Writes a JSON file with the four metrics plus metadata. Designed to be run in
parallel — one invocation per checkpoint, one GPU each.

Usage:
  python scripts/eval_checkpoint.py --run-name r4_distill \\
      --checkpoint checkpoints/preliminary/r4_distill/final.pt \\
      --output results/eval_r4_distill.json

For the random-init baseline:
  python scripts/eval_checkpoint.py --run-name r1_random_init --checkpoint random \\
      --output results/eval_r1_random_init.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.augmentations import make_eval_transform  # noqa: E402
from src.evaluator import (  # noqa: E402
    extract_features,
    few_shot_logreg,
    knn_classifier,
    linear_probe,
    load_student,
)


def _make_loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", required=True, help='Path to .pt or "random"')
    p.add_argument("--output", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--few-shot-n", type=int, default=5)
    p.add_argument("--few-shot-seeds", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Run {args.run_name} on {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        flush=True,
    )
    print(f"Checkpoint: {args.checkpoint}", flush=True)

    student = load_student(args.checkpoint, device)
    feature_dim = int(student.feature_dim)
    print(f"Student feature_dim: {feature_dim}", flush=True)

    transform = make_eval_transform(image_size=224)
    cifar_tr = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    cifar_te = datasets.CIFAR100(args.data_root, train=False, download=True, transform=transform)
    stl_tr = datasets.STL10(args.data_root, split="train", download=True, transform=transform)
    stl_te = datasets.STL10(args.data_root, split="test", download=True, transform=transform)

    print("Extracting features...", flush=True)
    timings = {}
    t0 = time.perf_counter()
    cifar_tr_x, cifar_tr_y = extract_features(student, _make_loader(cifar_tr, args.batch_size, args.num_workers), device)
    timings["extract_cifar_train_s"] = time.perf_counter() - t0
    print(f"  CIFAR-100 train: {tuple(cifar_tr_x.shape)} in {timings['extract_cifar_train_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    cifar_te_x, cifar_te_y = extract_features(student, _make_loader(cifar_te, args.batch_size, args.num_workers), device)
    timings["extract_cifar_test_s"] = time.perf_counter() - t0
    print(f"  CIFAR-100 test:  {tuple(cifar_te_x.shape)} in {timings['extract_cifar_test_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stl_tr_x, stl_tr_y = extract_features(student, _make_loader(stl_tr, args.batch_size, args.num_workers), device)
    timings["extract_stl_train_s"] = time.perf_counter() - t0
    print(f"  STL-10 train:    {tuple(stl_tr_x.shape)} in {timings['extract_stl_train_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stl_te_x, stl_te_y = extract_features(student, _make_loader(stl_te, args.batch_size, args.num_workers), device)
    timings["extract_stl_test_s"] = time.perf_counter() - t0
    print(f"  STL-10 test:     {tuple(stl_te_x.shape)} in {timings['extract_stl_test_s']:.1f}s", flush=True)

    print("Linear probe (CIFAR-100, SGD lr=0.1 cosine 100 epochs bs=256)...", flush=True)
    t0 = time.perf_counter()
    lp_acc = linear_probe(
        cifar_tr_x, cifar_tr_y, cifar_te_x, cifar_te_y,
        num_classes=100, device=device, seed=args.seed,
    )
    timings["linear_probe_s"] = time.perf_counter() - t0
    print(f"  top-1: {lp_acc * 100:.2f}%  ({timings['linear_probe_s']:.1f}s)", flush=True)

    print(f"kNN (CIFAR-100, k={args.knn_k}, T={args.knn_temperature})...", flush=True)
    t0 = time.perf_counter()
    knn_acc = knn_classifier(
        cifar_tr_x, cifar_tr_y, cifar_te_x, cifar_te_y,
        num_classes=100, device=device,
        k=args.knn_k, temperature=args.knn_temperature,
    )
    timings["knn_s"] = time.perf_counter() - t0
    print(f"  top-1: {knn_acc * 100:.2f}%  ({timings['knn_s']:.1f}s)", flush=True)

    print(f"{args.few_shot_n}-shot STL-10 (LogReg, {args.few_shot_seeds} seeds)...", flush=True)
    t0 = time.perf_counter()
    fs_mean, fs_std = few_shot_logreg(
        stl_tr_x, stl_tr_y, stl_te_x, stl_te_y,
        num_classes=10, n_shot=args.few_shot_n, n_seeds=args.few_shot_seeds,
    )
    timings["few_shot_s"] = time.perf_counter() - t0
    print(f"  {fs_mean * 100:.2f}% ± {fs_std * 100:.2f}%  ({timings['few_shot_s']:.1f}s)", flush=True)

    out = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "feature_dim": feature_dim,
        "linear_probe_cifar100_top1_pct": round(lp_acc * 100, 2),
        "knn_cifar100_top1_pct": round(knn_acc * 100, 2),
        "stl10_5shot_mean_pct": round(fs_mean * 100, 2),
        "stl10_5shot_std_pct": round(fs_std * 100, 2),
        "config": {
            "linear_probe": "SGD lr=0.1 momentum=0.9 wd=0, cosine 100 epochs, bs=256",
            "knn": f"k={args.knn_k}, cosine, weighted exp(sim/{args.knn_temperature})",
            "few_shot": f"n_shot={args.few_shot_n}, sklearn LR, {args.few_shot_seeds} seeds",
            "image_size": 224,
            "normalization": "ImageNet",
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
