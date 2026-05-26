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

from src.data.augmentations import (  # noqa: E402
    make_eval_transform,
    make_in100_eval_transform,
)
from src.evaluator import (  # noqa: E402
    extract_features,
    few_shot_logreg,
    knn_classifier,
    linear_probe,
    load_student,
    recalibrate_bn,
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
    p.add_argument(
        "--dataset",
        choices=["cifar100", "imagenet100"],
        default="cifar100",
        help="dataset for the linear-probe + kNN metrics (5-shot is always STL-10)",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--few-shot-n", type=int, default=5)
    p.add_argument("--few-shot-seeds", type=int, default=5)
    p.add_argument(
        "--bn-recalib", type=int, default=1,
        help="1=re-estimate BN running stats on probe-train data before "
             "feature extraction (fixes DDP/bf16 BN corruption). Applied "
             "uniformly to every checkpoint incl. random for a fair table.",
    )
    p.add_argument("--bn-recalib-batches", type=int, default=200)
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

    stl_transform = make_eval_transform(image_size=224)
    if args.dataset == "cifar100":
        probe_transform = make_eval_transform(image_size=224)
        probe_tr = datasets.CIFAR100(
            args.data_root, train=True, download=True, transform=probe_transform
        )
        probe_te = datasets.CIFAR100(
            args.data_root, train=False, download=True, transform=probe_transform
        )
        probe_name = "CIFAR-100"
        probe_resize = "Resize(224)->CenterCrop(224)"
    else:  # imagenet100
        from src.data.imagenet100 import get_imagenet100

        probe_tr = get_imagenet100(args.data_root, split="train", mode="eval")
        probe_te = get_imagenet100(args.data_root, split="validation", mode="eval")
        probe_name = "ImageNet-100"
        probe_resize = "Resize(256)->CenterCrop(224)"
    stl_tr = datasets.STL10(args.data_root, split="train", download=True, transform=stl_transform)
    stl_te = datasets.STL10(args.data_root, split="test", download=True, transform=stl_transform)

    timings = {}
    if args.bn_recalib:
        print(
            f"BN recalibration on {probe_name} train "
            f"({args.bn_recalib_batches} batches)...",
            flush=True,
        )
        t0 = time.perf_counter()
        recalib_loader = DataLoader(
            probe_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        recalibrate_bn(
            student, recalib_loader, device, n_batches=args.bn_recalib_batches
        )
        timings["bn_recalib_s"] = time.perf_counter() - t0
        print(f"  done in {timings['bn_recalib_s']:.1f}s", flush=True)

    print(f"Extracting features (probe dataset: {probe_name})...", flush=True)
    t0 = time.perf_counter()
    probe_tr_x, probe_tr_y = extract_features(student, _make_loader(probe_tr, args.batch_size, args.num_workers), device)
    timings["extract_probe_train_s"] = time.perf_counter() - t0
    print(f"  {probe_name} train: {tuple(probe_tr_x.shape)} in {timings['extract_probe_train_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    probe_te_x, probe_te_y = extract_features(student, _make_loader(probe_te, args.batch_size, args.num_workers), device)
    timings["extract_probe_test_s"] = time.perf_counter() - t0
    print(f"  {probe_name} test:  {tuple(probe_te_x.shape)} in {timings['extract_probe_test_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stl_tr_x, stl_tr_y = extract_features(student, _make_loader(stl_tr, args.batch_size, args.num_workers), device)
    timings["extract_stl_train_s"] = time.perf_counter() - t0
    print(f"  STL-10 train:    {tuple(stl_tr_x.shape)} in {timings['extract_stl_train_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stl_te_x, stl_te_y = extract_features(student, _make_loader(stl_te, args.batch_size, args.num_workers), device)
    timings["extract_stl_test_s"] = time.perf_counter() - t0
    print(f"  STL-10 test:     {tuple(stl_te_x.shape)} in {timings['extract_stl_test_s']:.1f}s", flush=True)

    print(f"Linear probe ({probe_name}, SGD lr=0.1 cosine 100 epochs bs=256)...", flush=True)
    t0 = time.perf_counter()
    lp_acc = linear_probe(
        probe_tr_x, probe_tr_y, probe_te_x, probe_te_y,
        num_classes=100, device=device, seed=args.seed,
    )
    timings["linear_probe_s"] = time.perf_counter() - t0
    print(f"  top-1: {lp_acc * 100:.2f}%  ({timings['linear_probe_s']:.1f}s)", flush=True)

    print(f"kNN ({probe_name}, k={args.knn_k}, T={args.knn_temperature})...", flush=True)
    t0 = time.perf_counter()
    knn_acc = knn_classifier(
        probe_tr_x, probe_tr_y, probe_te_x, probe_te_y,
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
        "probe_dataset": args.dataset,
        "linear_probe_top1_pct": round(lp_acc * 100, 2),
        "knn_top1_pct": round(knn_acc * 100, 2),
        # Dataset-specific aliases (keeps make_preliminary_table.py working).
        f"linear_probe_{args.dataset}_top1_pct": round(lp_acc * 100, 2),
        f"knn_{args.dataset}_top1_pct": round(knn_acc * 100, 2),
        "stl10_5shot_mean_pct": round(fs_mean * 100, 2),
        "stl10_5shot_std_pct": round(fs_std * 100, 2),
        "config": {
            "probe_dataset": args.dataset,
            "linear_probe": "SGD lr=0.1 momentum=0.9 wd=0, cosine 100 epochs, bs=256",
            "knn": f"k={args.knn_k}, cosine, weighted exp(sim/{args.knn_temperature})",
            "few_shot": f"n_shot={args.few_shot_n}, sklearn LR, {args.few_shot_seeds} seeds, STL-10",
            "image_size": 224,
            "probe_resize": probe_resize,
            "normalization": "ImageNet",
            "bn_recalib": bool(args.bn_recalib),
            "bn_recalib_batches": args.bn_recalib_batches if args.bn_recalib else 0,
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
