"""Evaluate the trained classifier head saved with an R5/R6 checkpoint.

R5 (Hinton) and R6 (FitNet) train a MobileNetV2 backbone + linear classifier
end-to-end. This script restores both and reports top-1 on the test set, with
BN running stats recalibrated on the train split (matches the protocol used by
eval_checkpoint.py for the frozen-backbone linear probe).

The same checkpoint should also be run through `eval_checkpoint.py` for the
frozen-backbone linear probe / kNN / 5-shot STL-10 metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.augmentations import make_eval_transform  # noqa: E402
from src.evaluator import recalibrate_bn  # noqa: E402
from src.students import get_student  # noqa: E402


def _make_loader(ds, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=False, persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _eval_acc(student, head, loader, device) -> float:
    student.eval(); head.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = head(student(x))
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(1, total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=["cifar100", "imagenet100"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bn-recalib", type=int, default=1)
    p.add_argument("--bn-recalib-batches", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {args.run_name} on {device}", flush=True)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    student = get_student("mobilenetv2").to(device)
    student.load_state_dict(ck["student_state_dict"])

    head_sd = ck["head_state_dict"]
    in_dim = head_sd["fc.weight"].shape[1]; num_cls = head_sd["fc.weight"].shape[0]
    head = nn.Sequential()
    head = nn.Linear(in_dim, num_cls).to(device)
    head.weight.data.copy_(head_sd["fc.weight"]); head.bias.data.copy_(head_sd["fc.bias"])

    tf = make_eval_transform(image_size=224)
    if args.dataset == "cifar100":
        tr = datasets.CIFAR100(args.data_root, train=True, download=True, transform=tf)
        te = datasets.CIFAR100(args.data_root, train=False, download=True, transform=tf)
    else:
        from src.data.imagenet100 import get_imagenet100
        tr = get_imagenet100(args.data_root, split="train", mode="eval")
        te = get_imagenet100(args.data_root, split="validation", mode="eval")

    timings = {}
    if args.bn_recalib:
        print(f"BN recalibration ({args.bn_recalib_batches} batches)...", flush=True)
        t0 = time.perf_counter()
        recalibrate_bn(student, _make_loader(tr, args.batch_size, args.num_workers, shuffle=True),
                       device, n_batches=args.bn_recalib_batches)
        timings["bn_recalib_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = _eval_acc(student, head, _make_loader(te, args.batch_size, args.num_workers, shuffle=False), device)
    timings["eval_s"] = time.perf_counter() - t0
    print(f"  classifier top-1: {acc * 100:.2f}%", flush=True)

    out = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "classifier_top1_pct": round(acc * 100, 2),
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "bn_recalib": bool(args.bn_recalib),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
