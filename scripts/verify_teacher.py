"""Verify DINO ViT-S/16 teacher: extract CIFAR-100 features, run linear probe,
write results/teacher_sanity.txt.

This is a sanity check, not the formal Eval 1 of the project. The probe uses
the same recipe (SGD lr=0.1, cosine, 100 epochs, batch=256, frozen features)
to keep the number comparable to what the project will report later.

Expected DINO ViT-S/16 CIFAR-100 linear probe accuracy: ~80-84%.
Acceptance window from PRELIMINARY.md Step 1: 78%-90%.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.teachers.dino import (  # noqa: E402
    DINO_VITS16_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dino_vits16,
)

DATA_ROOT = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def make_eval_transform() -> transforms.Compose:
    """Standard ImageNet-style eval pipeline at 224x224 with ImageNet stats.

    CIFAR-100 native resolution is 32x32; we resize directly to 224 (no center
    crop margin) since the original image is square and small.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                DINO_VITS16_INPUT_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(DINO_VITS16_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Extract features for an entire loader. Time only full-sized batches."""
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    full_batch_times_s: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        is_full = x.shape[0] == target_batch_size
        if is_full and device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model(x)
        if is_full and device.type == "cuda":
            torch.cuda.synchronize()
            full_batch_times_s.append(time.perf_counter() - t0)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels), full_batch_times_s


def linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    num_classes: int = 100,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.1,
    seed: int = 42,
) -> float:
    """SGD/cosine linear probe on cached features (the project's standard recipe)."""
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    feat_dim = train_x.shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optim = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    n = train_x.shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    classifier.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = classifier(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        sched.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_x)
        acc = (logits.argmax(dim=1) == test_y).float().mean().item()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["cifar100", "imagenet100"], default="cifar100"
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--out",
        default=None,
        help="output path; defaults to results/teacher_sanity[_<dataset>].txt",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        suffix = "" if args.dataset == "cifar100" else f"_{args.dataset}"
        args.out = str(RESULTS_DIR / f"teacher_sanity{suffix}.txt")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    )
    print(f"Device: {device} ({device_name})")

    print("Loading DINO ViT-S/16 from torch hub...")
    teacher = load_dino_vits16().to(device).eval()
    print(f"  feature_dim={teacher.feature_dim}, input_size={teacher.input_size}")

    if args.dataset == "cifar100":
        transform = make_eval_transform()
        resize_desc = (
            f"Resize({DINO_VITS16_INPUT_SIZE}, bicubic) -> "
            f"CenterCrop({DINO_VITS16_INPUT_SIZE})"
        )
        ds_desc = "CIFAR-100 (50k train / 10k test)"
        print(f"Loading CIFAR-100 from {args.data_root}...")
        Path(args.data_root).mkdir(parents=True, exist_ok=True)
        train_ds = datasets.CIFAR100(
            args.data_root, train=True, download=True, transform=transform
        )
        test_ds = datasets.CIFAR100(
            args.data_root, train=False, download=True, transform=transform
        )
    else:  # imagenet100
        from src.data.imagenet100 import get_imagenet100, verify_class_list

        resize_desc = (
            f"Resize(256, bicubic) -> CenterCrop({DINO_VITS16_INPUT_SIZE})"
        )
        ds_desc = "ImageNet-100 (CMC/MoCo subset; 126,689 train / 5,000 val)"
        print(f"Loading ImageNet-100 from {args.data_root}...")
        verify_class_list(args.data_root)
        train_ds = get_imagenet100(args.data_root, split="train", mode="eval")
        test_ds = get_imagenet100(args.data_root, split="validation", mode="eval")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    print("Extracting train features...")
    t0 = time.perf_counter()
    train_x, train_y, train_times = extract_features(
        teacher, train_loader, device, args.batch_size
    )
    print(f"  shape={tuple(train_x.shape)} in {time.perf_counter() - t0:.1f}s")

    print("Extracting test features...")
    t0 = time.perf_counter()
    test_x, test_y, test_times = extract_features(
        teacher, test_loader, device, args.batch_size
    )
    print(f"  shape={tuple(test_x.shape)} in {time.perf_counter() - t0:.1f}s")

    all_times = train_times + test_times
    if all_times:
        all_sorted = sorted(all_times)
        median_batch_time_s = all_sorted[len(all_sorted) // 2]
    else:
        median_batch_time_s = float("nan")

    feat_dim = train_x.shape[1]
    print(
        f"Running linear probe (SGD lr=0.1 momentum=0.9 wd=0, "
        f"cosine, {args.epochs} epochs, batch={args.batch_size})..."
    )
    t0 = time.perf_counter()
    acc = linear_probe(
        train_x,
        train_y,
        test_x,
        test_y,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  Linear probe top-1: {acc * 100:.2f}%  ({time.perf_counter() - t0:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# DINO ViT-S/16 sanity check on {ds_desc}",
        f"teacher                                        : facebookresearch/dino:main / dino_vits16",
        f"dataset                                        : {args.dataset} ({ds_desc})",
        f"feature_dim                                    : {feat_dim}",
        f"input_size                                     : {teacher.input_size}",
        f"normalization                                  : ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD}",
        f"resize/crop                                    : {resize_desc}",
        f"linear_probe_protocol                          : SGD lr=0.1 momentum=0.9 wd=0, cosine over {args.epochs} epochs, batch={args.batch_size}, frozen features",
        f"linear_probe_top1_pct                          : {acc * 100:.2f}",
        f"median_inference_time_per_batch_of_{args.batch_size}_seconds : {median_batch_time_s:.4f}",
        f"num_full_batches_timed                         : {len(all_times)}",
        f"device                                         : {device_name}",
        f"bf16_autocast                                  : True",
        f"seed                                           : {args.seed}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
