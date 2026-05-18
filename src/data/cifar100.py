"""CIFAR-100 dataset wrappers at 224x224.

`get_cifar100(split, mode)` returns a torchvision CIFAR-100 dataset with the
appropriate transform for the requested mode:
- 'supervised'  : single view, supervised-train augmentation
- 'two_view'    : DINO-style two-view augmentation (returns ((v1, v2), label))
- 'eval'        : deterministic Resize+CenterCrop+Normalize
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from torchvision import datasets

from src.data.augmentations import (
    TwoViewTransform,
    make_dino_view_transform,
    make_eval_transform,
    make_supervised_train_transform,
)

Split = Literal["train", "test"]
Mode = Literal["supervised", "two_view", "eval"]


def get_cifar100(
    data_root: str | Path,
    split: Split,
    mode: Mode,
    image_size: int = 224,
    download: bool = True,
) -> datasets.CIFAR100:
    """Return CIFAR-100 with a transform appropriate for `mode`."""
    is_train = split == "train"
    if mode == "supervised":
        transform = (
            make_supervised_train_transform(image_size)
            if is_train
            else make_eval_transform(image_size)
        )
    elif mode == "two_view":
        transform = TwoViewTransform(make_dino_view_transform(image_size))
    elif mode == "eval":
        transform = make_eval_transform(image_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return datasets.CIFAR100(
        str(data_root), train=is_train, download=download, transform=transform
    )
