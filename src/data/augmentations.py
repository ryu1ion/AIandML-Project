"""Augmentation pipelines for CIFAR-100 at 224x224 with ImageNet normalization.

Three modes are supported:
- supervised train : Resize(224) + RandomCrop(padding=4) + HFlip + Normalize
                     (literal reading of "standard CIFAR-100 augmentation" from
                     PRELIMINARY.md, applied at the upsampled 224 resolution)
- two-view SSL    : DINO-style global-view augmentation (RRC scale (0.4, 1.0),
                     HFlip, ColorJitter, RandomGrayscale, GaussianBlur, Normalize)
                     applied twice independently
- eval            : Resize(224) + CenterCrop(224) + Normalize
"""
from __future__ import annotations

from typing import Callable

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoViewTransform:
    """Apply the same transform pipeline twice to produce (v1, v2) views.

    Each call generates two stochastic samples from the same pipeline, so v1 != v2.
    """

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


def make_dino_view_transform(image_size: int = 224) -> transforms.Compose:
    """DINO-style global-view augmentation at `image_size`.

    Matches the DINO global-crop pipeline (RRC scale (0.4, 1.0), p=0.8 ColorJitter,
    p=0.2 RandomGrayscale, p=0.5 GaussianBlur). Wrap in TwoViewTransform to get
    two independent views per sample.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.4, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_supervised_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard CIFAR-100 augmentation, applied at the upsampled `image_size`."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic eval pipeline: Resize -> CenterCrop -> Normalize."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# ImageNet-100 builders (PHASE2.md §1 — native 224, no upsampling).
#
# These differ from the CIFAR-100 (preliminary) builders above and are kept
# separate so the preliminary phase stays bit-for-bit reproducible:
#   - two-view RRC scale (0.2, 1.0) and *asymmetric* DINO blur (p=0.5 / p=0.1)
#   - supervised RRC scale (0.08, 1.0) (standard ImageNet, not CIFAR pad-crop)
#   - eval Resize(256) -> CenterCrop(224) (standard ImageNet, not Resize(224))
# ---------------------------------------------------------------------------


class AsymmetricTwoViewTransform:
    """Apply two *different* pipelines to produce (v1, v2).

    Used for DINO-style asymmetric Gaussian blur: view 1 gets blur p=0.5,
    view 2 gets blur p=0.1 (Caron et al., 2021, §3 / appendix).
    """

    def __init__(self, transform1: Callable, transform2: Callable) -> None:
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, img):
        return self.transform1(img), self.transform2(img)


def make_in100_view_transform(
    image_size: int = 224, blur_p: float = 0.5
) -> transforms.Compose:
    """One DINO-style global view for IN-100 with configurable blur probability.

    RandomResizedCrop(scale=(0.2, 1.0), bicubic), HFlip(0.5),
    ColorJitter(0.4, 0.4, 0.2, 0.1) @ p=0.8, RandomGrayscale(0.2),
    GaussianBlur(kernel=23, sigma~U(0.1, 2.0)) @ p=`blur_p`, Normalize.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=blur_p,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_in100_two_view_transform(image_size: int = 224) -> AsymmetricTwoViewTransform:
    """DINO-style two-view augmentation with asymmetric blur (p=0.5 / p=0.1)."""
    return AsymmetricTwoViewTransform(
        make_in100_view_transform(image_size, blur_p=0.5),
        make_in100_view_transform(image_size, blur_p=0.1),
    )


def make_in100_supervised_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet supervised-train augmentation: RRC(0.08, 1.0) + HFlip."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_in100_eval_transform(
    image_size: int = 224, resize: int = 256
) -> transforms.Compose:
    """Standard ImageNet eval pipeline: Resize(256) -> CenterCrop(224) -> Normalize."""
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
