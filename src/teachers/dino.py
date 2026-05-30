"""DINO ViT-S/16 teacher loader.

Loads the official DINO ViT-S/16 self-supervised checkpoint from torch hub
(`facebookresearch/dino:main`, entry `dino_vits16`). The official hub model
returns the CLS-token embedding (dim 384) from its forward; we expose this
under a uniform `forward_features` API and freeze all parameters.

Reference: Caron et al., "Emerging Properties in Self-Supervised Vision
Transformers" (DINO), ICCV 2021. https://github.com/facebookresearch/dino
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# DINO / ImageNet normalization (the teacher was trained on ImageNet-1k with
# the standard ImageNet mean/std; inputs MUST be normalized this way at 224x224).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINO_VITS16_FEATURE_DIM = 384  # CLS-token embedding dim
DINO_VITS16_INPUT_SIZE = 224   # patch 16 -> 14x14 tokens at 224x224


class DinoTeacher(nn.Module):
    """Wraps the DINO ViT-S/16 hub model with a uniform `forward_features` API.

    The wrapped backbone returns the CLS-token embedding directly from its
    forward pass. We freeze all parameters and force eval mode so BN/Dropout
    are inert and gradients are blocked end-to-end.
    """

    feature_dim: int = DINO_VITS16_FEATURE_DIM
    input_size: int = DINO_VITS16_INPUT_SIZE

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DinoTeacher":  # type: ignore[override]
        # Keep the teacher permanently in eval mode regardless of parent .train() calls.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward_features(self, x: Tensor) -> Tensor:
        """Return CLS-token embedding of shape (B, 384) for input (B, 3, 224, 224)."""
        return self.backbone(x)

    @torch.no_grad()
    def forward_patch_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (cls_token, patch_tokens) for input (B, 3, 224, 224).

        cls_token:   (B, 384)
        patch_tokens: (B, 196, 384) — the 14x14 spatial patch embeddings
        """
        out = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, 197, 384)
        return out[:, 0], out[:, 1:]

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def load_dino_vits16(
    repo: str = "facebookresearch/dino:main",
    entry: str = "dino_vits16",
) -> DinoTeacher:
    """Load DINO ViT-S/16 from torch hub. Raises a clear error if unreachable."""
    try:
        backbone = torch.hub.load(repo, entry, trust_repo=True)
    except Exception as e:  # pragma: no cover - network-dependent
        raise RuntimeError(
            f"Failed to load DINO ViT-S/16 from torch hub ({repo}, entry={entry}). "
            "If you are offline, manually download the checkpoint from "
            "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/"
            "dino_deitsmall16_pretrain.pth and load it with the repo's vit_small() "
            "constructor at patch_size=16. Original error: " + repr(e)
        ) from e
    return DinoTeacher(backbone)
