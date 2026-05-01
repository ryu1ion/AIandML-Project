"""Produce the two preliminary figures.

fig1_loss_curves.png : R3 (SimSiam, neg-cosine) and R4 (distill, L2-on-norm)
                       training loss vs epoch on the same axes (twin y-axes
                       since the two losses live on different scales).

fig2_tsne.png        : t-SNE of MobileNetV2 backbone features on a fixed
                       1000-sample stratified subset of the CIFAR-100 test
                       set, side-by-side for R1 (random init), R3 (SimSiam),
                       R4 (distill), colored by class.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.augmentations import make_eval_transform  # noqa: E402
from src.evaluator import extract_features, load_student  # noqa: E402

CKPT_DIR = REPO_ROOT / "checkpoints" / "preliminary"
RESULTS = REPO_ROOT / "results"
FIG_DIR = RESULTS / "figures"


def fig1_loss_curves() -> None:
    r3 = json.loads((CKPT_DIR / "r3_simsiam" / "history.json").read_text())
    r4 = json.loads((CKPT_DIR / "r4_distill" / "history.json").read_text())

    epochs_r3 = [e["epoch"] + 1 for e in r3]
    loss_r3 = [e["loss"] for e in r3]
    epochs_r4 = [e["epoch"] + 1 for e in r4]
    loss_r4 = [e["loss"] for e in r4]

    fig, ax_r3 = plt.subplots(figsize=(7.0, 4.2))
    color_r3 = "tab:orange"
    color_r4 = "tab:blue"

    ax_r3.set_xlabel("Epoch")
    ax_r3.set_ylabel("R3 SimSiam loss (−cos sim)", color=color_r3)
    line_r3, = ax_r3.plot(epochs_r3, loss_r3, color=color_r3, label="R3 SimSiam (left axis)")
    ax_r3.tick_params(axis="y", labelcolor=color_r3)

    ax_r4 = ax_r3.twinx()
    ax_r4.set_ylabel("R4 Distill loss (L2 on L2-norm features)", color=color_r4)
    line_r4, = ax_r4.plot(epochs_r4, loss_r4, color=color_r4, label="R4 Distill (right axis)")
    ax_r4.tick_params(axis="y", labelcolor=color_r4)

    ax_r3.set_title("Training loss vs epoch — R3 (SimSiam) and R4 (Label-free distillation)")
    fig.legend(handles=[line_r3, line_r4], loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    out = FIG_DIR / "fig1_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def _stratified_indices(labels: np.ndarray, total: int, num_classes: int, seed: int = 42) -> np.ndarray:
    """Pick `total` indices stratified by class. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    per_class = total // num_classes  # 10 for total=1000, num_classes=100
    extras = total - per_class * num_classes
    chosen: list[np.ndarray] = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        n_take = per_class + (1 if c < extras else 0)
        if cls_idx.size < n_take:
            chosen.append(cls_idx)
        else:
            chosen.append(rng.choice(cls_idx, size=n_take, replace=False))
    return np.concatenate(chosen)


def fig2_tsne(num_points: int = 1000, num_classes: int = 100, seed: int = 42) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_eval_transform(image_size=224)
    cifar_te = datasets.CIFAR100(str(REPO_ROOT / "data"), train=False, download=False, transform=transform)
    labels_all = np.array(cifar_te.targets)

    indices = _stratified_indices(labels_all, num_points, num_classes, seed=seed)
    subset = Subset(cifar_te, indices.tolist())
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    sub_labels = labels_all[indices]

    runs = [
        ("R1 Random init", "random"),
        ("R3 SSL from scratch", str(CKPT_DIR / "r3_simsiam" / "final.pt")),
        ("R4 Label-free distill", str(CKPT_DIR / "r4_distill" / "final.pt")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    cmap = plt.get_cmap("gist_ncar")
    colors = cmap(np.linspace(0, 1, num_classes))

    for ax, (title, ckpt) in zip(axes, runs):
        t0 = time.perf_counter()
        student = load_student(ckpt, device).eval()
        feats, _ = extract_features(student, loader, device)
        feats = feats.numpy()
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, max_iter=1000)
        emb = tsne.fit_transform(feats)
        print(f"{title}: features {feats.shape} -> TSNE in {time.perf_counter() - t0:.1f}s", flush=True)

        ax.scatter(emb[:, 0], emb[:, 1], c=colors[sub_labels], s=8, alpha=0.85, linewidth=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"t-SNE of MobileNetV2 features on {num_points} stratified CIFAR-100 test samples "
        f"(perplexity 30; colors = 100 classes)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig2_tsne.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_loss_curves()
    fig2_tsne()


if __name__ == "__main__":
    main()
