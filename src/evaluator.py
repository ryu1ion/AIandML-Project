"""Evaluation primitives shared by the eval scripts.

Three protocols:
- linear_probe        : SGD lr=0.1 / momentum=0.9 / wd=0 / cosine over 100 epochs,
                        batch 256, on cached features (PRELIMINARY.md Eval 1).
- knn_classifier      : DINO-style cosine-similarity weighted kNN with k=20, T=0.07
                        (PRELIMINARY.md Eval 2).
- few_shot_logreg     : sklearn LogisticRegression on `n_shot` samples per class,
                        averaged over `n_seeds` seeds (PRELIMINARY.md Eval 3).

Plus utilities to load a saved student backbone and extract features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.students import MobileNetV2Student, get_student


def load_student(checkpoint: str | Path | None, device: torch.device) -> MobileNetV2Student:
    """Load a MobileNetV2 student from a saved checkpoint, or fresh random-init.

    `checkpoint` may be a path to a .pt file produced by `src.trainer`, or the
    string "random" / None for a fresh-init student (R1 baseline).
    """
    student = get_student("mobilenetv2").to(device).eval()
    if checkpoint is None or str(checkpoint).lower() == "random":
        return student
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = ckpt["student_state_dict"]
    student.load_state_dict(state)
    student.eval()
    return student


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `loader` and return (features, labels) on CPU."""
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"
        ):
            f = model(x)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.1,
    seed: int = 42,
) -> float:
    """SGD/cosine linear probe on cached features. Returns top-1 accuracy in [0, 1]."""
    torch.manual_seed(seed)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    classifier = nn.Linear(train_x.shape[1], num_classes).to(device)
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
        return float((logits.argmax(dim=1) == test_y).float().mean().item())


def knn_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 1024,
) -> float:
    """DINO-style weighted-vote kNN using cosine similarity.

    Top-k neighbors per test sample are weighted by exp(sim / T) and votes are
    summed per class. Returns top-1 accuracy in [0, 1]. Test set is processed
    in chunks to bound memory.
    """
    train_x = F.normalize(train_x.to(device), dim=-1, p=2)
    train_y = train_y.to(device)
    test_x = F.normalize(test_x.to(device), dim=-1, p=2)
    test_y = test_y.to(device)

    correct = 0
    total = 0
    for start in range(0, test_x.shape[0], chunk_size):
        chunk = test_x[start : start + chunk_size]
        sims = chunk @ train_x.T  # (chunk, Ntrain)
        top_sim, top_idx = sims.topk(k, dim=-1, largest=True, sorted=True)
        weights = torch.exp(top_sim / temperature)  # (chunk, k)
        top_labels = train_y[top_idx]  # (chunk, k)
        one_hot = F.one_hot(top_labels, num_classes=num_classes).float()  # (chunk, k, C)
        scores = (weights.unsqueeze(-1) * one_hot).sum(dim=1)  # (chunk, C)
        pred = scores.argmax(dim=-1)
        correct += int((pred == test_y[start : start + chunk_size]).sum().item())
        total += int(chunk.shape[0])
    return correct / total


def few_shot_logreg(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    n_shot: int = 5,
    n_seeds: int = 5,
    max_iter: int = 1000,
) -> tuple[float, float]:
    """Few-shot logistic regression: sample `n_shot` per class, fit, eval on test.

    Returns (mean_accuracy, std_accuracy) over `n_seeds` resampled support sets.
    """
    from sklearn.linear_model import LogisticRegression

    tr_x = train_x.numpy()
    tr_y = train_y.numpy()
    te_x = test_x.numpy()
    te_y = test_y.numpy()

    accs: list[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx_per_class = []
        for c in range(num_classes):
            cls_idx = np.where(tr_y == c)[0]
            chosen = rng.choice(cls_idx, size=n_shot, replace=False)
            idx_per_class.append(chosen)
        idx = np.concatenate(idx_per_class)
        x_sup = tr_x[idx]
        y_sup = tr_y[idx]
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(x_sup, y_sup)
        accs.append(float(clf.score(te_x, te_y)))
    return float(np.mean(accs)), float(np.std(accs))
