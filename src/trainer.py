"""Training loop for the preliminary phase.

Supports three tasks via cfg.task:
- 'supervised' : MobileNetV2 + linear classifier, CE loss, single (CIFAR-style) view.
- 'simsiam'    : MobileNetV2 + projector + predictor, SimSiam loss, two views.
- 'distill'    : MobileNetV2 + projection head, L2-on-normalized to a frozen
                 DINO ViT-S/16 teacher; both views go through teacher and student,
                 loss averaged over views.

Single GPU, plain PyTorch, bf16 autocast on by default.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.cifar100 import get_cifar100
from src.losses.feature_matching import l2_normalized_mse_loss
from src.utils.wandb_logging import WandbLogger
from src.losses.simsiam import SimSiamPredictor, SimSiamProjector, simsiam_loss
from src.projection_heads import MLPProjectionHead
from src.students import get_student
from src.teachers import get_teacher

Task = Literal["supervised", "simsiam", "distill"]


@dataclass
class TrainConfig:
    task: Task
    student: str = "mobilenetv2"
    teacher: str = "dino_vits16"  # only used when task='distill'
    dataset: str = "cifar100"  # 'cifar100' (preliminary) or 'imagenet100' (phase2)
    data_root: str = "data"
    image_size: int = 224
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adamw' or 'sgd'
    schedule: str = "cosine"  # 'cosine' or 'constant'
    warmup_epochs: int = 0
    num_workers: int = 8
    seed: int = 42
    bf16: bool = True
    num_classes: int = 100  # CIFAR-100 and ImageNet-100 both have 100 classes
    proj_hidden: int = 768  # 2 * teacher_dim (DINO ViT-S/16 dim = 384)
    proj_out: int = 384     # teacher feature dim
    out_dir: str = "checkpoints/preliminary/run"
    log_every: int = 50
    # Debug/smoke knob only (not a hyperparameter): cap iterations per epoch.
    # 0 = no limit (the value used for all reported runs).
    limit_train_batches: int = 0
    # W&B (PHASE2 hard-constraint #3). Default off so the preliminary phase
    # stays reproducible; phase2 configs/CLI set wandb=True.
    wandb: bool = False
    wandb_project: str = "label-free-distill-phase2"
    wandb_mode: str = "offline"  # no credentials in this env -> offline
    wandb_run_name: str | None = None
    wandb_entity: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SupervisedHead(nn.Module):
    """Linear classifier on top of pooled backbone features."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, f):
        return self.fc(f)


def _build_modules(cfg: TrainConfig, device: torch.device) -> dict[str, nn.Module | None]:
    student = get_student(cfg.student).to(device)

    if cfg.task == "supervised":
        head = _SupervisedHead(student.feature_dim, cfg.num_classes).to(device)
        return {"student": student, "head": head, "teacher": None, "predictor": None}

    if cfg.task == "simsiam":
        projector = SimSiamProjector(student.feature_dim).to(device)
        predictor = SimSiamPredictor(in_dim=projector.out_dim).to(device)
        return {
            "student": student,
            "head": projector,
            "teacher": None,
            "predictor": predictor,
        }

    if cfg.task == "distill":
        head = MLPProjectionHead(
            student.feature_dim, out_dim=cfg.proj_out, hidden_dim=cfg.proj_hidden
        ).to(device)
        teacher = get_teacher(cfg.teacher).to(device).eval()
        return {"student": student, "head": head, "teacher": teacher, "predictor": None}

    raise ValueError(f"Unknown task: {cfg.task}")


def _get_train_dataset(cfg: TrainConfig, mode: str):
    """Return the *train*-split dataset for `cfg.dataset` in the given mode.

    Both CIFAR-100 and ImageNet-100 expose the same item contract:
    (img, label) for supervised/eval, ((v1, v2), label) for two_view.
    """
    name = cfg.dataset.lower()
    if name in {"cifar100", "cifar-100"}:
        return get_cifar100(
            cfg.data_root, split="train", mode=mode, image_size=cfg.image_size
        )
    if name in {"imagenet100", "imagenet-100", "in100", "in-100"}:
        from src.data.imagenet100 import get_imagenet100  # lazy: pulls `datasets`

        return get_imagenet100(
            cfg.data_root, split="train", mode=mode, image_size=cfg.image_size
        )
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _make_loader(cfg: TrainConfig) -> DataLoader:
    mode = "supervised" if cfg.task == "supervised" else "two_view"
    ds = _get_train_dataset(cfg, mode)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _build_schedule(
    optim: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    total = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if cfg.schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def _make_eval_batch(cfg: TrainConfig, device: torch.device, n: int = 256) -> torch.Tensor:
    """Fixed deterministic batch for representation diagnostics.

    Uses CIFAR-100 train, the eval (no-augmentation) transform, and the first
    `n` samples in dataset order. The same images are used every epoch so that
    `feature_std` deltas reflect the model, not the inputs.
    """
    ds = _get_train_dataset(cfg, mode="eval")
    n = min(n, len(ds))
    images = torch.stack([ds[i][0] for i in range(n)]).to(device)
    return images


@torch.no_grad()
def _compute_feature_std(
    student: nn.Module,
    eval_images: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Diversity diagnostic: per-feature std of L2-normalized backbone output,
    averaged over feature dims, then scaled by sqrt(d).

    With this scaling, fully-diverse uniform features on the d-sphere give
    a value of 1.0 and fully collapsed features give 0.0, so a single absolute
    threshold (e.g., 0.1) is meaningful regardless of d.
    """
    was_training = student.training
    student.eval()
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            f = student(eval_images)
        f = f.float()
        d = f.shape[-1]
        f_norm = F.normalize(f, dim=-1, p=2)
        return float(f_norm.std(dim=0).mean().item() * math.sqrt(d))
    finally:
        if was_training:
            student.train()


def _save_checkpoint(
    path: Path,
    *,
    cfg: TrainConfig,
    student: nn.Module,
    head: nn.Module,
    predictor: nn.Module | None,
    feature_dim: int | None,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "task": cfg.task,
            "student_state_dict": student.state_dict(),
            "head_state_dict": head.state_dict(),
            "predictor_state_dict": predictor.state_dict() if predictor is not None else None,
            "config": asdict(cfg),
            "feature_dim": feature_dim,
            "history": history,
        },
        path,
    )


def train(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss", "lr"])

    loader = _make_loader(cfg)
    steps_per_epoch = len(loader)

    modules = _build_modules(cfg, device)
    student: nn.Module = modules["student"]  # type: ignore[assignment]
    head: nn.Module = modules["head"]  # type: ignore[assignment]
    teacher = modules["teacher"]
    predictor = modules["predictor"]
    feature_dim = getattr(student, "feature_dim", None)

    eval_images = _make_eval_batch(cfg, device)

    trainable: list[torch.nn.Parameter] = list(student.parameters()) + list(head.parameters())
    if predictor is not None:
        trainable += list(predictor.parameters())

    optimizer = _build_optimizer(trainable, cfg)
    scheduler = _build_schedule(optimizer, cfg, steps_per_epoch)

    use_amp = cfg.bf16 and device.type == "cuda"

    # Pre-training feature_std (epoch 0 = random init or fresh build).
    init_fs = _compute_feature_std(student, eval_images, device, use_amp)
    print(f"[{cfg.task}] init feature_std={init_fs:.4f}  (d={feature_dim})", flush=True)

    wb = WandbLogger(
        enabled=cfg.wandb,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name or out_dir.name,
        config=asdict(cfg),
        mode=cfg.wandb_mode,
        entity=cfg.wandb_entity,
        out_dir=str(out_dir),
    )
    if wb.run_dir is not None:
        (out_dir / "wandb_run.txt").write_text(
            f"mode={cfg.wandb_mode}\nproject={cfg.wandb_project}\n"
            f"run_dir={wb.run_dir}\n"
        )
    wb.log({"feature_std": init_fs, "epoch": 0}, step=0)

    history: list[dict[str, float]] = []
    aborted = False
    abort_reason: str | None = None
    global_step = 0
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    try:
        for epoch in range(cfg.epochs):
            student.train()
            head.train()
            if predictor is not None:
                predictor.train()
            if teacher is not None:
                teacher.eval()

            ep_loss_sum = 0.0
            ep_loss_n = 0
            t0 = time.perf_counter()

            for batch_idx, batch in enumerate(loader):
                if cfg.limit_train_batches and batch_idx >= cfg.limit_train_batches:
                    break
                if cfg.task == "supervised":
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        logits = head(student(x))
                        loss = F.cross_entropy(logits, y)
                elif cfg.task == "simsiam":
                    (x1, x2), _ = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        z1 = head(student(x1))
                        z2 = head(student(x2))
                        p1 = predictor(z1)
                        p2 = predictor(z2)
                        loss = simsiam_loss(p1, p2, z1, z2)
                elif cfg.task == "distill":
                    (x1, x2), _ = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        s1 = head(student(x1))
                        s2 = head(student(x2))
                        with torch.no_grad():
                            t1 = teacher(x1)
                            t2 = teacher(x2)
                        loss = 0.5 * (
                            l2_normalized_mse_loss(s1, t1) + l2_normalized_mse_loss(s2, t2)
                        )
                else:
                    raise ValueError(cfg.task)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_v = float(loss.item())
                ep_loss_sum += loss_v
                ep_loss_n += 1
                global_step += 1

                if global_step % cfg.log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    log_writer.writerow([epoch, global_step, loss_v, cur_lr])
                    log_file.flush()
                    wb.log(
                        {"train/loss": loss_v, "lr": cur_lr, "epoch": epoch},
                        step=global_step,
                    )

            avg = ep_loss_sum / max(1, ep_loss_n)
            dt = time.perf_counter() - t0
            fs = _compute_feature_std(student, eval_images, device, use_amp)
            history.append(
                {"epoch": epoch, "loss": avg, "time_s": dt, "feature_std": fs}
            )
            print(
                f"[{cfg.task}] epoch {epoch + 1}/{cfg.epochs}  loss={avg:.4f}  "
                f"feat_std={fs:.4f}  time={dt:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
            wb.log(
                {
                    "train/epoch_loss": avg,
                    "feature_std": fs,
                    "epoch_time_s": dt,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

            # Persist history every epoch so external monitors can read partial state.
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

            # Save the epoch-50 checkpoint after the 50th epoch completes.
            if (epoch + 1) == 50:
                _save_checkpoint(
                    out_dir / "epoch50.pt",
                    cfg=cfg,
                    student=student,
                    head=head,
                    predictor=predictor,
                    feature_dim=feature_dim,
                    history=history,
                )
                print(f"[{cfg.task}] saved epoch50.pt", flush=True)

            # SimSiam representation-collapse abort: feature_std < 0.1 for two
            # consecutive epochs after epoch 10 (1-indexed).
            if cfg.task == "simsiam" and (epoch + 1) > 10 and len(history) >= 2:
                if history[-1]["feature_std"] < 0.1 and history[-2]["feature_std"] < 0.1:
                    abort_reason = (
                        f"SimSiam representation collapse: feature_std "
                        f"{history[-2]['feature_std']:.4f} -> {history[-1]['feature_std']:.4f} "
                        f"both < 0.1 after epoch 10"
                    )
                    (out_dir / "aborted.txt").write_text(abort_reason + "\n")
                    print(f"[simsiam] ABORTING: {abort_reason}", flush=True)
                    aborted = True
                    break
    finally:
        log_file.close()

    final_path = out_dir / ("aborted.pt" if aborted else "final.pt")
    _save_checkpoint(
        final_path,
        cfg=cfg,
        student=student,
        head=head,
        predictor=predictor,
        feature_dim=feature_dim,
        history=history,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    final_loss = history[-1]["loss"] if history else float("nan")
    final_fs = history[-1]["feature_std"] if history else init_fs
    wb.summary(
        {
            "final/epoch_loss": final_loss,
            "final/feature_std": final_fs,
            "final/epochs_run": len(history),
            "aborted": aborted,
            "abort_reason": abort_reason or "",
        }
    )
    wandb_run_dir = wb.run_dir
    wb.finish()

    return {
        "history": history,
        "out_dir": str(out_dir),
        "aborted": aborted,
        "abort_reason": abort_reason,
        "final_path": str(final_path),
        "wandb_run_dir": wandb_run_dir,
    }
