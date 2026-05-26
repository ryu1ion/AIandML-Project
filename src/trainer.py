"""Training loop (preliminary CIFAR-100 + Phase 2 ImageNet-100).

Supports three tasks via cfg.task:
- 'supervised' : MobileNetV2 + linear classifier, CE loss, single view.
- 'simsiam'    : MobileNetV2 + projector + predictor, SimSiam loss, two views.
- 'distill'    : MobileNetV2 + projection head, L2-on-normalized to a frozen
                 DINO ViT-S/16 teacher; both views go through teacher and
                 student, loss averaged over views.

Runs on a single GPU (plain PyTorch) or multi-GPU via
``torchrun --nproc_per_node=N`` (DDP). The single-GPU code path is unchanged
from the preliminary phase, so CIFAR-100 results stay reproducible. bf16
autocast is on by default.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.cifar100 import get_cifar100
from src.losses.feature_hooks import (
    MidFeatureGrabber,
    get_mobilenetv2_mid_module,
)
from src.losses.feature_matching import l2_normalized_mse_loss
from src.losses.fitnet import FitNetAdapter, FitNetLoss
from src.losses.hinton_kd import HintonKDLoss
from src.losses.simsiam import SimSiamPredictor, SimSiamProjector, simsiam_loss
from src.projection_heads import MLPProjectionHead
from src.students import get_student
from src.teachers import get_teacher
from src.teachers.supervised_r50 import load_supervised_resnet50
from src.utils.distributed import (
    DistInfo,
    all_reduce_mean,
    broadcast_flag,
    cleanup_distributed,
    init_distributed,
)
from src.utils.wandb_logging import WandbLogger

Task = Literal["supervised", "simsiam", "distill", "hinton_kd", "fitnet"]


@dataclass
class TrainConfig:
    task: Task
    student: str = "mobilenetv2"
    teacher: str = "dino_vits16"  # only used when task='distill'
    dataset: str = "cifar100"  # 'cifar100' (preliminary) or 'imagenet100' (phase2)
    data_root: str = "data"
    image_size: int = 224
    epochs: int = 100
    batch_size: int = 256  # per-GPU batch size
    lr: float = 1e-3  # base LR (see lr_scale_rule)
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adamw' or 'sgd'
    schedule: str = "cosine"  # 'cosine' or 'constant'
    warmup_epochs: int = 0
    label_smoothing: float = 0.0  # supervised CE only
    # Continue training from a previous final.pt: loads student/head/predictor
    # state_dicts (NOT optimizer state). Fresh optimizer + schedule for the
    # next `epochs` epochs.
    resume_from: str | None = None
    # LR scaling vs global batch: 'none' (preliminary) or 'linear'
    # (lr_eff = lr * global_batch / 256, PHASE2 §4).
    lr_scale_rule: str = "none"
    sync_bn: bool = False  # SyncBatchNorm under DDP (phase2 IN-100 SSL)
    # IN-100 two-view aug for SSL/distill: 'in100' (PHASE2 §1 aggressive) or
    # 'mild' (preliminary-style RRC(0.4,1.0)+symmetric blur, lower target var).
    two_view_aug: str = "in100"
    # DDP+bf16 corrupts BN running stats (collapses eval-mode features).
    # Re-estimate them over eval-transform train data before saving final.pt
    # so checkpoints are directly usable; the evaluator also recalibrates.
    bn_recalib_on_save: bool = True
    bn_recalib_batches: int = 200
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
    # R5/R6 (NEW_BENCH): path to a supervised R-50 R_teacher final.pt and KD hparams.
    teacher_checkpoint: str | None = None
    kd_temperature: float = 4.0
    kd_alpha: float = 0.9
    fitnet_beta: float = 1.0
    student_pretrained: bool = False  # R_teacher fallback: load ImageNet weights


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
    student = get_student(cfg.student, pretrained=cfg.student_pretrained).to(device)

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

    if cfg.task in ("hinton_kd", "fitnet"):
        if not cfg.teacher_checkpoint:
            raise ValueError(
                f"task={cfg.task} requires teacher_checkpoint (path to R_teacher final.pt)"
            )
        head = _SupervisedHead(student.feature_dim, cfg.num_classes).to(device)
        teacher = load_supervised_resnet50(
            cfg.teacher_checkpoint, num_classes=cfg.num_classes, device=device
        )
        # FitNet uses an extra 1x1 adapter; expose it via the `predictor` slot
        # so the existing optimizer wiring picks up its parameters.
        predictor = None
        if cfg.task == "fitnet":
            predictor = FitNetAdapter(in_channels=96, out_channels=1024).to(device)
        return {
            "student": student,
            "head": head,
            "teacher": teacher,
            "predictor": predictor,
        }

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
            cfg.data_root,
            split="train",
            mode=mode,
            image_size=cfg.image_size,
            two_view_aug=cfg.two_view_aug,
        )
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _make_loader(
    cfg: TrainConfig, dist_info: DistInfo
) -> tuple[DataLoader, DistributedSampler | None]:
    # R5/R6 use labels -> single-view supervised batches like task=supervised.
    if cfg.task in ("supervised", "hinton_kd", "fitnet"):
        mode = "supervised"
    else:
        mode = "two_view"
    ds = _get_train_dataset(cfg, mode)
    sampler: DistributedSampler | None = None
    if dist_info.distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            drop_last=True,
        )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return loader, sampler


def _effective_lr(cfg: TrainConfig, world_size: int) -> float:
    """Apply the configured LR-vs-global-batch scaling rule (PHASE2 §4)."""
    if cfg.lr_scale_rule == "linear":
        global_batch = cfg.batch_size * world_size
        return cfg.lr * global_batch / 256.0
    if cfg.lr_scale_rule in ("none", ""):
        return cfg.lr
    raise ValueError(f"Unknown lr_scale_rule: {cfg.lr_scale_rule}")


def _build_optimizer(
    cfg: TrainConfig,
    *,
    student: nn.Module,
    head: nn.Module,
    predictor: nn.Module | None,
    world_size: int,
) -> tuple[torch.optim.Optimizer, list]:
    """Build the optimizer and the matching per-group LR-lambda list.

    SimSiam: the predictor uses a *fixed* base LR (no batch-scaling, no
    cosine decay) per Chen & He, 2021 §4.4; the backbone+projector use the
    scaled LR with warmup+cosine.
    """
    eff_lr = _effective_lr(cfg, world_size)
    base_params = list(student.parameters()) + list(head.parameters())

    if cfg.task == "simsiam" and predictor is not None:
        groups = [
            {"params": base_params, "lr": eff_lr},
            {"params": list(predictor.parameters()), "lr": cfg.lr, "fixed_lr": True},
        ]
    else:
        if predictor is not None:
            base_params += list(predictor.parameters())
        groups = [{"params": base_params, "lr": eff_lr}]

    name = cfg.optimizer.lower()
    if name == "adamw":
        optim = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    elif name == "sgd":
        optim = torch.optim.SGD(groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    return optim, groups


def _build_schedule(
    optim: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
    groups: list,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch

    def main_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if cfg.schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def fixed_lambda(step: int) -> float:
        return 1.0

    lambdas = [
        fixed_lambda if g.get("fixed_lr") else main_lambda for g in groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optim, lambdas)


def _make_eval_batch(cfg: TrainConfig, device: torch.device, n: int = 256) -> torch.Tensor:
    """Fixed deterministic batch for representation diagnostics.

    Uses the train split, the eval (no-augmentation) transform, and the first
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
) -> tuple[float, float]:
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
        norm_std = float(f_norm.std(dim=0).mean().item() * math.sqrt(d))
        # Raw between-sample variance (NOT normalized): detects the magnitude
        # collapse that the direction-only norm_std masks (e.g. the DDP+bf16
        # BN-eval collapse, where features become near-constant ~1e-12).
        raw_var = float(f.var(dim=0).mean().item())
        return norm_std, raw_var
    finally:
        if was_training:
            student.train()


def _unwrap(m: nn.Module) -> nn.Module:
    """Return the underlying module (strip DDP) for state_dict / inference."""
    return m.module if isinstance(m, DDP) else m


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
            "student_state_dict": _unwrap(student).state_dict(),
            "head_state_dict": _unwrap(head).state_dict(),
            "predictor_state_dict": (
                _unwrap(predictor).state_dict() if predictor is not None else None
            ),
            "config": asdict(cfg),
            "feature_dim": feature_dim,
            "history": history,
        },
        path,
    )


def _maybe_sync_bn(m: nn.Module, dist_info: DistInfo, cfg: TrainConfig) -> nn.Module:
    if dist_info.distributed and cfg.sync_bn:
        return nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return m


def _wrap_ddp(m: nn.Module, dist_info: DistInfo) -> nn.Module:
    if not dist_info.distributed:
        return m
    has_params = any(p.requires_grad for p in m.parameters())
    if not has_params:
        return m
    return DDP(m, device_ids=[dist_info.local_rank], find_unused_parameters=False)


def train(cfg: TrainConfig) -> dict[str, Any]:
    dist_info = init_distributed()
    is_main = dist_info.is_main
    set_seed(cfg.seed + dist_info.rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_info.local_rank}")
    else:
        device = torch.device("cpu")

    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str)
        )
        log_path = out_dir / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "loss", "lr"])

    loader, sampler = _make_loader(cfg, dist_info)
    steps_per_epoch = len(loader)

    modules = _build_modules(cfg, device)
    student: nn.Module = modules["student"]  # type: ignore[assignment]
    head: nn.Module = modules["head"]  # type: ignore[assignment]
    teacher = modules["teacher"]
    predictor = modules["predictor"]
    feature_dim = getattr(student, "feature_dim", None)

    if cfg.resume_from:
        if is_main:
            print(f"[{cfg.task}] resume_from={cfg.resume_from}", flush=True)
        ck = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        student.load_state_dict(ck["student_state_dict"])
        if head is not None and ck.get("head_state_dict") is not None:
            try:
                head.load_state_dict(ck["head_state_dict"])
            except RuntimeError as e:
                if is_main:
                    print(f"[{cfg.task}] head state_dict mismatch -> "
                          f"reinit head ({e!r})", flush=True)
        if predictor is not None and ck.get("predictor_state_dict") is not None:
            try:
                predictor.load_state_dict(ck["predictor_state_dict"])
            except RuntimeError as e:
                if is_main:
                    print(f"[{cfg.task}] predictor state_dict mismatch -> "
                          f"reinit predictor ({e!r})", flush=True)

    eval_images = _make_eval_batch(cfg, device) if is_main else None

    # Optimizer/schedule are built on the *raw* modules (params identical
    # across ranks; DDP syncs them at construction).
    optimizer, groups = _build_optimizer(
        cfg,
        student=student,
        head=head,
        predictor=predictor,
        world_size=dist_info.world_size,
    )
    scheduler = _build_schedule(optimizer, cfg, steps_per_epoch, groups)

    # SyncBN (optional) then DDP-wrap trainable modules. Teacher stays raw
    # (frozen, eval, no grad) and is never in the autograd graph.
    student = _wrap_ddp(_maybe_sync_bn(student, dist_info, cfg), dist_info)
    head = _wrap_ddp(_maybe_sync_bn(head, dist_info, cfg), dist_info)
    if predictor is not None:
        predictor = _wrap_ddp(_maybe_sync_bn(predictor, dist_info, cfg), dist_info)

    use_amp = cfg.bf16 and device.type == "cuda"

    # R5/R6 setup: loss objects + (FitNet) mid-feature hook on the student.
    hinton_loss_fn: HintonKDLoss | None = None
    fitnet_loss_fn: FitNetLoss | None = None
    fitnet_adapter = None
    student_mid_hook: MidFeatureGrabber | None = None
    if cfg.task == "hinton_kd":
        hinton_loss_fn = HintonKDLoss(
            temperature=cfg.kd_temperature, alpha=cfg.kd_alpha
        )
    elif cfg.task == "fitnet":
        fitnet_loss_fn = FitNetLoss(beta=cfg.fitnet_beta)
        fitnet_adapter = predictor  # DDP-wrapped (if distributed)
        student_mid_hook = MidFeatureGrabber(
            get_mobilenetv2_mid_module(_unwrap(student))
        )

    # feature_std runs on rank 0 only -> always use the *unwrapped* module so
    # the diagnostic forward never enters a DDP collective (a DDP forward on a
    # subset of ranks deadlocks).
    init_fs, init_rawvar = (
        _compute_feature_std(_unwrap(student), eval_images, device, use_amp)
        if is_main
        else (0.0, 0.0)
    )
    if is_main:
        eff_lr = _effective_lr(cfg, dist_info.world_size)
        print(
            f"[{cfg.task}] world_size={dist_info.world_size} "
            f"global_batch={cfg.batch_size * dist_info.world_size} "
            f"eff_lr={eff_lr:.4g}  init feature_std={init_fs:.4f} "
            f"raw_var={init_rawvar:.3e} (d={feature_dim})",
            flush=True,
        )

    wb = WandbLogger(
        enabled=cfg.wandb and is_main,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name or out_dir.name,
        config={**asdict(cfg), "world_size": dist_info.world_size},
        mode=cfg.wandb_mode,
        entity=cfg.wandb_entity,
        out_dir=str(out_dir),
    )
    if is_main and wb.run_dir is not None:
        (out_dir / "wandb_run.txt").write_text(
            f"mode={cfg.wandb_mode}\nproject={cfg.wandb_project}\n"
            f"run_dir={wb.run_dir}\n"
        )
    wb.log({"feature_std": init_fs, "epoch": 0}, step=0)

    history: list[dict[str, float]] = []
    aborted = False
    abort_reason: str | None = None
    global_step = 0
    log_file = open(log_path, "a", newline="") if is_main else None
    log_writer = csv.writer(log_file) if log_file is not None else None
    try:
        for epoch in range(cfg.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
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
                        loss = F.cross_entropy(
                            logits, y, label_smoothing=cfg.label_smoothing
                        )
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
                elif cfg.task == "hinton_kd":
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        s_logits = head(student(x))
                        with torch.no_grad():
                            t_logits = teacher.classify(x)
                        loss, comps = hinton_loss_fn(s_logits, t_logits, y)
                elif cfg.task == "fitnet":
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        s_pooled = student(x)
                        s_mid_raw = student_mid_hook.feat  # (B, 96, 14, 14)
                        s_mid = fitnet_adapter(s_mid_raw)
                        s_logits = head(s_pooled)
                        with torch.no_grad():
                            _, t_mid = teacher.classify_and_mid(x)
                        # Adapter output may be bf16; teacher_mid is float in
                        # no_grad. Cast teacher to match for stable MSE.
                        loss, comps = fitnet_loss_fn(
                            s_mid, t_mid.to(s_mid.dtype), s_logits, y
                        )
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

                if is_main and global_step % cfg.log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    log_writer.writerow([epoch, global_step, loss_v, cur_lr])
                    log_file.flush()
                    wb.log(
                        {"train/loss": loss_v, "lr": cur_lr, "epoch": epoch},
                        step=global_step,
                    )

            avg = ep_loss_sum / max(1, ep_loss_n)
            avg = all_reduce_mean(avg, dist_info, device)
            dt = time.perf_counter() - t0
            fs, raw_var = (
                _compute_feature_std(_unwrap(student), eval_images, device, use_amp)
                if is_main
                else (0.0, 0.0)
            )
            if is_main:
                history.append(
                    {
                        "epoch": epoch,
                        "loss": avg,
                        "time_s": dt,
                        "feature_std": fs,
                        "feature_raw_var": raw_var,
                    }
                )
                print(
                    f"[{cfg.task}] epoch {epoch + 1}/{cfg.epochs}  loss={avg:.4f}  "
                    f"feat_std={fs:.4f}  raw_var={raw_var:.3e}  time={dt:.1f}s  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                wb.log(
                    {
                        "train/epoch_loss": avg,
                        "feature_std": fs,
                        "feature_raw_var": raw_var,
                        "epoch_time_s": dt,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
                (out_dir / "history.json").write_text(json.dumps(history, indent=2))

            # Save checkpoints at epoch 30 / 70 / 100 (PHASE2 conventions).
            if is_main and (epoch + 1) in (30, 70, 100):
                _save_checkpoint(
                    out_dir / f"epoch{epoch + 1}.pt",
                    cfg=cfg,
                    student=student,
                    head=head,
                    predictor=predictor,
                    feature_dim=feature_dim,
                    history=history,
                )
                print(f"[{cfg.task}] saved epoch{epoch + 1}.pt", flush=True)

            # SimSiam representation-collapse abort: feature_std < 0.1 for two
            # consecutive epochs after epoch 10 (1-indexed). The decision is
            # made on rank 0 and broadcast so all ranks break together.
            do_abort = False
            if cfg.task == "simsiam" and is_main and (epoch + 1) > 10 and len(history) >= 2:
                if history[-1]["feature_std"] < 0.1 and history[-2]["feature_std"] < 0.1:
                    abort_reason = (
                        f"SimSiam representation collapse: feature_std "
                        f"{history[-2]['feature_std']:.4f} -> "
                        f"{history[-1]['feature_std']:.4f} both < 0.1 after epoch 10"
                    )
                    (out_dir / "aborted.txt").write_text(abort_reason + "\n")
                    print(f"[simsiam] ABORTING: {abort_reason}", flush=True)
                    do_abort = True
            do_abort = broadcast_flag(do_abort, dist_info, device)
            if do_abort:
                aborted = True
                break
    finally:
        if log_file is not None:
            log_file.close()

    # NOTE: BN running stats accumulated under DDP+bf16 are biased and
    # collapse eval-mode features. We intentionally do NOT recalibrate here:
    # a SyncBatchNorm train-mode pass during the rank-coordinated shutdown
    # deadlocks. Instead, `scripts/eval_checkpoint.py` re-estimates BN stats
    # (single-process, plain BN, deadlock-free) uniformly for every checkpoint
    # before extracting features — see `src.evaluator.recalibrate_bn`. The
    # per-epoch `feature_raw_var` diagnostic surfaces this collapse in logs.
    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "aborted": aborted,
        "abort_reason": abort_reason,
        "history": history,
    }
    if is_main:
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
        result["final_path"] = str(final_path)
        result["wandb_run_dir"] = wb.run_dir
    wb.finish()
    cleanup_distributed(dist_info)
    return result
