"""Training loop for label-free MobileNetV2 ← DINO ViT-S/16 distillation.

Three task variants share the same teacher / student / augmentation /
optimizer pipeline; only the loss differs:

- 'distill' : Base — L2 on L2-normalized features against the teacher CLS.
- 'ours'    : Base + three optional auxiliary structural losses
              (Local Structural, Global Semantic, Cross-View Invariant).
- 'paperkd' : Base + SP-KD (Tung & Mori, ICCV 2019)
              and RKD-distance (Park et al., CVPR 2019).

Runs on a single GPU (plain PyTorch) or multi-GPU via
``torchrun --nproc_per_node=N`` (DDP). bf16 autocast is on by default.
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
    get_mobilenetv2_spatial_module,
)
from src.losses.feature_matching import l2_normalized_mse_loss
from src.losses.ours_distill_loss import OursDistillLoss
from src.losses.paper_kd_losses import (
    rkd_distance_loss,
    similarity_preserving_loss,
)
from src.projection_heads import MLPProjectionHead, PatchProjectionHead
from src.students import get_student
from src.teachers import get_teacher
from src.utils.distributed import (
    DistInfo,
    all_reduce_mean,
    cleanup_distributed,
    init_distributed,
)
from src.utils.wandb_logging import WandbLogger

Task = Literal["distill", "ours", "paperkd"]


@dataclass
class TrainConfig:
    task: Task
    student: str = "mobilenetv2"
    teacher: str = "dino_vits16"
    dataset: str = "cifar100"  # 'cifar100' or 'imagenet100'
    data_root: str = "data"
    image_size: int = 224
    epochs: int = 100
    batch_size: int = 256  # per-GPU batch size
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adamw' or 'sgd'
    schedule: str = "cosine"  # 'cosine' or 'constant'
    warmup_epochs: int = 0
    # Continue training from a previous final.pt (loads student/head/predictor
    # state_dicts only; optimizer/schedule are reset).
    resume_from: str | None = None
    # LR scaling vs global batch: 'none' (default) or 'linear' (lr * global_batch / 256).
    lr_scale_rule: str = "none"
    sync_bn: bool = False
    two_view_aug: str = "in100"  # IN-100 two-view aug: 'in100' or 'mild'
    bn_recalib_on_save: bool = True
    bn_recalib_batches: int = 200
    num_workers: int = 8
    seed: int = 42
    bf16: bool = True
    proj_hidden: int = 768  # 2 * teacher_dim (DINO ViT-S/16 dim = 384)
    proj_out: int = 384     # teacher feature dim
    out_dir: str = "checkpoints/run"
    log_every: int = 50
    # debug/smoke knob (not a hyperparameter): cap iterations per epoch.
    limit_train_batches: int = 0
    wandb: bool = False
    wandb_project: str = "label-free-distill"
    wandb_mode: str = "offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    student_pretrained: bool = False
    # "ours" method config: three auxiliary distillation objectives ADDED on
    # top of the base L2-normalized feature-matching loss (L_base is preserved
    # exactly). Final objective:
    #   L_total = L_base
    #           + lambda_local * L_local_structural
    #           + lambda_global * L_global_semantic
    #           + lambda_cross  * L_cross_view_invariant
    # Each term can be independently disabled via use_*_loss flags.
    use_local_structural_loss: bool = True
    use_global_semantic_loss: bool = True
    use_cross_view_invariant_loss: bool = True
    lambda_local: float = 0.1
    lambda_global: float = 0.5
    lambda_cross: float = 0.5
    # Backward-compat alias (older configs/CLI used lambda_view); if non-None,
    # it overrides lambda_cross. New code should use lambda_cross.
    lambda_view: float | None = None
    # Separate teacher / student temperatures for relation distillation
    # (KL mode only). MSE mode ignores temperatures.
    relation_temperature_teacher: float = 0.07
    relation_temperature_student: float = 0.1
    patch_relation_loss_type: str = "mse"     # "mse" or "kl"
    global_relation_loss_type: str = "mse"    # "mse" or "kl"
    cross_view_relation_loss_type: str = "mse"  # "mse" or "kl"
    local_relation_mode: str = "full"  # "full", "sample"
    local_max_tokens: int = 196
    global_mask_diagonal: bool = True
    ours_warmup_frac: float = 0.0  # fraction of total steps for aux loss warmup
    ours_use_patch_proj: bool = True  # project student patches to teacher dim
    ours_patch_proj_dim: int = 384
    # Legacy fields kept so older YAML configs still load (now unused in the
    # default code path — superseded by relation_temperature_{teacher,student}).
    local_temperature: float = 0.1
    global_temperature: float = 0.1
    view_temperature: float = 0.1
    local_mode: str | None = None   # legacy alias for patch_relation_loss_type
    global_mode: str | None = None  # legacy alias for global_relation_loss_type
    view_mode: str | None = None    # legacy alias for cross_view_relation_loss_type
    # Paper-backed additive distillation (task="paperkd"):
    #   L_total = L_base + lambda_sp * L_SP + lambda_rkd * L_RKD-D
    # - SP-KD: Tung & Mori, ICCV 2019
    # - RKD-D (distance-wise): Park et al., CVPR 2019
    # Setting both lambdas to 0 reproduces the base method exactly.
    lambda_sp: float = 1.0
    lambda_rkd: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_modules(cfg: TrainConfig, device: torch.device) -> dict[str, nn.Module | None]:
    student = get_student(cfg.student, pretrained=cfg.student_pretrained).to(device)
    teacher = get_teacher(cfg.teacher).to(device).eval()
    head = MLPProjectionHead(
        student.feature_dim, out_dim=cfg.proj_out, hidden_dim=cfg.proj_hidden
    ).to(device)

    # task="ours" optionally adds a small loss-only patch projection head
    # (used only when L_local-structural is active).
    predictor: nn.Module | None = None
    if cfg.task == "ours" and cfg.ours_use_patch_proj:
        predictor = PatchProjectionHead(
            in_dim=student.feature_dim,
            out_dim=cfg.ours_patch_proj_dim,
        ).to(device)

    if cfg.task not in ("distill", "ours", "paperkd"):
        raise ValueError(f"Unknown task: {cfg.task}")

    return {"student": student, "head": head, "teacher": teacher, "predictor": predictor}


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
    # All supported tasks (distill / ours / paperkd) consume two augmented views.
    ds = _get_train_dataset(cfg, mode="two_view")
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
    """Build a single-group optimizer over student + head (+ predictor if any)."""
    eff_lr = _effective_lr(cfg, world_size)
    params = list(student.parameters()) + list(head.parameters())
    if predictor is not None:
        params += list(predictor.parameters())
    groups = [{"params": params, "lr": eff_lr}]

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

    # task-specific setup
    ours_loss_fn: OursDistillLoss | None = None
    ours_spatial_hook: MidFeatureGrabber | None = None
    if cfg.task == "paperkd" and is_main:
        _s_raw = _unwrap(student)
        _t_raw = teacher
        print(
            f"[paperkd] === Sanity check ===\n"
            f"  L_total = L_base "
            f"+ {cfg.lambda_sp} * L_SP (Tung & Mori, ICCV 2019) "
            f"+ {cfg.lambda_rkd} * L_RKD-D (Park et al., CVPR 2019)\n"
            f"  teacher: {cfg.teacher}, feature_dim={_t_raw.feature_dim}\n"
            f"  teacher frozen: "
            f"{not any(p.requires_grad for p in _t_raw.parameters())}\n"
            f"  student: {cfg.student}, feature_dim={_s_raw.feature_dim}\n"
            f"  projector: MLP({_s_raw.feature_dim}->{cfg.proj_hidden}->{cfg.proj_out})\n"
            f"  batch_size: {cfg.batch_size}, num_views: 2\n"
            f"  DDP: {dist_info.distributed}\n"
            f"  set lambda_sp=lambda_rkd=0 to reproduce base\n"
            f"========================",
            flush=True,
        )
    if cfg.task == "ours":
        ours_spatial_hook = MidFeatureGrabber(
            get_mobilenetv2_spatial_module(_unwrap(student))
        )
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = int(cfg.ours_warmup_frac * total_steps)
        # Apply use_*_loss switches by zeroing out lambdas (cheap and clean).
        eff_lambda_local = cfg.lambda_local if cfg.use_local_structural_loss else 0.0
        eff_lambda_global = cfg.lambda_global if cfg.use_global_semantic_loss else 0.0
        # Backward-compat: lambda_view (if set) overrides lambda_cross.
        eff_lambda_cross = cfg.lambda_view if cfg.lambda_view is not None else cfg.lambda_cross
        if not cfg.use_cross_view_invariant_loss:
            eff_lambda_cross = 0.0
        # Resolve loss-type for each term (legacy *_mode fields override if set).
        local_type = cfg.local_mode or cfg.patch_relation_loss_type
        global_type = cfg.global_mode or cfg.global_relation_loss_type
        cross_type = cfg.view_mode or cfg.cross_view_relation_loss_type
        ours_loss_fn = OursDistillLoss(
            lambda_local=eff_lambda_local,
            lambda_global=eff_lambda_global,
            lambda_view=eff_lambda_cross,
            local_temperature=cfg.relation_temperature_student,
            global_temperature=cfg.relation_temperature_student,
            view_temperature=cfg.relation_temperature_student,
            local_mode=local_type,
            global_mode=global_type,
            view_mode=cross_type,
            local_relation_mode=cfg.local_relation_mode,
            local_max_tokens=cfg.local_max_tokens,
            global_mask_diagonal=cfg.global_mask_diagonal,
            warmup_steps=warmup_steps,
        )
        if is_main:
            _s_raw = _unwrap(student)
            _t_raw = teacher
            print(
                f"[ours] === Sanity check ===\n"
                f"  L_total = L_base "
                f"+ {eff_lambda_local} * L_local "
                f"+ {eff_lambda_global} * L_global "
                f"+ {eff_lambda_cross} * L_cross\n"
                f"  use_local={cfg.use_local_structural_loss} "
                f"use_global={cfg.use_global_semantic_loss} "
                f"use_cross={cfg.use_cross_view_invariant_loss}\n"
                f"  relation types: local={local_type} "
                f"global={global_type} cross={cross_type}\n"
                f"  temperatures: teacher={cfg.relation_temperature_teacher} "
                f"student={cfg.relation_temperature_student}\n"
                f"  teacher patch tokens: ({_t_raw.feature_dim},) per patch, "
                f"14x14=196 patches at 224x224\n"
                f"  student feature_dim: {_s_raw.feature_dim}\n"
                f"  use_patch_proj: {cfg.ours_use_patch_proj} "
                f"(out_dim={cfg.ours_patch_proj_dim})\n"
                f"  proj_out (global head): {cfg.proj_out}\n"
                f"  batch_size (per-GPU): {cfg.batch_size} "
                f"global={cfg.batch_size * dist_info.world_size}\n"
                f"  num_views: 2\n"
                f"  DDP: {dist_info.distributed}\n"
                f"  warmup_steps: {warmup_steps}/{total_steps}\n"
                f"========================",
                flush=True,
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
                if cfg.task == "distill":
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
                elif cfg.task == "paperkd":
                    # base + lambda_sp * SP-KD + lambda_rkd * RKD-distance
                    # All three terms averaged over the two views (same
                    # pattern as task="distill" for the base term).
                    (x1, x2), _ = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        s1 = head(student(x1))
                        s2 = head(student(x2))
                        with torch.no_grad():
                            t1 = teacher(x1)
                            t2 = teacher(x2)

                        # Base loss (UNCHANGED from task="distill")
                        loss_base = 0.5 * (
                            l2_normalized_mse_loss(s1, t1)
                            + l2_normalized_mse_loss(s2, t2)
                        )

                        # SP-KD (Tung & Mori, ICCV 2019) — computed in fp32
                        # for stable Gram MSE under bf16 autocast.
                        if cfg.lambda_sp > 0:
                            loss_sp = 0.5 * (
                                similarity_preserving_loss(s1.float(), t1.float())
                                + similarity_preserving_loss(s2.float(), t2.float())
                            )
                        else:
                            loss_sp = torch.zeros(
                                (), device=device, dtype=torch.float32
                            )

                        # RKD-distance (Park et al., CVPR 2019)
                        if cfg.lambda_rkd > 0:
                            loss_rkd = 0.5 * (
                                rkd_distance_loss(s1.float(), t1.float())
                                + rkd_distance_loss(s2.float(), t2.float())
                            )
                        else:
                            loss_rkd = torch.zeros(
                                (), device=device, dtype=torch.float32
                            )

                        loss = (
                            loss_base
                            + cfg.lambda_sp * loss_sp
                            + cfg.lambda_rkd * loss_rkd
                        )

                    if is_main and global_step % cfg.log_every == 0:
                        wb.log(
                            {
                                "train/loss_base": float(loss_base.item()),
                                "train/loss_sp": float(loss_sp.item()),
                                "train/loss_rkd_distance": float(loss_rkd.item()),
                                "train/loss_total": float(loss.item()),
                                "train/lambda_sp": cfg.lambda_sp,
                                "train/lambda_rkd": cfg.lambda_rkd,
                            },
                            step=global_step,
                        )
                        if global_step <= cfg.log_every * 3:
                            print(
                                f"  [paperkd] step={global_step} "
                                f"base={loss_base.item():.4f} "
                                f"sp={loss_sp.item():.4f} "
                                f"rkd={loss_rkd.item():.4f} "
                                f"total={loss.item():.4f}",
                                flush=True,
                            )
                elif cfg.task == "ours":
                    (x1, x2), _ = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        # Student forward through DDP wrapper; grab spatial
                        # features from the hook before pooling. Clone v1's
                        # spatial map before the v2 forward overwrites the hook
                        # (only when the local loss is enabled — otherwise the
                        # spatial maps are unused and we skip the clone to save
                        # memory).
                        need_spatial = cfg.use_local_structural_loss and cfg.lambda_local > 0
                        s_pooled_v1 = student(x1)
                        s_spatial_v1 = ours_spatial_hook.feat.clone() if need_spatial else None
                        s_pooled_v2 = student(x2)
                        s_spatial_v2 = ours_spatial_hook.feat if need_spatial else None
                        s_global_v1 = head(s_pooled_v1)
                        s_global_v2 = head(s_pooled_v2)

                        with torch.no_grad():
                            if need_spatial:
                                t_cls_v1, t_patch_v1 = teacher.forward_patch_features(x1)
                                t_cls_v2, t_patch_v2 = teacher.forward_patch_features(x2)
                            else:
                                t_cls_v1 = teacher(x1)
                                t_cls_v2 = teacher(x2)
                                t_patch_v1 = None
                                t_patch_v2 = None

                        # Base L2 loss (same as task=distill)
                        loss_base = 0.5 * (
                            l2_normalized_mse_loss(s_global_v1, t_cls_v1)
                            + l2_normalized_mse_loss(s_global_v2, t_cls_v2)
                        )

                        # Only build patch tokens when the local loss is on
                        # (saves ~400MB of activations and avoids OOM at bs=256
                        # when other processes share the GPU).
                        if cfg.use_local_structural_loss and eff_lambda_local > 0:
                            s_spatial_v1_up = F.interpolate(
                                s_spatial_v1, size=(14, 14), mode="bilinear",
                                align_corners=False,
                            )
                            s_spatial_v2_up = F.interpolate(
                                s_spatial_v2, size=(14, 14), mode="bilinear",
                                align_corners=False,
                            )
                            s_patch_v1 = s_spatial_v1_up.flatten(2).transpose(1, 2)  # (B, 196, 1280)
                            s_patch_v2 = s_spatial_v2_up.flatten(2).transpose(1, 2)
                            if predictor is not None:  # patch projection head
                                s_patch_v1 = predictor(s_patch_v1)
                                s_patch_v2 = predictor(s_patch_v2)
                        else:
                            s_patch_v1 = None
                            s_patch_v2 = None
                            t_patch_v1 = None
                            t_patch_v2 = None

                        ours_comps = ours_loss_fn(
                            s_patch_v1=s_patch_v1,
                            s_patch_v2=s_patch_v2,
                            t_patch_v1=t_patch_v1,
                            t_patch_v2=t_patch_v2,
                            s_global_v1=s_global_v1,
                            s_global_v2=s_global_v2,
                            t_global_v1=t_cls_v1,
                            t_global_v2=t_cls_v2,
                            step=global_step,
                        )
                        loss = loss_base + ours_comps["loss_ours"]

                    if is_main and global_step % cfg.log_every == 0:
                        b = float(loss_base.item())
                        l = float(ours_comps["loss_local"].item())
                        g = float(ours_comps["loss_global"].item())
                        c = float(ours_comps["loss_view"].item())
                        # Cosine sim diagnostic for student-vs-teacher CLS
                        with torch.no_grad():
                            cos = F.cosine_similarity(
                                s_global_v1.float(), t_cls_v1.float(), dim=-1
                            ).mean().item()
                        log_d = {
                            "train/loss_base": b,
                            "train/loss_local": l,
                            "train/loss_global": g,
                            "train/loss_cross": c,
                            "train/loss_aux_weighted": float(ours_comps["loss_ours"].item()),
                            "train/loss_total": float(loss.item()),
                            "train/warmup_factor": float(ours_comps["warmup_factor"].item()),
                            "train/cos_sim_st": cos,
                            "train/s_norm": float(s_global_v1.detach().float().norm(dim=-1).mean().item()),
                            "train/t_norm": float(t_cls_v1.float().norm(dim=-1).mean().item()),
                            # Ratios: how big is base vs each auxiliary term?
                            # Useful for picking sensible lambda values.
                            "train/ratio_base_over_local": b / max(l, 1e-12),
                            "train/ratio_base_over_global": b / max(g, 1e-12),
                            "train/ratio_base_over_cross": b / max(c, 1e-12),
                            "train/lambda_local": eff_lambda_local,
                            "train/lambda_global": eff_lambda_global,
                            "train/lambda_cross": eff_lambda_cross,
                        }
                        wb.log(log_d, step=global_step)
                        if global_step <= cfg.log_every * 3:
                            print(
                                f"  [ours] step={global_step} "
                                f"base={b:.4f} local={l:.4f} global={g:.4f} "
                                f"cross={c:.4f} total={loss.item():.4f} "
                                f"cos_st={cos:.3f} "
                                f"ratios(b/l,b/g,b/c)="
                                f"{b/max(l,1e-12):.1f},{b/max(g,1e-12):.1f},"
                                f"{b/max(c,1e-12):.1f}",
                                flush=True,
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
        "history": history,
    }
    if is_main:
        final_path = out_dir / "final.pt"
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
            }
        )
        result["final_path"] = str(final_path)
        result["wandb_run_dir"] = wb.run_dir
    wb.finish()
    cleanup_distributed(dist_info)
    return result
