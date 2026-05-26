"""Minimal DDP helpers (torchrun-driven).

CLAUDE.md mandates that distributed code work with both
``torchrun --nproc_per_node=4`` and a single GPU. When torchrun env vars
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``) are absent we degrade to a
single-process world of size 1, so the same code path runs unchanged on one
GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Init the process group from torchrun env vars, or run single-process."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        # Generous timeout: rank-0 does solo work (BN recalibration before
        # final save, ~2-3 min) while other ranks wait at the cleanup barrier.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
        dist.barrier()
        return DistInfo(rank, local_rank, world_size, distributed=True)
    return DistInfo(rank=0, local_rank=0, world_size=1, distributed=False)


def cleanup_distributed(info: DistInfo) -> None:
    if info.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: float, info: DistInfo, device: torch.device) -> float:
    """Average a Python scalar across ranks (no-op when single-process)."""
    if not info.distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / info.world_size)


def broadcast_flag(flag: bool, info: DistInfo, device: torch.device) -> bool:
    """Broadcast a boolean from rank 0 to all ranks (no-op single-process)."""
    if not info.distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())
