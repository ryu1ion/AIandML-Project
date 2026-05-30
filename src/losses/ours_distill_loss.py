"""Our distillation losses: local structural, global semantic, cross-view invariant.

Each loss is a standalone nn.Module so it can be unit-tested independently.
OursDistillLoss orchestrates all three and returns a dict of loss components.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LocalStructuralLoss(nn.Module):
    """Distill patch-level spatial relations from teacher to student.

    Computes patch-to-patch similarity matrices for both teacher and student,
    then minimizes their divergence (MSE or KL).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mode: str = "mse",
        relation_mode: str = "full",
        max_tokens: int = 196,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.relation_mode = relation_mode
        self.max_tokens = max_tokens

    def forward(self, s_patch: Tensor, t_patch: Tensor) -> Tensor:
        """
        s_patch: (B, N_s, C_s) student patch features
        t_patch: (B, N_t, C_t) teacher patch features (detached)
        """
        s_patch = s_patch.float()
        t_patch = t_patch.float()

        N_s = s_patch.shape[1]
        N_t = t_patch.shape[1]

        if self.relation_mode == "sample" and N_s > self.max_tokens:
            idx = torch.randperm(N_s, device=s_patch.device)[: self.max_tokens]
            s_patch = s_patch[:, idx]
        if self.relation_mode == "sample" and N_t > self.max_tokens:
            idx = torch.randperm(N_t, device=t_patch.device)[: self.max_tokens]
            t_patch = t_patch[:, idx]

        s_patch = F.normalize(s_patch, dim=-1)
        t_patch = F.normalize(t_patch, dim=-1)

        R_s = s_patch @ s_patch.transpose(-1, -2)  # (B, N, N)
        R_t = t_patch @ t_patch.transpose(-1, -2)

        if self.mode == "mse":
            return ((R_s - R_t) ** 2).mean()

        R_s_log = F.log_softmax(R_s / self.temperature, dim=-1)
        R_t_soft = F.softmax(R_t / self.temperature, dim=-1)
        kl = F.kl_div(R_s_log, R_t_soft, reduction="batchmean")
        return kl


class GlobalSemanticLoss(nn.Module):
    """Transfer teacher's cross-sample semantic geometry to the student.

    Computes batch-level similarity matrices from global features and
    minimizes their divergence.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mode: str = "kl",
        mask_diagonal: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.mask_diagonal = mask_diagonal

    def forward(self, s_global: Tensor, t_global: Tensor) -> Tensor:
        """
        s_global: (B, C_s) student global features
        t_global: (B, C_t) teacher global features (detached)
        """
        s_global = s_global.float()
        t_global = t_global.float()

        s_global = F.normalize(s_global, dim=-1)
        t_global = F.normalize(t_global, dim=-1)

        G_s = s_global @ s_global.T  # (B, B)
        G_t = t_global @ t_global.T

        if self.mask_diagonal:
            B = G_s.shape[0]
            mask = ~torch.eye(B, dtype=torch.bool, device=G_s.device)
            if self.mode == "mse":
                return ((G_s - G_t) ** 2)[mask].mean()
            # For KL: zero out diagonal before softmax to exclude self-similarity
            large_neg = torch.tensor(-1e9, device=G_s.device)
            G_s = G_s.masked_fill(~mask, large_neg)
            G_t = G_t.masked_fill(~mask, large_neg)

        if self.mode == "mse":
            return ((G_s - G_t) ** 2).mean()

        G_s_log = F.log_softmax(G_s / self.temperature, dim=-1)
        G_t_soft = F.softmax(G_t / self.temperature, dim=-1)
        kl = F.kl_div(G_s_log, G_t_soft, reduction="batchmean")
        return kl


class CrossViewInvariantLoss(nn.Module):
    """Distill teacher's cross-view invariance to the student.

    For two views of the same images, computes cross-view similarity matrices
    and minimizes their divergence between teacher and student.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mode: str = "kl",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(
        self,
        s_v1: Tensor,
        s_v2: Tensor,
        t_v1: Tensor,
        t_v2: Tensor,
    ) -> Tensor:
        """
        s_v1, s_v2: (B, C_s) student global features for view 1 and 2
        t_v1, t_v2: (B, C_t) teacher global features for view 1 and 2 (detached)
        """
        s_v1 = F.normalize(s_v1.float(), dim=-1)
        s_v2 = F.normalize(s_v2.float(), dim=-1)
        t_v1 = F.normalize(t_v1.float(), dim=-1)
        t_v2 = F.normalize(t_v2.float(), dim=-1)

        teacher_cross = t_v1 @ t_v2.T  # (B, B)
        student_cross = s_v1 @ s_v2.T

        if self.mode == "kl":
            s_log = F.log_softmax(student_cross / self.temperature, dim=-1)
            t_soft = F.softmax(teacher_cross / self.temperature, dim=-1)
            return F.kl_div(s_log, t_soft, reduction="batchmean")

        return ((student_cross - teacher_cross) ** 2).mean()


class OursDistillLoss(nn.Module):
    """Orchestrator combining base L2 distill + our three auxiliary losses."""

    def __init__(
        self,
        lambda_local: float = 1.0,
        lambda_global: float = 0.5,
        lambda_view: float = 0.5,
        local_temperature: float = 0.1,
        global_temperature: float = 0.1,
        view_temperature: float = 0.1,
        local_mode: str = "mse",
        global_mode: str = "kl",
        view_mode: str = "kl",
        local_relation_mode: str = "full",
        local_max_tokens: int = 196,
        global_mask_diagonal: bool = True,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_local = lambda_local
        self.lambda_global = lambda_global
        self.lambda_view = lambda_view
        self.warmup_steps = warmup_steps

        self.local_loss = LocalStructuralLoss(
            temperature=local_temperature,
            mode=local_mode,
            relation_mode=local_relation_mode,
            max_tokens=local_max_tokens,
        )
        self.global_loss = GlobalSemanticLoss(
            temperature=global_temperature,
            mode=global_mode,
            mask_diagonal=global_mask_diagonal,
        )
        self.view_loss = CrossViewInvariantLoss(
            temperature=view_temperature,
            mode=view_mode,
        )

    def _warmup_factor(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return 1.0
        return step / self.warmup_steps

    def forward(
        self,
        *,
        s_patch_v1: Tensor | None = None,
        s_patch_v2: Tensor | None = None,
        t_patch_v1: Tensor | None = None,
        t_patch_v2: Tensor | None = None,
        s_global_v1: Tensor,
        s_global_v2: Tensor,
        t_global_v1: Tensor,
        t_global_v2: Tensor,
        step: int = 0,
    ) -> dict[str, Tensor]:
        w = self._warmup_factor(step)
        losses: dict[str, Tensor] = {}
        zero = torch.tensor(0.0, device=s_global_v1.device)

        # Local structural loss (averaged over views)
        if (
            self.lambda_local > 0
            and s_patch_v1 is not None
            and t_patch_v1 is not None
        ):
            l_local_v1 = self.local_loss(s_patch_v1, t_patch_v1)
            l_local_v2 = self.local_loss(s_patch_v2, t_patch_v2) if (
                s_patch_v2 is not None and t_patch_v2 is not None
            ) else l_local_v1
            losses["loss_local"] = 0.5 * (l_local_v1 + l_local_v2)
        else:
            losses["loss_local"] = zero

        # Global semantic loss (concatenate both views into one batch)
        if self.lambda_global > 0:
            s_global_cat = torch.cat([s_global_v1, s_global_v2], dim=0)
            t_global_cat = torch.cat([t_global_v1, t_global_v2], dim=0)
            losses["loss_global"] = self.global_loss(s_global_cat, t_global_cat)
        else:
            losses["loss_global"] = zero

        # Cross-view invariant loss
        if self.lambda_view > 0:
            losses["loss_view"] = self.view_loss(
                s_global_v1, s_global_v2, t_global_v1, t_global_v2
            )
        else:
            losses["loss_view"] = zero

        losses["loss_ours"] = (
            w * self.lambda_local * losses["loss_local"]
            + w * self.lambda_global * losses["loss_global"]
            + w * self.lambda_view * losses["loss_view"]
        )
        losses["warmup_factor"] = torch.tensor(w, device=s_global_v1.device)
        return losses
