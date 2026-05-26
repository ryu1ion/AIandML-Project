# Phase 2 run configs

One YAML per trainable run, driven by the single `scripts/train.py`.

| Run | Config | Method |
|-----|--------|--------|
| R1 | _(none — eval-only)_ | Random-init MobileNetV2; no training. Evaluated directly via `scripts/eval_checkpoint.py --checkpoint random`. |
| R2 | `r2_supervised.yaml` | Supervised CE (labels-only reference) |
| R3 | `r3_simsiam.yaml` | SimSiam SSL from scratch (no teacher) |
| R4 | `r4_distill_l2.yaml` | Label-free L2 distillation from DINO ViT-S/16 |

Launch (4-GPU DDP, the per-run setting that keeps wall-clock < 6 h):

```bash
torchrun --nproc_per_node=4 --master_port=$PORT \
  scripts/train.py --config configs/experiment/phase2/r4_distill_l2.yaml
```

Single-GPU also works (`python scripts/train.py --config ...`) but a
100-epoch IN-100 run is ~10 h on one 3090 — DDP is required to satisfy
PHASE2's "no single run > 6 h" constraint.

LR scaling: `lr_scale_rule: linear` ⇒ `lr_eff = lr × (256·world_size)/256`
(global batch 1024 on 4 GPUs ⇒ ×4). R4 (AdamW) uses `none` — Adam is
batch-robust, per PHASE2 §4.
