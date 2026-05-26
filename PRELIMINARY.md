# Preliminary Results — Execution Plan

## Context
This is a focused sub-task of the larger project described in `CLAUDE.md`. Read `CLAUDE.md` first for project context, conventions, and constraints. This document overrides any conflicting guidance in `CLAUDE.md` for the preliminary phase only.

## Goal
Produce a preliminary results section (1 table, 2 figures, ~1–2 pages of text) that demonstrates:
1. The label-free distillation pipeline runs end-to-end.
2. The teacher is loaded correctly and transfers meaningful representations.
3. Early signal that label-free distillation outperforms training-from-scratch baselines.

This is **not** the final results. Scope is deliberately small. Do not expand it.

## Hard Constraints for This Phase

1. **Dataset: CIFAR-100 only.** Do not use ImageNet-100, do not use STL-10 for distillation. STL-10 is used only for transfer evaluation.
2. **One teacher: DINO ViT-S/16.** Load via `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`. Do not load MoCo or DINOv2 yet.
3. **One student: MobileNetV2.** From timm, with `num_classes=0`. Do not try MobileNetV3 or ResNet-18 yet.
4. **One distillation loss: L2 on L2-normalized features.** No relational loss, no SEED, no multi-loss combinations.
5. **No hyperparameter tuning.** Use the defaults in this doc for all four runs. Identical hyperparameters across runs is more important than optimal hyperparameters.
6. **Total compute budget: 12 GPU-hours.** If you exceed this, stop and report.
7. **No new files outside the structure in `CLAUDE.md`.** Reuse the structure.

## The Four Runs

Run all four with identical optimizer, schedule, batch size, and number of epochs. Only the *training objective* changes.

| Run ID | Name | What it does |
|--------|------|--------------|
| `R1` | `random_init` | No training. Just load a fresh MobileNetV2. |
| `R2` | `supervised` | Train MobileNetV2 on CIFAR-100 with cross-entropy on labels. |
| `R3` | `ssl_scratch` | Train MobileNetV2 on CIFAR-100 with SimSiam (no teacher, no labels). |
| `R4` | `label_free_distill` | Train MobileNetV2 by matching DINO ViT-S/16 features via L2 loss (no labels). |

**Shared training config for R2, R3, R4:**
- Epochs: 100
- Batch size: 256
- Optimizer: AdamW, lr 1e-3, weight decay 1e-4
- Schedule: cosine decay to 0
- Mixed precision: bf16
- Seed: 42
- Single GPU per run (run all 4 in parallel across your 4 GPUs)

**Augmentation for R3 and R4:** two-view DINO-style augmentation (random resized crop, horizontal flip, color jitter, gaussian blur). Both views go through the teacher and student; loss is computed on each view and averaged.

**Augmentation for R2:** standard CIFAR-100 augmentation (random crop with padding 4, horizontal flip, normalization).

**Image size:** Resize CIFAR-100 to **224×224** for all runs. This is required because DINO ViT-S/16 expects 224. Yes, this is wasteful for CIFAR — accept it for the preliminary phase.

**Projection head for R4:** 2-layer MLP, `Linear(student_feat_dim, 2*teacher_dim) → BN → GELU → Linear(2*teacher_dim, teacher_dim)`. DINO ViT-S/16 teacher dim is 384.

## Evaluation Protocols

Run all three evaluations on all four checkpoints. That's 12 numbers total.

### Eval 1: Linear Probe on CIFAR-100
- Freeze backbone, train a single linear layer on CIFAR-100 train split.
- Optimizer: SGD, lr 0.1, momentum 0.9, weight decay 0
- Schedule: cosine decay over 100 epochs
- Batch size: 256
- Report: top-1 accuracy on CIFAR-100 test split.

### Eval 2: kNN on CIFAR-100
- Extract features (post-backbone, pre-projection-head) for train and test sets.
- L2-normalize features.
- k=20, cosine similarity, weighted vote with temperature 0.07 (DINO protocol).
- Report: top-1 accuracy.

### Eval 3: 5-shot Transfer to STL-10
- Extract features on STL-10 train and test (10 classes).
- Sample 5 examples per class as the labeled support set.
- Train a logistic regression classifier (sklearn, default settings) on those 50 examples.
- Evaluate on STL-10 test split.
- Repeat over **5 random seeds**, report mean ± std.

## Deliverables

When all four runs and all evaluations complete, produce:

### 1. Results table (`results/preliminary_table.md`)

```
| Method                  | Linear Probe (CIFAR-100) | kNN (CIFAR-100) | 5-shot STL-10 (mean ± std) |
|-------------------------|-------------------------|-----------------|---------------------------|
| R1: Random init         |                         |                 |                           |
| R2: Supervised          |                         |                 |                           |
| R3: SSL from scratch    |                         |                 |                           |
| R4: Label-free distill  |                         |                 |                           |
```

### 2. Figures (`results/figures/`)

- `fig1_loss_curves.png`: training loss over epochs for R3 and R4 on the same axes.
- `fig2_tsne.png`: t-SNE of features on 1000 CIFAR-100 test samples for R1, R3, R4 side-by-side, colored by class.

### 3. Sanity-check log (`results/teacher_sanity.txt`)

A short text file with:
- DINO ViT-S/16 linear probe accuracy on CIFAR-100 (expect ~80–84%)
- DINO ViT-S/16 output feature dim (expect 384)
- Inference time per batch of 256 on one 3090

### 4. Preliminary results writeup (`results/preliminary.md`)

~1–2 pages, structured as:

1. **Setup** (1 paragraph): exact teacher, student, dataset, loss, key hyperparameters.
2. **Sanity check** (1 short paragraph): cite the number from `teacher_sanity.txt` and confirm it's within 1–2% of DINO's published CIFAR-100 numbers.
3. **Main result**: the table from deliverable 1.
4. **Observations** (1 paragraph): describe what the numbers show. Compare R4 vs. R1 (pipeline check), R4 vs. R3 (does the teacher help?), R4 vs. R2 (is label-free competitive with supervised?). Be honest — if a comparison goes the wrong way, say so and hypothesize why.
5. **Next steps** (1 short paragraph): connect to the full project plan (scale to ImageNet-100, add relational loss, test more teachers).

Do not write speculative claims. Stick to what the numbers show.

## Execution Order

Do these strictly in order. After each step, report status before continuing.

**Step 1 — Verify teacher (~1 hour)**
- Implement `src/teachers/dino.py` with the loader and `forward_features` interface.
- Implement `scripts/verify_teacher.py`: loads DINO, runs linear probe on CIFAR-100, writes `results/teacher_sanity.txt`.
- Run it. If linear probe < 78% or > 90%, stop and investigate.

**STOP and report:** the sanity-check number.

**Step 2 — Implement training (~2 hours)**
- Implement `src/students/mobilenetv2.py` (thin wrapper around timm).
- Implement `src/projection_heads.py` (2-layer MLP).
- Implement `src/losses/feature_matching.py` (L2 on normalized features) and a SimSiam loss for R3.
- Implement `src/data/cifar100.py` with both standard and two-view augmentation modes.
- Implement `src/trainer.py` with a single training loop that handles all three objectives (supervised CE, SimSiam, distillation L2) via a `task` flag in the config.
- Implement `scripts/train.py` driven by a simple config (no Hydra needed yet — use a plain YAML or argparse).
- Run a 3-epoch smoke test of R4 on a single GPU. Confirm loss decreases.

**STOP and report:** smoke-test loss curve.

**Step 3 — Run all four (~3 hours wall-clock)**
- Launch R2, R3, R4 in parallel on 3 of the 4 GPUs. R1 needs no training.
- Monitor every 30 minutes. If any run diverges (NaN loss, accuracy stuck at random), stop and report.

**STOP and report:** final training losses for R2, R3, R4.

**Step 4 — Implement and run evaluations (~3 hours)**
- Implement `scripts/linear_probe.py`, `scripts/knn_eval.py`, `scripts/few_shot_transfer.py`.
- Run all three evals on all four checkpoints. Parallelize across 4 GPUs.

**STOP and report:** the populated results table.

**Step 5 — Figures and writeup (~2 hours)**
- Generate `fig1_loss_curves.png` from W&B logs.
- Generate `fig2_tsne.png` using sklearn TSNE (perplexity 30, 1000 samples).
- Write `results/preliminary.md`.

**STOP and report:** done. Show the writeup.

## What to Do If Things Go Wrong

- **Teacher accuracy way off:** check input normalization (DINO expects ImageNet mean/std, not CIFAR mean/std), check image size (224, not 32), check that BN is in eval mode for the frozen teacher.
- **R4 loss not decreasing:** check that teacher is in `eval()` mode and gradients are blocked through it. Check projection head output dim matches teacher dim (384). Check feature normalization is applied to both before L2.
- **R4 < R3:** likely the projection head is too small or the loss isn't normalized properly. Try cosine loss instead of L2 on normalized features (mathematically similar but sometimes more stable). If still failing, report and we'll discuss.
- **Out of memory:** drop batch size to 128 (and adjust LR by sqrt(2)). DINO ViT-S/16 forward at 224×224 with batch 256 fits on a 3090 but it's tight.

## What NOT to Do

- Do not run on ImageNet-100. That is for the next phase.
- Do not add a second teacher, second student, or second loss. Scope creep kills preliminary results.
- Do not tune hyperparameters between runs. Use the same config for R2, R3, R4.
- Do not skip R3 (SSL-from-scratch). It is the most important baseline.
- Do not skip the teacher sanity check. Many failures are silent and only caught here.
- Do not write speculative claims in the writeup. Numbers only.

## Definition of Done for Preliminary Results

All of:
- [ ] `results/teacher_sanity.txt` exists with a sane DINO accuracy number.
- [ ] All four checkpoints saved under `checkpoints/preliminary/`.
- [ ] `results/preliminary_table.md` populated with all 12 numbers.
- [ ] `results/figures/fig1_loss_curves.png` and `fig2_tsne.png` exist.
- [ ] `results/preliminary.md` is written and self-contained.
- [ ] All runs logged in W&B with public links recorded in `results/preliminary.md`.
- [ ] Total wall-clock time ≤ 12 GPU-hours.

Begin with Step 1. Confirm the plan, then start.
