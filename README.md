# Label-Free Distillation from Self-Supervised Teachers to Compact Students

Distill a frozen public DINO ViT-S/16 teacher into a MobileNetV2 student **without using any labels during distillation**, then evaluate the student via linear probe (CIFAR-100), kNN (CIFAR-100), and 5-shot transfer (STL-10).

This repository currently contains the **preliminary phase** (small-scale, CIFAR-100 only); see `PRELIMINARY.md` for the exact spec and `CLAUDE.md` for the broader project plan.

---

## Headline result

| Method | Linear Probe (CIFAR-100) | kNN (CIFAR-100) | 5-shot STL-10 (mean ± std) |
|--------|-------------------------|-----------------|----------------------------|
| R1: Random init | 11.05 | 9.64 | 17.30 ± 1.10 |
| R2: Supervised | 65.81 | 65.17 | 38.09 ± 3.80 |
| R3: SSL from scratch (SimSiam) | 18.48 | 12.41 | 26.62 ± 2.00 |
| **R4: Label-free distill (DINO → MobileNetV2)** | **75.96** | **70.26** | **46.17 ± 3.32** |

R4 wins on every metric — including +10.2 pp linear-probe and +8.1 pp 5-shot transfer over the supervised baseline trained on the same student/data.

Full writeup: [`results/preliminary.md`](results/preliminary.md). Figures in [`results/figures/`](results/figures/).

---

## What has been done

**Phase 1 — Teacher**: DINO ViT-S/16 loaded from torch hub, frozen, exposing a uniform `forward_features(x) -> Tensor` API (CLS-token embedding, dim 384). Verified by a CIFAR-100 linear probe (78.39% top-1, within the 78–90% acceptance window). See `src/teachers/dino.py`, `scripts/verify_teacher.py`, `results/teacher_sanity.txt`.

**Phase 2 — Student + distillation training**: MobileNetV2 student (timm `mobilenetv2_100`, num_classes=0, 1280-d), 2-layer MLP projection head `Linear → BN → GELU → Linear` to teacher dim, L2-on-L2-normalized-features loss, two-view DINO-style augmentation. Single training script (`src/trainer.py` + `scripts/train.py`) handles three tasks via a `task` flag: `supervised`, `simsiam`, `distill`. Per-epoch `feature_std` diagnostic and SimSiam representation-collapse abort built in. See `src/students/`, `src/projection_heads.py`, `src/losses/`, `src/data/`, `src/trainer.py`.

**Phase 3 — Four runs**: R1 random-init MobileNetV2 (no training), R2 supervised CE, R3 SimSiam from scratch, R4 label-free distillation. R2/R3/R4 all 100 epochs, AdamW lr=1e-3 / wd=1e-4, cosine schedule, batch 256, bf16, seed 42, single GPU per run, run in parallel across 3 GPUs. Total training compute ≈ 8.0 GPU-hours.

**Phase 4 — Three evaluations on each of the four checkpoints**: CIFAR-100 linear probe (SGD lr=0.1 / cosine / 100 epochs / bs=256), CIFAR-100 kNN (k=20, cosine, weighted-vote temperature 0.07 — DINO protocol), 5-shot STL-10 (sklearn `LogisticRegression` averaged over 5 random support sets). See `src/evaluator.py`, `scripts/eval_checkpoint.py`.

**Phase 5 — Figures and writeup**: training-loss curves for R3 vs R4 (twin y-axes), t-SNE of MobileNetV2 features on 1000 stratified CIFAR-100 test samples for R1/R3/R4 side-by-side. See `scripts/make_figures.py`.

Total compute used: **~9.7 GPU-hours / 12 GPU-hour budget**.

---

## Repo layout

```
DisMo/
├── README.md                          # this file
├── CLAUDE.md                          # full project plan
├── PRELIMINARY.md                     # preliminary-phase spec
├── src/
│   ├── teachers/dino.py               # DINO ViT-S/16 loader (frozen, eval mode)
│   ├── students/mobilenetv2.py        # timm MobileNetV2 with num_classes=0
│   ├── projection_heads.py            # 2-layer MLP head
│   ├── losses/
│   │   ├── feature_matching.py        # L2-on-norm + cosine
│   │   └── simsiam.py                 # projector + predictor + sym -cos loss
│   ├── data/
│   │   ├── augmentations.py           # supervised / two-view / eval pipelines
│   │   └── cifar100.py                # CIFAR-100 wrappers at 224×224
│   ├── trainer.py                     # training loop (supervised / simsiam / distill)
│   └── evaluator.py                   # linear_probe, knn, few_shot_logreg, extract_features
├── scripts/
│   ├── verify_teacher.py              # teacher sanity → results/teacher_sanity.txt
│   ├── train.py                       # one of {R2, R3, R4}; YAML config + CLI overrides
│   ├── eval_checkpoint.py             # all 3 evals on one checkpoint → JSON
│   ├── make_preliminary_table.py      # JSONs → results/preliminary_table.md
│   └── make_figures.py                # fig1 + fig2
├── checkpoints/preliminary/
│   ├── r2_supervised/  {config,history,log,epoch50.pt,final.pt}
│   ├── r3_simsiam/     {config,history,log,epoch50.pt,final.pt}
│   └── r4_distill/     {config,history,log,epoch50.pt,final.pt}
├── results/
│   ├── teacher_sanity.txt
│   ├── preliminary_table.md           # the headline numbers
│   ├── preliminary.md                 # full prose writeup
│   ├── eval_r{1..4}_*.json            # per-checkpoint eval details
│   └── figures/{fig1_loss_curves.png, fig2_tsne.png}
├── data/                              # CIFAR-100 + STL-10 (downloaded on demand)
└── logs/                              # stdout/stderr from background runs
```

---

## Setup

Tested on Linux + 4× NVIDIA RTX 3090 (24 GB each), Python 3.11, PyTorch 2.6 + CUDA 12.4.

```bash
pip install torch torchvision timm scikit-learn matplotlib pyyaml numpy
```

That's it — datasets are downloaded on first use into `data/`.

---

## How to run (preliminary phase)

All commands below assume the working directory is the repo root.

### 1. Verify the teacher (~30 s)

```bash
python scripts/verify_teacher.py
```

Loads DINO ViT-S/16, extracts CIFAR-100 features at 224×224 with ImageNet normalization, runs a linear probe, writes `results/teacher_sanity.txt`. Expected accuracy: 78–90%.

### 2. Train the four runs (R1 needs no training)

R1 (random init) requires no training. R2/R3/R4 each take ~3.5 h on a single RTX 3090; you can run all three in parallel on three different GPUs:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --task supervised --epochs 100 --batch-size 256 \
    --lr 1e-3 --weight-decay 1e-4 --optimizer adamw --schedule cosine \
    --num-workers 8 --bf16 1 --seed 42 \
    --out-dir checkpoints/preliminary/r2_supervised > logs/r2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --task simsiam --epochs 100 --batch-size 256 \
    --lr 1e-3 --weight-decay 1e-4 --optimizer adamw --schedule cosine \
    --num-workers 8 --bf16 1 --seed 42 \
    --out-dir checkpoints/preliminary/r3_simsiam > logs/r3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --task distill --epochs 100 --batch-size 256 \
    --lr 1e-3 --weight-decay 1e-4 --optimizer adamw --schedule cosine \
    --num-workers 8 --bf16 1 --seed 42 \
    --out-dir checkpoints/preliminary/r4_distill > logs/r4.log 2>&1 &

wait
```

Each writes `config.json`, `log.csv`, `history.json`, `epoch50.pt`, and `final.pt` under its `--out-dir`.

### 3. Evaluate all four checkpoints (~25 min on 4 GPUs in parallel)

Each eval extracts features for CIFAR-100 train/test and STL-10 train/test, then runs the three protocols on cached features:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint.py \
    --run-name r1_random_init --checkpoint random \
    --output results/eval_r1_random_init.json > logs/eval_r1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/eval_checkpoint.py \
    --run-name r2_supervised \
    --checkpoint checkpoints/preliminary/r2_supervised/final.pt \
    --output results/eval_r2_supervised.json > logs/eval_r2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/eval_checkpoint.py \
    --run-name r3_simsiam \
    --checkpoint checkpoints/preliminary/r3_simsiam/final.pt \
    --output results/eval_r3_simsiam.json > logs/eval_r3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/eval_checkpoint.py \
    --run-name r4_distill \
    --checkpoint checkpoints/preliminary/r4_distill/final.pt \
    --output results/eval_r4_distill.json > logs/eval_r4.log 2>&1 &

wait
```

Note: STL-10 first-time download is ~2.6 GB. Pre-downloading once before launching the four jobs in parallel avoids a race:

```bash
python -c "from torchvision import datasets; datasets.STL10('data', split='train', download=True); datasets.STL10('data', split='test', download=True)"
```

### 4. Build the table

```bash
python scripts/make_preliminary_table.py     # → results/preliminary_table.md
```

### 5. Build the figures

```bash
python scripts/make_figures.py               # → results/figures/{fig1,fig2}.png
```

`fig1` reads from `checkpoints/preliminary/r{3,4}_*/history.json`. `fig2` re-extracts features (a few seconds per checkpoint) and runs sklearn t-SNE.

---

## Configuration knobs

`scripts/train.py` accepts either a YAML config (`--config path.yaml`) or CLI flags; CLI overrides YAML. Key flags map 1:1 to the `TrainConfig` dataclass in `src/trainer.py`. The most useful ones:

```
--task {supervised,simsiam,distill}     # which run
--epochs INT                             # default 100
--batch-size INT                         # default 256
--lr FLOAT                               # default 1e-3 (AdamW)
--weight-decay FLOAT                     # default 1e-4
--optimizer {adamw,sgd}
--schedule {cosine,constant}
--bf16 {0,1}                             # default 1
--proj-out INT                           # teacher dim, default 384 for DINO ViT-S/16
--proj-hidden INT                        # default 2*proj_out
--out-dir PATH                           # where to save checkpoints + logs
```

---

## Per-epoch diagnostics

The trainer logs a representation-diversity diagnostic alongside loss:

`feature_std` = `(per-dim std of L2-normalized backbone features over a fixed 256-sample eval batch).mean() × √d`

With this scaling, fully diverse features → ~1.0, fully collapsed features → 0.0; the SimSiam path aborts the run if `feature_std < 0.1` for two consecutive epochs after epoch 10 (no R3 abort was triggered in the preliminary phase).

---

## Notable design choices

- All inputs are resized to **224×224** with **ImageNet** mean/std, including CIFAR-100 — required because DINO ViT-S/16 was trained at 224×224 with ImageNet normalization. This is wasteful for CIFAR (32×32 → 224×224 upsample) but consistent with PRELIMINARY.md.
- The student's pre-projection-head backbone features (1280-d) are what the eval scripts use; the projection head is *only* used during distillation training.
- Mixed precision (bf16) is on by default — set `--bf16 0` to disable.
- Seed 42 is set for Python, NumPy, and PyTorch (CPU + CUDA) at the start of every run.
- Logs are local CSV/JSON, not W&B (the preliminary phase deferred the W&B integration to the IN-100 scale-up phase).

---

## What's next (full project, not yet implemented)

Per `CLAUDE.md`, after the preliminary phase the project scales to ImageNet-100 (the "real" experiments), adds MoCo v3 and DINOv2 teachers, the relational and SEED losses, multiple students (MobileNetV3-Small, ResNet-18), longer schedules, and the full ablations + 6-page report + 10-slide deck. The code in this repo (factory-style teacher/student/loss modules + a single config-driven trainer) is structured to scale without architectural changes — adding a new teacher means a new `src/teachers/<name>.py`, registering it in `get_teacher`, and pointing a config at it.
