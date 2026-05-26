# Training Pipeline Specification (Phase 2)

This document is the reviewable, reproducible specification of the label-free
distillation pipeline at **ImageNet-100** scale. It addresses the professor's
request for full pipeline detail: data, forward pass, loss math, optimizer per
method, and evaluation protocols. It complements `CLAUDE.md` (project plan) and
`PHASE2.md` (this phase's spec); where this document gives concrete numbers,
they are the authoritative values used by `scripts/train.py`.

Canonical code references are given inline so the spec and the implementation
cannot silently diverge.

---

## 1. Data pipeline

### 1.1 Dataset and class list

- **Dataset:** ImageNet-100 — the standard 100-class subset introduced by
  **CMC** and reused by **MoCo** and most label-free-distillation work. The 100
  WNIDs and names are taken verbatim, in order, from
  `https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt`.
- **Source:** the HuggingFace dataset `clane9/imagenet-100` (parquet shards),
  whose `scripts/classes.py` cites the same CMC file. Download with
  `python scripts/download_imagenet100.py --out data/imagenet100`.
- **Canonical list in repo:** `src/data/imagenet100.py` → `IMAGENET100_CLASSES`
  (an `OrderedDict[WNID -> name]`; integer label `i` == the i-th entry). The
  full 100-class list with WNIDs is reproduced in [Appendix A](#appendix-a-imagenet-100-class-list-cmcmoco).
- **Ordering guarantee:** `verify_class_list()` parses the dataset's own README
  `dataset_info` and asserts it matches the embedded CMC list index-for-index.
  `scripts/download_imagenet100.py` runs this check after every download, so a
  silent re-ordering (which would corrupt every label-dependent metric) fails
  loudly.

### 1.2 Train / val split

Standard ImageNet train/val splits **restricted to the 100 classes**:

| Split | Images | Per class | Role |
|-------|--------|-----------|------|
| `train` | 126,689 | ~1,300 (variable) | distillation / SSL / supervised training; linear-probe & kNN train features |
| `validation` | 5,000 | 50 (exact) | linear-probe & kNN evaluation |

ImageNet-100 has no separate test split; the 50/class `validation` set is the
standard ImageNet validation restricted to the 100 classes and is used as the
held-out evaluation set. No test labels are used for tuning (CLAUDE.md §What
NOT to do).

### 1.3 Image resolution

**224×224, native.** Unlike the preliminary CIFAR-100 phase (which upsampled
32×32 → 224 and is kept bit-for-bit reproducible via the separate
`make_dino_view_transform` / `make_supervised_train_transform` /
`make_eval_transform` builders), IN-100 images are full resolution, so no
upsampling is needed. The IN-100 builders below are **new** and live alongside
the CIFAR ones in `src/data/augmentations.py`.

### 1.4 Two-view augmentation — distillation (R4, R5, R9–R11) and SimSiam (R3)

Code: `make_in100_two_view_transform` →
`AsymmetricTwoViewTransform(view@blur_p=0.5, view@blur_p=0.1)`,
each view built by `make_in100_view_transform`.

Per view, in order:

1. `RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=bicubic)`
2. `RandomHorizontalFlip(p=0.5)`
3. `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)` applied with `p=0.8`
4. `RandomGrayscale(p=0.2)`
5. `GaussianBlur(kernel=23, sigma~U(0.1, 2.0))` applied with **`p=0.5` for view 1, `p=0.1` for view 2** (DINO asymmetric blur)
6. `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` (ImageNet)

Both views go through teacher and student; the distillation loss is computed
on each view and averaged (see §2).

### 1.5 Single-view augmentation — supervised (R2, R6) and labeled-KD (R7, R8)

Code: `make_in100_supervised_train_transform`.

1. `RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=bicubic)`
2. `RandomHorizontalFlip(p=0.5)`
3. `Normalize(ImageNet mean/std)`

### 1.6 Eval augmentation (all linear-probe / kNN / feature extraction)

Code: `make_in100_eval_transform`.

1. `Resize(256, interpolation=bicubic)`
2. `CenterCrop(224)`
3. `Normalize(ImageNet mean/std)`

---

## 2. Forward pass

```
                 ┌─────────────────────────────────────────────┐
   image x ──┬──>│ two-view aug (§1.4)  ──> v1, v2               │
             │   └─────────────────────────────────────────────┘
             │
   v1, v2 ───┼──> frozen teacher  (eval, torch.no_grad, bf16)
             │        └─> z_t1, z_t2   (384-d ViT-S/16 & DINOv2 ViT-S/14;
             │                          2048-d MoCo v3 R-50)
             │
   v1, v2 ───┴──> student backbone (MobileNetV2, timm, num_classes=0)
                      └─> h_s1, h_s2   (1280-d)
                          └─> projection head MLP (§3.6)
                              └─> z_s1, z_s2   (matched to teacher dim)

   loss = ( L(z_s1, z_t1) + L(z_s2, z_t2) ) / 2

   backward through student + projection head ONLY
   (teacher params frozen; teacher is never in the autograd graph)
```

- The teacher is permanently in eval mode (`DinoTeacher.train()` is a no-op)
  and wrapped in `torch.no_grad()`, so BN/attention stats are frozen and no
  gradient flows into it.
- Mixed precision: **bf16 autocast on by default** (`cfg.bf16`, disable with
  `--bf16 0`). Teacher and student forward both run under autocast; loss is
  computed in fp32 after `.float()` where numerically relevant.
- Supervised / SimSiam paths reuse the same trainer with a `task` flag
  (`supervised` → linear head + CE on single view; `simsiam` → projector +
  predictor + symmetric negative-cosine on the two views).

---

## 3. Loss functions (math)

Notation: `z_s`, `z_t` are student/teacher embeddings for one image; a batch
has `B` samples; `‖·‖` is the L2 norm; `ẑ = z/‖z‖`.

### 3.1 L2 feature matching (R4, R9–R11) — `losses/feature_matching.py:l2_normalized_mse_loss`

```
L_l2 = (1/B) Σ_i ‖ ẑ_s_i − ẑ_t_i ‖_2^2          ,  ẑ = z / ‖z‖
```

### 3.2 Cosine (loss ablation) — `losses/feature_matching.py:cosine_distance_loss`

```
L_cos = (1/B) Σ_i ( 1 − cos(z_s_i, z_t_i) )
```

Equivalence: for unit-norm inputs, `‖ẑ_s − ẑ_t‖^2 = 2 − 2·cos(ẑ_s, ẑ_t)`, so
`L_l2 = 2 · L_cos`. Verified numerically in `tests/test_losses.py`.

### 3.3 Relational (R5, new this phase) — `losses/relational.py`

Given L2-normalized teacher embeddings, the within-batch cosine-similarity
matrix is `S_t ∈ ℝ^{B×B}`, `S_t[i,j] = ẑ_t_i · ẑ_t_j`; `S_s` likewise. Then

```
L_rel = ‖ S_s − S_t ‖_F^2 / B^2
```

Optionally with off-diagonal masking (exclude `i==j`, dividing by `B^2−B`).
R5 uses the combined objective with `λ = 1.0` (one λ-sweep in {0.1, 1.0, 10.0}
on R5 only, PHASE2 §Step 5):

```
L_R5 = L_l2 + λ · L_rel
```

### 3.4 Hinton KD — labeled (R7) — `losses/hinton_kd.py`

Both teacher (R6 supervised R-50) and student carry a 100-way classification
head producing logits `z_t`, `z_s`. With temperature `τ=4`, weight `α=0.5`:

```
L_kd = τ^2 · KL( softmax(z_t/τ)  ‖  softmax(z_s/τ) )  +  α · CE(z_s, y)
```

There is **no Hinton-LF variant** (label-free Hinton): Hinton KD needs a
softmax over classes, and DINO has no classification head. Adapting it would
require either training a head on the teacher (leaks labels) or DINO's
prototype softmax (a non-trivial method change). This asymmetry is documented,
not papered over (PHASE2 note after the run table).

### 3.5 FitNet — labeled (R8) and label-free (R9) — `losses/fitnet.py`

With a learnable adapter `g_s` (1×1 conv) mapping a student mid-block feature
map `h_s_mid` to the teacher mid feature `h_t_mid`:

```
L_fitnet = ‖ g_s(h_s_mid) − h_t_mid ‖_2^2  +  β · CE(z_s, y)     ,  β = 1.0
```

- **R8 (labeled):** teacher = R6 supervised ResNet-50; match teacher **stage 3**
  (mid-network) output; CE term on labels.
- **R9 (FitNet-LF, label-free):** teacher = DINO ViT-S/16; match the output of
  **block 6** (mid-depth) of the ViT; the student-side adapter maps MobileNetV2
  **stage-3** features to the DINO block-6 feature dim; **no CE term** (β=0,
  label-free) — this is the documented Interpretation B adaptation.

### 3.6 Projection head — `projection_heads.py:MLPProjectionHead`

`Linear(student_dim, hidden) → BatchNorm1d → GELU → Linear(hidden, teacher_dim)`,
`hidden = 2 · teacher_dim` by default (e.g. 1280 → 768 → 384 for DINO ViT-S/16;
1280 → 4096 → 2048 for MoCo v3 R-50). Ablation: 2 vs 3 layers on R4 only.

---

## 4. Optimizer and schedule per method

Per-method recipes (PHASE2 §4). LR scaling is **linear in batch size**:
`lr_eff = base_lr × (global_batch / 256)`. Global batch = 256/GPU × **2 GPUs**
(DDP) = **512** unless noted (R4/R5 AdamW use `lr_scale_rule: none` — Adam is
batch-robust). Schedules are cosine-decay-to-0.

> **Compute deviation (documented).** PHASE2 §4 specifies **100 epochs** and
> its budget assumed 4-GPU DDP. This environment is constrained to **2 GPUs**
> with a 40 GPU-h cap; measured cost at 100 ep/2 GPUs is ~26 GPU-h for Step 1
> alone (R4 ≈ 6 h wall, at the per-run limit) and ~90+ GPU-h for the full
> phase — infeasible. By explicit decision, **all runs use 50 epochs**
> (identical across every run, so comparisons stay fair). The table below
> still lists the PHASE2 "100 ep" recipe for reference; the *as-run* value is
> 50 ep, recorded in every `configs/experiment/phase2/*.yaml`.

| Run | Method | Optimizer | Base LR | Schedule | Weight decay | Notes |
|-----|--------|-----------|---------|----------|--------------|-------|
| R2 | Supervised | SGD (mom 0.9) | 0.1 × batch/256 | cosine, 100 ep | 1e-4 | 5-ep warmup |
| R3 | SimSiam | SGD (mom 0.9) | 0.05 × batch/256 | cosine, 100 ep | 1e-4 | predictor LR fixed (not scaled) |
| R4 | L2 distill | AdamW | 1e-3 | cosine, 100 ep | 1e-4 | 5-ep warmup |
| R5 | Relational distill | AdamW | 1e-3 | cosine, 100 ep | 1e-4 | same as R4; λ=1.0 |
| R6 | Supervised teacher (R-50) | SGD (mom 0.9) | 0.1 × batch/256 | cosine, 100 ep | 1e-4 | standard ImageNet recipe; 5-ep warmup |
| R7 | Hinton KD | SGD (mom 0.9) | 0.1 × batch/256 | cosine, 100 ep | 1e-4 | needs R6 teacher first |
| R8 | FitNet (labeled) | SGD (mom 0.9) | 0.1 × batch/256 | cosine, 100 ep | 1e-4 | needs R6 teacher |
| R9 | FitNet-LF | AdamW | 1e-3 | cosine, 100 ep | 1e-4 | DINO mid features, no CE |
| R10 | L2 distill, MoCo v3 R-50 | AdamW | 1e-3 | cosine, 100 ep | 1e-4 | proj out dim 2048 |
| R11 | L2 distill, DINOv2 ViT-S/14 | AdamW | 1e-3 | cosine, 100 ep | 1e-4 | proj out dim 384 |

R1 (random-init) needs no training. Common (as run): **50 epochs**, bf16,
seed 42, 2-GPU DDP, SyncBatchNorm on, checkpoints at epoch 30 + final
(epoch 50). Any deviation from a method's published defaults is recorded in
that run's committed config under `configs/experiment/phase2/`.

---

## 5. Evaluation protocols

Implemented in `src/evaluator.py` (built in Phase 1, reused unchanged). All
three are run on every checkpoint's **backbone features** (post-backbone,
pre-projection-head), extracted with the §1.6 eval transform.

### 5.1 Linear probe (IN-100)

- Freeze backbone; train a single `Linear(feat_dim, 100)` on `train` features.
- **SGD, lr=0.1, momentum=0.9, weight_decay=0**, cosine decay over **100
  epochs**, batch **256**.
- Report: top-1 accuracy on the IN-100 `validation` split.

### 5.2 kNN (IN-100) — DINO protocol

- Extract & **L2-normalize** train/val backbone features.
- **k=20**, cosine similarity, weighted vote with weight `exp(sim/τ)`,
  **temperature τ=0.07**.
- Report: top-1 accuracy on `validation`.

### 5.3 5-shot transfer (STL-10)

- Extract backbone features for STL-10 train/test (10 classes).
- Sample **5 labeled examples per class** (50 total); fit sklearn
  `LogisticRegression` (default settings, `max_iter=1000`).
- Evaluate on the full STL-10 test split; repeat over **5 random seeds**;
  report **mean ± std**.

---

## 6. Logging & reproducibility

- **W&B** (`PHASE2.md` hard-constraint #3): project **`label-free-distill-phase2`**.
  This environment has no W&B credentials, so runs use **`wandb` offline mode**
  (logs written under `wandb/`, syncable later); the per-run W&B run dir,
  config, git commit hash, and final metrics are recorded. Local CSV/JSON
  (`log.csv`, `history.json`) is retained as backup. This offline-vs-online
  choice is the one documented deviation and is noted in the report.
- Every run: explicit seeds (Python/NumPy/Torch/CUDA), default **42**; full
  config saved next to the checkpoint; git commit hash logged.
- Single training script: all runs from `scripts/train.py` + a config in
  `configs/experiment/phase2/`. No per-experiment script forks.

---

## Appendix A. ImageNet-100 class list (CMC/MoCo)

Integer label = list index (0-based). Source:
`HobbitLong/CMC/blob/master/imagenet100.txt`; canonical copy in
`src/data/imagenet100.py:IMAGENET100_CLASSES`.

Left column = indices 0–49, right column = indices 50–99 (100 entries total).
This table is generated directly from the canonical module; if they ever
disagree, `src/data/imagenet100.py` is authoritative.

| # | WNID | Class | # | WNID | Class |
|---|------|-------|---|------|-------|
| 0 | n02869837 | bonnet, poke bonnet | 50 | n02259212 | leafhopper |
| 1 | n01749939 | green mamba | 51 | n07715103 | cauliflower |
| 2 | n02488291 | langur | 52 | n03947888 | pirate, pirate ship |
| 3 | n02107142 | Doberman, Doberman pinscher | 53 | n04026417 | purse |
| 4 | n13037406 | gyromitra | 54 | n02326432 | hare |
| 5 | n02091831 | Saluki, gazelle hound | 55 | n03637318 | lampshade, lamp shade |
| 6 | n04517823 | vacuum, vacuum cleaner | 56 | n01980166 | fiddler crab |
| 7 | n04589890 | window screen | 57 | n02113799 | standard poodle |
| 8 | n03062245 | cocktail shaker | 58 | n02086240 | Shih-Tzu |
| 9 | n01773797 | garden spider, Aranea diademata | 59 | n03903868 | pedestal, plinth, footstall |
| 10 | n01735189 | garter snake, grass snake | 60 | n02483362 | gibbon, Hylobates lar |
| 11 | n07831146 | carbonara | 61 | n04127249 | safety pin |
| 12 | n07753275 | pineapple, ananas | 62 | n02089973 | English foxhound |
| 13 | n03085013 | computer keyboard, keypad | 63 | n03017168 | chime, bell, gong |
| 14 | n04485082 | tripod | 64 | n02093428 | American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier |
| 15 | n02105505 | komondor | 65 | n02804414 | bassinet |
| 16 | n01983481 | American lobster, Northern lobster, Maine lobster, Homarus americanus | 66 | n02396427 | wild boar, boar, Sus scrofa |
| 17 | n02788148 | bannister, banister, balustrade, balusters, handrail | 67 | n04418357 | theater curtain, theatre curtain |
| 18 | n03530642 | honeycomb | 68 | n02172182 | dung beetle |
| 19 | n04435653 | tile roof | 69 | n01729322 | hognose snake, puff adder, sand viper |
| 20 | n02086910 | papillon | 70 | n02113978 | Mexican hairless |
| 21 | n02859443 | boathouse | 71 | n03787032 | mortarboard |
| 22 | n13040303 | stinkhorn, carrion fungus | 72 | n02089867 | Walker hound, Walker foxhound |
| 23 | n03594734 | jean, blue jean, denim | 73 | n02119022 | red fox, Vulpes vulpes |
| 24 | n02085620 | Chihuahua | 74 | n03777754 | modem |
| 25 | n02099849 | Chesapeake Bay retriever | 75 | n04238763 | slide rule, slipstick |
| 26 | n01558993 | robin, American robin, Turdus migratorius | 76 | n02231487 | walking stick, walkingstick, stick insect |
| 27 | n04493381 | tub, vat | 77 | n03032252 | cinema, movie theater, movie theatre, movie house, picture palace |
| 28 | n02109047 | Great Dane | 78 | n02138441 | meerkat, mierkat |
| 29 | n04111531 | rotisserie | 79 | n02104029 | kuvasz |
| 30 | n02877765 | bottlecap | 80 | n03837869 | obelisk |
| 31 | n04429376 | throne | 81 | n03494278 | harmonica, mouth organ, harp, mouth harp |
| 32 | n02009229 | little blue heron, Egretta caerulea | 82 | n04136333 | sarong |
| 33 | n01978455 | rock crab, Cancer irroratus | 83 | n03794056 | mousetrap |
| 34 | n02106550 | Rottweiler | 84 | n03492542 | hard disc, hard disk, fixed disk |
| 35 | n01820546 | lorikeet | 85 | n02018207 | American coot, marsh hen, mud hen, water hen, Fulica americana |
| 36 | n01692333 | Gila monster, Heloderma suspectum | 86 | n04067472 | reel |
| 37 | n07714571 | head cabbage | 87 | n03930630 | pickup, pickup truck |
| 38 | n02974003 | car wheel | 88 | n03584829 | iron, smoothing iron |
| 39 | n02114855 | coyote, prairie wolf, brush wolf, Canis latrans | 89 | n02123045 | tabby, tabby cat |
| 40 | n03785016 | moped | 90 | n04229816 | ski mask |
| 41 | n03764736 | milk can | 91 | n02100583 | vizsla, Hungarian pointer |
| 42 | n03775546 | mixing bowl | 92 | n03642806 | laptop, laptop computer |
| 43 | n02087046 | toy terrier | 93 | n04336792 | stretcher |
| 44 | n07836838 | chocolate sauce, chocolate syrup | 94 | n03259280 | Dutch oven |
| 45 | n04099969 | rocking chair, rocker | 95 | n02116738 | African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus |
| 46 | n04592741 | wing | 96 | n02108089 | boxer |
| 47 | n03891251 | park bench | 97 | n03424325 | gasmask, respirator, gas helmet |
| 48 | n02701002 | ambulance | 98 | n01855672 | goose |
| 49 | n03379051 | football helmet | 99 | n02090622 | borzoi, Russian wolfhound |
