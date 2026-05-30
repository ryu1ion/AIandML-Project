# Experiment Summary Table

All runs use the same setting: CIFAR-100, MobileNetV2 ← DINO ViT-S/16,
single-GPU bs=256, AdamW lr=1e-3, cosine, bf16, seed=42, 100 epochs unless
noted. Evaluation: linear probe (SGD lr=0.1 cosine 100 ep), kNN (k=20),
5-shot STL-10 (5 seeds), BN stats recalibrated on probe-train.

| Method                                                | LP    | kNN   | STL-10        | Δ LP vs base |
|-------------------------------------------------------|-------|-------|---------------|--------------|
| **Base** (re-run, our trainer)                        | 74.50 | 68.09 | 42.00 ± 3.29  | —            |
| **Ours** = base + 0.1·L_local-structural + patch_proj  | **74.78** | **68.59** | **45.03 ± 2.95** | **+0.28** |
| ours_iterA (λ_local=0.05)                              | 74.71 | 67.88 | 40.70 ± 3.05  | +0.21        |
| ours_exp4_cross_only (base + 0.5·L_cross)              | 74.56 | 68.18 | 43.84 ± 2.67  | +0.06        |
| paperkd + SP-KD (Tung & Mori, 2019)                    | 74.59 | 68.06 | 42.87 ± 2.26  | +0.09        |
| paperkd + RKD-D (Park et al., 2019)                    | 74.53 | 68.57 | 42.93 ± 3.68  | +0.03        |
| paperkd + SP-KD + RKD-D                                | 74.31 | 67.91 | 43.77 ± 3.52  | −0.19        |
| ours_exp3_global_only                                  | 74.39 | 68.04 | 44.17 ± 3.36  | −0.11        |
| ours_exp1_full (base + 0.1·local + 0.5·global + 0.5·cross) | 74.35 | 67.53 | 40.96 ± 3.05 | −0.15  |
| ours_iter3b (local-only at iter2 settings)             | 74.33 | 67.27 | 41.45 ± 3.32  | −0.17        |
| ours_iter3a (local + global)                           | 74.04 | 67.32 | 42.64 ± 3.50  | −0.46        |
| ours_iter2 (3 losses + patch_proj)                     | 73.85 | 67.10 | 41.98 ± 3.22  | −0.65        |
| ours_iterB (3 losses, KL relation)                     | 73.02 | 66.57 | 42.42 ± 2.59  | −1.48        |
| ours_iter1 (lower λ, 3 losses, no patch_proj)          | 68.90 | 66.73 | 41.37 ± 2.64  | −5.60        |
| **ours_distill (3 days ago, originally proposed)**     | **68.28** | 65.88 | 41.47 ± 1.99 | **−6.22** |

Recipe-only changes (not part of the proposed method, excluded by the
≥2-loss-side modifications criterion):

| Method                                  | LP    | kNN   | STL-10        | Δ LP vs base |
|-----------------------------------------|-------|-------|---------------|--------------|
| Base + warmup=10 + 150 epochs           | 74.87 | 69.13 | 44.77 ± 3.10  | +0.37        |

Failed experiments (recipe issues, not loss issues — listed for
transparency; not used in comparisons):

| Method                                  | LP    | Reason                              |
|-----------------------------------------|-------|-------------------------------------|
| DDP base / paperkd / DisCo runs         | ~22   | DDP + sync-BN + bf16 collapse       |
| Wider-projector variants (h=2048)       | 73.5–74.0 | Projector capacity hurts in this setup |
