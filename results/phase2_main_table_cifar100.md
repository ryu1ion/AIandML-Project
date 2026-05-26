# Phase 2 — CIFAR-100 main results

| Method | Distill from | Labels in distill? | Linear Probe | kNN | 5-shot STL-10 | Classifier |
|---|---|---|---|---|---|---|
| R1 Random init | — | — | 11.05 | 9.64 | 17.30 ± 1.10 | — |
| R2 Supervised | — | ✓ | 65.81 | 65.17 | 38.09 ± 3.80 | — |
| R3 SSL scratch | — | ✗ | 18.48 | 12.41 | 26.62 ± 2.00 | — |
| R5 Hinton KD | R_teacher | ✓ | 76.64 | 76.96 | 48.53 ± 2.91 | 76.95 |
| R6 FitNet | R_teacher | ✓ | 74.94 | 74.82 | 41.70 ± 2.50 | 75.10 |
| **R4 Label-free distill (ours)** | DINO ViT-S/16 | ✗ | 75.96 | 70.26 | 46.17 ± 3.32 | — |
