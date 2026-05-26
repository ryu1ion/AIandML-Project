# Phase 2 — ImageNet-100 main results

| Method | Distill from | Labels in distill? | Linear Probe | kNN | 5-shot STL-10 | Classifier |
|---|---|---|---|---|---|---|
| R1 Random init | — | — | 4.38 | 2.38 | 14.89 ± 0.77 | — |
| R5 Hinton KD | R_teacher | ✓ | 51.62 | 47.76 | 49.88 ± 2.05 | 47.74 |
| R6 FitNet | R_teacher | ✓ | 65.26 | 61.40 | 55.78 ± 1.79 | 63.00 |
