# Phase 2 Step 1 — R1-R4 on ImageNet-100 (with BN-recalibrated eval)

Eval: IN-100 linear probe (SGD 0.1/cos/100ep/bs256, frozen features); IN-100 kNN (k=20, cosine, T=0.07, DINO protocol); STL-10 5-shot logistic regression, 5 seeds. All extractions use a uniform BN recalibration (200 batches of IN-100 train, eval transform) to correct the DDP+bf16 BN running-stat collapse.

| Run | Method | LP IN-100 (%) | kNN IN-100 (%) | STL-10 5-shot (%) |
|-----|--------|---------------|----------------|-------------------|
| R1 | Random-init MNv2 (no training) | 4.38 | 2.38 | 14.89 ± 0.77 |
| R2 | Supervised (v2: SGD eff lr 0.05, 80ep, label smoothing 0.1) | 12.98 | 9.88 | 20.66 ± 1.59 |
| R3 | SimSiam SSL from scratch (v2: SGD eff lr 0.05, 80ep, no teacher) | 7.70 | 4.70 | 20.21 ± 0.24 |
| R4 | Label-free distill from DINO ViT-S/16 (v2: AdamW 1e-3, milder aug, 80ep) | pending | pending | pending |
