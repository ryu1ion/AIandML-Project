"""Render the Phase 2 Step 1 R1-R4 table from per-run eval JSONs.

Reads:
  results/phase2/eval_r1_random_init.json
  results/phase2/eval_r2_supervised_v2.json
  results/phase2/eval_r3_simsiam_v2.json
  results/phase2/eval_r4_distill_l2_v2.json

Missing files are rendered as "pending". Writes Markdown to stdout and
(if --out given) to a file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("R1", "Random-init MNv2 (no training)", "eval_r1_random_init.json"),
    ("R2", "Supervised (v2: SGD eff lr 0.05, 80ep, label smoothing 0.1)",
     "eval_r2_supervised_v2.json"),
    ("R3", "SimSiam SSL from scratch (v2: SGD eff lr 0.05, 80ep, no teacher)",
     "eval_r3_simsiam_v2.json"),
    ("R4", "Label-free distill from DINO ViT-S/16 (v2: AdamW 1e-3, milder aug, 80ep)",
     "eval_r4_distill_l2_v2.json"),
]


def fmt(d: dict | None) -> tuple[str, str, str]:
    if d is None:
        return ("pending", "pending", "pending")
    lp = f"{d['linear_probe_top1_pct']:.2f}"
    knn = f"{d['knn_top1_pct']:.2f}"
    stl = f"{d['stl10_5shot_mean_pct']:.2f} ± {d['stl10_5shot_std_pct']:.2f}"
    return (lp, knn, stl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/phase2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    lines = [
        "# Phase 2 Step 1 — R1-R4 on ImageNet-100 (with BN-recalibrated eval)",
        "",
        "Eval: IN-100 linear probe (SGD 0.1/cos/100ep/bs256, frozen features); "
        "IN-100 kNN (k=20, cosine, T=0.07, DINO protocol); "
        "STL-10 5-shot logistic regression, 5 seeds. All extractions use a "
        "uniform BN recalibration (200 batches of IN-100 train, eval transform) "
        "to correct the DDP+bf16 BN running-stat collapse.",
        "",
        "| Run | Method | LP IN-100 (%) | kNN IN-100 (%) | STL-10 5-shot (%) |",
        "|-----|--------|---------------|----------------|-------------------|",
    ]
    for rid, label, fn in ROWS:
        p = rd / fn
        d = json.loads(p.read_text()) if p.exists() else None
        lp, knn, stl = fmt(d)
        lines.append(f"| {rid} | {label} | {lp} | {knn} | {stl} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
