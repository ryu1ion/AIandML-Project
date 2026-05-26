"""Build phase2_main_table_{cifar100,in100}.md from results/phase2/*.json.

Reads:
  results/phase2/eval_{run}_{dataset}.json     (backbone-frozen suite)
  results/phase2/clf_{run}_{dataset}.json      (R5/R6 classifier-head top-1)

Writes one Markdown table per dataset, with columns:
  Method | Distill from | Labels in distill? | Linear Probe (backbone) | kNN | 5-shot STL-10 | Classifier (R5/R6)

For CIFAR-100 R1-R4, falls back to the preliminary eval_*.json under results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS_CIFAR = [
    # (display_name, run_key, distill_from, labels, eval_json_path, clf_json_path)
    ("R1 Random init", "r1_random_init", "—", "—",
     "results/eval_r1_random_init.json", None),
    ("R2 Supervised", "r2_supervised", "—", "✓",
     "results/eval_r2_supervised.json", None),
    ("R3 SSL scratch", "r3_simsiam", "—", "✗",
     "results/eval_r3_simsiam.json", None),
    ("R5 Hinton KD", "r5_hinton_cifar100", "R_teacher", "✓",
     "results/phase2/eval_r5_hinton_cifar100.json",
     "results/phase2/clf_r5_hinton_cifar100.json"),
    ("R6 FitNet", "r6_fitnet_cifar100", "R_teacher", "✓",
     "results/phase2/eval_r6_fitnet_cifar100.json",
     "results/phase2/clf_r6_fitnet_cifar100.json"),
    ("**R4 Label-free distill (ours)**", "r4_distill", "DINO ViT-S/16", "✗",
     "results/eval_r4_distill.json", None),
]

ROWS_IN100 = [
    ("R1 Random init", "r1_random_init", "—", "—",
     "results/phase2/eval_r1_random_init.json", None),
    ("R5 Hinton KD", "r5_hinton_in100_v2", "R_teacher", "✓",
     "results/phase2/eval_r5_hinton_in100_v2.json",
     "results/phase2/clf_r5_hinton_in100_v2.json"),
    ("R6 FitNet", "r6_fitnet_in100", "R_teacher", "✓",
     "results/phase2/eval_r6_fitnet_in100.json",
     "results/phase2/clf_r6_fitnet_in100.json"),
]


def _load(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _fmt_pm(mean, std) -> str:
    if mean is None: return "—"
    if std is None: return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _build_table(rows, title: str) -> str:
    out = [f"# {title}\n",
           "| Method | Distill from | Labels in distill? | "
           "Linear Probe | kNN | 5-shot STL-10 | Classifier |",
           "|---|---|---|---|---|---|---|"]
    for name, _, src, lbl, eval_p, clf_p in rows:
        d = _load(eval_p) or {}
        c = _load(clf_p) or {}
        lp = d.get("linear_probe_top1_pct") or d.get("linear_probe_cifar100_top1_pct")
        knn = d.get("knn_top1_pct") or d.get("knn_cifar100_top1_pct")
        fs_m = d.get("stl10_5shot_mean_pct")
        fs_s = d.get("stl10_5shot_std_pct")
        clf = c.get("classifier_top1_pct")
        out.append(
            f"| {name} | {src} | {lbl} | {_fmt(lp)} | {_fmt(knn)} | "
            f"{_fmt_pm(fs_m, fs_s)} | {_fmt(clf)} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cifar = _build_table(ROWS_CIFAR, "Phase 2 — CIFAR-100 main results")
    in100 = _build_table(ROWS_IN100, "Phase 2 — ImageNet-100 main results")
    (out_dir / "phase2_main_table_cifar100.md").write_text(cifar)
    (out_dir / "phase2_main_table_in100.md").write_text(in100)
    print(cifar); print(); print(in100)


if __name__ == "__main__":
    main()
