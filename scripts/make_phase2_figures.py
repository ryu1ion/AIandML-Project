"""Generate the three Phase-2 figures from results/phase2 + results/.

Outputs:
  results/figures/fig_kd_comparison_cifar.png
  results/figures/fig_kd_comparison_in100.png
  results/figures/fig_scale_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load(p) -> dict:
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


# (display, eval_path) -- order matches NEW_BENCH bar layout (R2, R3, R5, R6, R4)
CIFAR_METHODS = [
    ("R2 Sup", REPO / "results/eval_r2_supervised.json", "labeled"),
    ("R3 SSL", REPO / "results/eval_r3_simsiam.json", "label-free"),
    ("R5 Hinton", REPO / "results/phase2/eval_r5_hinton_cifar100.json", "labeled"),
    ("R6 FitNet", REPO / "results/phase2/eval_r6_fitnet_cifar100.json", "labeled"),
    ("R4 LF-distill (ours)", REPO / "results/eval_r4_distill.json", "label-free"),
]
IN100_METHODS = [
    ("R1 Random", REPO / "results/phase2/eval_r1_random_init.json", "label-free"),
    ("R5 Hinton", REPO / "results/phase2/eval_r5_hinton_in100.json", "labeled"),
    ("R6 FitNet", REPO / "results/phase2/eval_r6_fitnet_in100.json", "labeled"),
]

METRIC_KEYS = [
    ("Linear Probe", "linear_probe_top1_pct"),
    ("kNN", "knn_top1_pct"),
    ("5-shot STL-10", "stl10_5shot_mean_pct"),
]


def _bar(ax, methods, title):
    n = len(methods); ngroup = len(METRIC_KEYS)
    width = 0.8 / n
    x = np.arange(ngroup)
    for i, (name, p, cat) in enumerate(methods):
        d = _load(p)
        vals = [float(d.get(k, np.nan)) for _, k in METRIC_KEYS]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=name, color="tab:blue" if cat == "label-free" else "tab:orange")
    ax.set_xticks(x); ax.set_xticklabels([k for k, _ in METRIC_KEYS])
    ax.set_ylabel("top-1 (%)"); ax.set_title(title); ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    out = REPO / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar(ax, CIFAR_METHODS, "Phase 2 — CIFAR-100")
    fig.tight_layout(); fig.savefig(out / "fig_kd_comparison_cifar.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar(ax, IN100_METHODS, "Phase 2 — ImageNet-100")
    fig.tight_layout(); fig.savefig(out / "fig_kd_comparison_in100.png", dpi=160)
    plt.close(fig)
    # Scale comparison: linear probe on each method, CIFAR vs IN-100
    pairs = [
        ("R5 Hinton",
         REPO / "results/phase2/eval_r5_hinton_cifar100.json",
         REPO / "results/phase2/eval_r5_hinton_in100.json"),
        ("R6 FitNet",
         REPO / "results/phase2/eval_r6_fitnet_cifar100.json",
         REPO / "results/phase2/eval_r6_fitnet_in100.json"),
        ("R4 LF-distill",
         REPO / "results/eval_r4_distill.json",
         REPO / "results/phase2/eval_r4_distill_in100.json"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35; idx = np.arange(len(pairs))
    cifar_vals = [_load(c).get("linear_probe_top1_pct", np.nan) for _, c, _ in pairs]
    in_vals = [_load(i).get("linear_probe_top1_pct", np.nan) for _, _, i in pairs]
    ax.bar(idx - width/2, cifar_vals, width, label="CIFAR-100", color="tab:green")
    ax.bar(idx + width/2, in_vals, width, label="ImageNet-100", color="tab:purple")
    ax.set_xticks(idx); ax.set_xticklabels([n for n, _, _ in pairs])
    ax.set_ylabel("Linear probe top-1 (%)")
    ax.set_title("Source-domain linear probe — CIFAR vs IN-100")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_scale_comparison.png", dpi=160)
    plt.close(fig)
    print("Wrote 3 figures under", out)


if __name__ == "__main__":
    main()
