"""Aggregate the four eval JSONs into results/preliminary_table.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"

ROWS = [
    ("R1: Random init",           "eval_r1_random_init.json"),
    ("R2: Supervised",            "eval_r2_supervised.json"),
    ("R3: SSL from scratch",      "eval_r3_simsiam.json"),
    ("R4: Label-free distill",    "eval_r4_distill.json"),
]


def main() -> None:
    lines = [
        "| Method | Linear Probe (CIFAR-100) | kNN (CIFAR-100) | 5-shot STL-10 (mean ± std) |",
        "|--------|-------------------------|-----------------|----------------------------|",
    ]
    for label, fname in ROWS:
        d = json.loads((RESULTS / fname).read_text())
        lp = d["linear_probe_cifar100_top1_pct"]
        knn = d["knn_cifar100_top1_pct"]
        fs_m = d["stl10_5shot_mean_pct"]
        fs_s = d["stl10_5shot_std_pct"]
        lines.append(f"| {label} | {lp:.2f} | {knn:.2f} | {fs_m:.2f} ± {fs_s:.2f} |")
    out = "\n".join(lines) + "\n"
    (RESULTS / "preliminary_table.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
