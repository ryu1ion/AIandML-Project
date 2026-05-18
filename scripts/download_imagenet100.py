"""Download the ImageNet-100 dataset (HF `clane9/imagenet-100`) and verify it.

`clane9/imagenet-100` is the standard CMC/MoCo 100-class subset distributed as
parquet shards (17 train + 1 validation, ~8.4 GB). This script snapshots only
the data + provenance files into `data/imagenet100/` and then verifies the
dataset's label ordering matches the embedded CMC class list.

Usage:
    python scripts/download_imagenet100.py --out data/imagenet100

Reproducibility note: no HF token is required (public dataset). Unauthenticated
downloads are rate-limited; set HF_TOKEN for faster transfer if available.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPO_ID = "clane9/imagenet-100"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "imagenet100"))
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {out} ...", flush=True)
    t0 = time.time()
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=["data/*.parquet", "README.md", "scripts/classes.py"],
        max_workers=args.max_workers,
    )
    dt = time.time() - t0
    n_parquet = len(list((out / "data").glob("*.parquet")))
    print(f"Done in {dt:.0f}s ({n_parquet} parquet shards).", flush=True)

    from src.data.imagenet100 import (
        NUM_CLASSES,
        get_imagenet100,
        verify_class_list,
    )

    verify_class_list(out)
    print(f"[ok] label ordering matches embedded CMC list ({NUM_CLASSES} classes)")
    for split in ("train", "validation"):
        ds = get_imagenet100(out, split=split, mode="eval")
        print(f"[ok] {split}: {len(ds)} images")


if __name__ == "__main__":
    main()
