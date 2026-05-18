"""ImageNet-100 dataset (the standard CMC/MoCo 100-class subset).

Class list provenance
---------------------
The 100 WNIDs and human-readable names below are the *standard ImageNet-100*
subset introduced by CMC and reused by MoCo and most label-free-distillation
work. They are copied verbatim, in order, from:

    https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt

The HuggingFace dataset `clane9/imagenet-100` (used as the data source here)
cites the same CMC file in `scripts/classes.py`, and its integer ``label``
column is ordered identically to the list below (label ``i`` == the i-th entry
of ``IMAGENET100_CLASSES``). ``verify_class_list()`` checks this contract
against the dataset's own README so a silent re-ordering can never go unnoticed.

Data source
-----------
``clane9/imagenet-100`` is distributed as parquet shards (17 train + 1 val):
126,689 train images, 5,000 val images (50/class) — the standard ImageNet
train/val splits restricted to these 100 classes. Download with
``scripts/download_imagenet100.py`` (snapshot to ``data/imagenet100/``).

The dataset object mirrors ``src.data.cifar100.get_cifar100``:
``get_imagenet100(data_root, split, mode, image_size)`` returns a torch
``Dataset`` yielding ``(img, label)`` (supervised/eval) or ``((v1, v2), label)``
(two_view), so the existing trainer/evaluator work unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from src.data.augmentations import (
    make_eval_transform,
    make_in100_eval_transform,
    make_in100_supervised_train_transform,
    make_in100_two_view_transform,
)

# WNID -> name, in CMC `imagenet100.txt` order. Integer label == list index.
IMAGENET100_CLASSES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("n02869837", "bonnet, poke bonnet"),
        ("n01749939", "green mamba"),
        ("n02488291", "langur"),
        ("n02107142", "Doberman, Doberman pinscher"),
        ("n13037406", "gyromitra"),
        ("n02091831", "Saluki, gazelle hound"),
        ("n04517823", "vacuum, vacuum cleaner"),
        ("n04589890", "window screen"),
        ("n03062245", "cocktail shaker"),
        ("n01773797", "garden spider, Aranea diademata"),
        ("n01735189", "garter snake, grass snake"),
        ("n07831146", "carbonara"),
        ("n07753275", "pineapple, ananas"),
        ("n03085013", "computer keyboard, keypad"),
        ("n04485082", "tripod"),
        ("n02105505", "komondor"),
        ("n01983481", "American lobster, Northern lobster, Maine lobster, Homarus americanus"),
        ("n02788148", "bannister, banister, balustrade, balusters, handrail"),
        ("n03530642", "honeycomb"),
        ("n04435653", "tile roof"),
        ("n02086910", "papillon"),
        ("n02859443", "boathouse"),
        ("n13040303", "stinkhorn, carrion fungus"),
        ("n03594734", "jean, blue jean, denim"),
        ("n02085620", "Chihuahua"),
        ("n02099849", "Chesapeake Bay retriever"),
        ("n01558993", "robin, American robin, Turdus migratorius"),
        ("n04493381", "tub, vat"),
        ("n02109047", "Great Dane"),
        ("n04111531", "rotisserie"),
        ("n02877765", "bottlecap"),
        ("n04429376", "throne"),
        ("n02009229", "little blue heron, Egretta caerulea"),
        ("n01978455", "rock crab, Cancer irroratus"),
        ("n02106550", "Rottweiler"),
        ("n01820546", "lorikeet"),
        ("n01692333", "Gila monster, Heloderma suspectum"),
        ("n07714571", "head cabbage"),
        ("n02974003", "car wheel"),
        ("n02114855", "coyote, prairie wolf, brush wolf, Canis latrans"),
        ("n03785016", "moped"),
        ("n03764736", "milk can"),
        ("n03775546", "mixing bowl"),
        ("n02087046", "toy terrier"),
        ("n07836838", "chocolate sauce, chocolate syrup"),
        ("n04099969", "rocking chair, rocker"),
        ("n04592741", "wing"),
        ("n03891251", "park bench"),
        ("n02701002", "ambulance"),
        ("n03379051", "football helmet"),
        ("n02259212", "leafhopper"),
        ("n07715103", "cauliflower"),
        ("n03947888", "pirate, pirate ship"),
        ("n04026417", "purse"),
        ("n02326432", "hare"),
        ("n03637318", "lampshade, lamp shade"),
        ("n01980166", "fiddler crab"),
        ("n02113799", "standard poodle"),
        ("n02086240", "Shih-Tzu"),
        ("n03903868", "pedestal, plinth, footstall"),
        ("n02483362", "gibbon, Hylobates lar"),
        ("n04127249", "safety pin"),
        ("n02089973", "English foxhound"),
        ("n03017168", "chime, bell, gong"),
        ("n02093428", "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"),
        ("n02804414", "bassinet"),
        ("n02396427", "wild boar, boar, Sus scrofa"),
        ("n04418357", "theater curtain, theatre curtain"),
        ("n02172182", "dung beetle"),
        ("n01729322", "hognose snake, puff adder, sand viper"),
        ("n02113978", "Mexican hairless"),
        ("n03787032", "mortarboard"),
        ("n02089867", "Walker hound, Walker foxhound"),
        ("n02119022", "red fox, Vulpes vulpes"),
        ("n03777754", "modem"),
        ("n04238763", "slide rule, slipstick"),
        ("n02231487", "walking stick, walkingstick, stick insect"),
        ("n03032252", "cinema, movie theater, movie theatre, movie house, picture palace"),
        ("n02138441", "meerkat, mierkat"),
        ("n02104029", "kuvasz"),
        ("n03837869", "obelisk"),
        ("n03494278", "harmonica, mouth organ, harp, mouth harp"),
        ("n04136333", "sarong"),
        ("n03794056", "mousetrap"),
        ("n03492542", "hard disc, hard disk, fixed disk"),
        ("n02018207", "American coot, marsh hen, mud hen, water hen, Fulica americana"),
        ("n04067472", "reel"),
        ("n03930630", "pickup, pickup truck"),
        ("n03584829", "iron, smoothing iron"),
        ("n02123045", "tabby, tabby cat"),
        ("n04229816", "ski mask"),
        ("n02100583", "vizsla, Hungarian pointer"),
        ("n03642806", "laptop, laptop computer"),
        ("n04336792", "stretcher"),
        ("n03259280", "Dutch oven"),
        ("n02116738", "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus"),
        ("n02108089", "boxer"),
        ("n03424325", "gasmask, respirator, gas helmet"),
        ("n01855672", "goose"),
        ("n02090622", "borzoi, Russian wolfhound"),
    ]
)

IMAGENET100_WNIDS: list[str] = list(IMAGENET100_CLASSES.keys())
IMAGENET100_NAMES: list[str] = list(IMAGENET100_CLASSES.values())
NUM_CLASSES = 100

assert len(IMAGENET100_WNIDS) == NUM_CLASSES, "IN-100 class list must have 100 entries"
assert len(set(IMAGENET100_WNIDS)) == NUM_CLASSES, "IN-100 WNIDs must be unique"

Split = Literal["train", "validation"]
Mode = Literal["supervised", "two_view", "eval"]


def _resolve_dataset_dir(data_root: str | Path) -> Path:
    """Accept either the repo `data/` dir or a direct imagenet100 dir."""
    root = Path(data_root)
    candidates = [root, root / "imagenet100"]
    for c in candidates:
        if (c / "data").is_dir() and any((c / "data").glob("*.parquet")):
            return c
    raise FileNotFoundError(
        f"ImageNet-100 parquet not found under {root} (looked in "
        f"{[str(c / 'data') for c in candidates]}). Download it first:\n"
        f"  python scripts/download_imagenet100.py --out {root}/imagenet100"
    )


def _parquet_files(ds_dir: Path) -> dict[str, list[str]]:
    data_dir = ds_dir / "data"
    train = sorted(str(p) for p in data_dir.glob("train-*.parquet"))
    val = sorted(str(p) for p in data_dir.glob("validation-*.parquet"))
    if not train or not val:
        raise FileNotFoundError(
            f"Expected train-*.parquet and validation-*.parquet under {data_dir}; "
            f"found {len(train)} train / {len(val)} val shards."
        )
    return {"train": train, "validation": val}


def _parse_readme_class_names(readme: str) -> dict[int, str]:
    """Extract the label ClassLabel `names` map from the dataset README.

    The README begins with a YAML front-matter block (between the first two
    ``---`` lines) holding ``dataset_info.features``. Parsing it with a real
    YAML loader correctly handles folded block scalars (``>-``) that long
    class names use, which a line regex would mishandle.
    """
    import yaml

    parts = readme.split("---")
    # parts[0] is '' (file starts with '---'); parts[1] is the YAML front-matter.
    if len(parts) < 3:
        return {}
    meta = yaml.safe_load(parts[1])
    feats = (meta or {}).get("dataset_info", {}).get("features", [])
    for f in feats:
        if f.get("name") == "label":
            names = f.get("dtype", {}).get("class_label", {}).get("names", {})
            return {int(k): str(v) for k, v in names.items()}
    return {}


def verify_class_list(ds_dir: str | Path) -> None:
    """Assert the dataset's own label ordering matches the embedded CMC list.

    Parses the `class_label.names` map from the dataset README's
    `dataset_info` YAML and compares it index-for-index to
    ``IMAGENET100_NAMES``. Raises AssertionError on any mismatch (silent
    re-ordering would corrupt every label-dependent metric).
    """
    ds_dir = _resolve_dataset_dir(ds_dir)
    readme = (ds_dir / "README.md").read_text()
    names = _parse_readme_class_names(readme)
    if len(names) < NUM_CLASSES:
        raise AssertionError(
            f"Could not parse 100 class names from {ds_dir/'README.md'} "
            f"(parsed {len(names)}); cannot verify class ordering."
        )
    mismatches = [
        (i, names[i], IMAGENET100_NAMES[i])
        for i in range(NUM_CLASSES)
        if names.get(i, "").split(",")[0].strip()
        != IMAGENET100_NAMES[i].split(",")[0].strip()
    ]
    if mismatches:
        head = mismatches[:5]
        raise AssertionError(
            f"IN-100 label ordering mismatch vs dataset README "
            f"({len(mismatches)} classes differ). First few "
            f"(index, dataset, embedded-CMC): {head}"
        )


class ImageNet100(Dataset):
    """ImageNet-100 backed by `clane9/imagenet-100` local parquet shards.

    Yields ``(img, label)`` for supervised/eval modes and ``((v1, v2), label)``
    for two_view mode. Images are decoded to RGB before transform.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Split,
        mode: Mode,
        image_size: int = 224,
    ) -> None:
        import datasets as hfds

        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        if mode not in ("supervised", "two_view", "eval"):
            raise ValueError(f"Unknown mode: {mode}")

        ds_dir = _resolve_dataset_dir(data_root)
        files = _parquet_files(ds_dir)
        hfds.disable_progress_bars()
        ds = hfds.load_dataset(
            "parquet", data_files=files, split=split, keep_in_memory=False
        )
        # Parquet drops HF feature metadata; restore the Image decoder so
        # __getitem__ yields PIL images rather than {bytes, path} structs.
        ds = ds.cast_column("image", hfds.Image(decode=True))
        ds = ds.with_format(None)
        self._ds = ds
        self.split = split
        self.mode = mode
        self.image_size = image_size

        if mode == "supervised":
            self.transform = (
                make_in100_supervised_train_transform(image_size)
                if split == "train"
                else make_in100_eval_transform(image_size)
            )
        elif mode == "two_view":
            self.transform = make_in100_two_view_transform(image_size)
        else:  # eval
            self.transform = make_in100_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        rec = self._ds[idx]
        img = rec["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = int(rec["label"])
        return self.transform(img), label


def get_imagenet100(
    data_root: str | Path,
    split: Split,
    mode: Mode,
    image_size: int = 224,
) -> ImageNet100:
    """Return ImageNet-100 with a transform appropriate for `mode`.

    Mirrors ``src.data.cifar100.get_cifar100``. `split` is 'train' or
    'validation' (IN-100 has no separate test split; the 50/class validation
    set is the standard ImageNet val restricted to the 100 classes).
    """
    return ImageNet100(data_root, split=split, mode=mode, image_size=image_size)


# Re-export so callers can build the IN-100 eval transform without importing
# augmentations directly (kept symmetric with the CIFAR eval transform).
in100_eval_transform = make_in100_eval_transform
generic_eval_transform = make_eval_transform


if __name__ == "__main__":  # lightweight CLI sanity / provenance check
    import argparse

    ap = argparse.ArgumentParser(description="IN-100 loader sanity check")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    verify_class_list(args.data_root)
    print(f"[ok] class ordering matches embedded CMC list ({NUM_CLASSES} classes)")
    for split in ("train", "validation"):
        ds = get_imagenet100(args.data_root, split=split, mode="eval")
        x, y = ds[0]
        print(
            f"[ok] {split}: n={len(ds)}  sample0 img={tuple(x.shape)} "
            f"label={y} ({IMAGENET100_NAMES[y]})"
        )
    print("first 3 WNIDs:", IMAGENET100_WNIDS[:3])
    print("last 3 WNIDs :", IMAGENET100_WNIDS[-3:])
