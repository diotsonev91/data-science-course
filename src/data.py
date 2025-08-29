# src/data.py
import os, random, numpy as np
from typing import Tuple, Dict, List, Optional
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
from PIL import ImageOps, Image
from torchvision.datasets import ImageFolder


# Reuse helpers/constants from utils.py
from utils import (
    IMAGE_EXTS, is_image, verify_image, ensure_dir
)

# ---------- EDA helpers ----------

def count_images_per_class(directory: str) -> Dict[str, int]:
    """
    Robust image counter (recursive) for a single-level ImageFolder:
    directory/
      ClassA/*.jpg
      ClassB/subfolders_ok/*.png  (rglob search)
    """
    root = Path(directory)
    out: Dict[str, int] = {}
    if not root.exists():
        return out
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        n = sum(1 for p in cls_dir.rglob("*") if is_image(p))
        out[cls_dir.name] = n
    return out

def count_images_with_mapping(root: str, rename_map: Dict[str, str]) -> Dict[str, int]:
    """
    Count images under `root/*` but aggregate by canonical names via `rename_map`.
    Only folders present in rename_map are considered (e.g., 'freshapples' -> 'Apple').
    """
    root_p = Path(root)
    out = Counter()
    if not root_p.exists():
        return {}
    for d in root_p.iterdir():
        if not d.is_dir():
            continue
        raw = d.name
        if raw not in rename_map:
            continue
        canon = rename_map[raw]
        n = sum(1 for p in d.rglob("*") if is_image(p))
        out[canon] += n
    return dict(out)

def quick_image_sanity_check(
    directory: str,
    expect_size: Optional[Tuple[int, int]] = (100, 100),
    expect_modes: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Light sanity check. If expect_size=None, only verifies images can be opened.
    Returns counts of mismatches aggregated across classes.
    """
    mismatches = {"bad_size": 0, "bad_mode": 0}
    root = Path(directory)
    if not root.exists():
        return mismatches

    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.rglob("*"):
            if not is_image(p):
                continue
            try:
                with Image.open(p) as im:
                    if expect_size is not None and im.size != expect_size:
                        mismatches["bad_size"] += 1
                    if expect_modes is not None and im.mode not in expect_modes:
                        mismatches["bad_mode"] += 1
            except Exception:
                mismatches["bad_size"] += 1
                mismatches["bad_mode"] += 1
    return mismatches

# ---------- transforms (pad → resize → tensor) ----------

def equalize_grayscale(img: Image.Image) -> Image.Image:
    """Convert to L (grayscale) and histogram-equalize."""
    return ImageOps.equalize(img.convert("L"))

from torchvision.transforms import functional as F

class PadToSquare:
    """Make image square by center-cropping the longer side (no padding/bars)."""
    def __init__(self, fill=0, mode=None):
        self.fill = fill   # kept for API compatibility
        self.mode = mode
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        m = min(w, h)
        left = (w - m) // 2
        top  = (h - m) // 2
        return img.crop((left, top, left + m, top + m))


def get_gray_transforms(img_size: int = 100, augmented: bool = False):
    """
    Grayscale pipeline with center padding to square, then resize to img_size.
    Augmentations (if any) are applied BEFORE padding+resize to mimic real distortions.
    """
    pad = PadToSquare(fill=0)
    base = [
        pad,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    if not augmented:
        return transforms.Compose([
            transforms.Lambda(equalize_grayscale),
        ] + base)

    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=10, shear=10),
    ]
    return transforms.Compose([
        transforms.Lambda(equalize_grayscale),
    ] + aug + base)

def get_rgb_transforms(img_size: int = 100, augmented: bool = False, use_noise: bool = False):
    """
    RGB pipeline with center padding to square, then resize to img_size.
    RandomErasing must run on tensors (after ToTensor).
    """
    pad = PadToSquare(fill=(0, 0, 0))

    if not augmented:
        return transforms.Compose([
            pad,
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    pil_augs = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    if use_noise:
        pil_augs += [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]

    tensor_augs = []
    if use_noise:
        tensor_augs += [transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))]

    return transforms.Compose(
        pil_augs
        + [pad, transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        + tensor_augs
        + [transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    )

# ---------- class weights ----------

def compute_class_weights(folder: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter([y for _, y in folder.samples])
    n_classes = len(folder.classes)
    w = torch.tensor([1.0 / counts[i] for i in range(n_classes)], dtype=torch.float32)
    # normalize to mean ≈ 1 for stable loss scaling
    return w * (n_classes / w.sum())

# ---------- Windows-safe worker seeding ----------

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------- dataset maintenance for hybrid sources ----------

def split_imagefolder(
    src_root: str | Path,
    dst_root: str | Path,
    classes_raw_to_canon: Dict[str, str],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
    check_images: bool = True
) -> Dict:
    """
    Create train/val/test under dst_root from a flat source (e.g., Fruit-262).
    Copies files with deterministic shuffling; returns simple stats.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    random.seed(seed)
    src = Path(src_root)
    dst = ensure_dir(dst_root)

    stats = {"ratios": ratios, "seed": seed, "classes": {}}

    for raw_cls, canon_cls in classes_raw_to_canon.items():
        src_dir = src / raw_cls
        if not src_dir.is_dir():
            stats["classes"][canon_cls] = {"train": 0, "val": 0, "test": 0, "missing": True}
            continue

        files = [p for p in src_dir.rglob("*") if is_image(p)]
        if check_images:
            files = [p for p in files if verify_image(p)]

        random.shuffle(files)
        n = len(files)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])

        parts = {
            "train": files[:n_tr],
            "val":   files[n_tr:n_tr+n_va],
            "test":  files[n_tr+n_va:],
        }

        # materialize
        for split_name, lst in parts.items():
            out_dir = ensure_dir(Path(dst) / split_name / canon_cls)
            for i, f in enumerate(lst):
                dest = out_dir / f.name
                if dest.exists():  # avoid accidental overwrites
                    dest = out_dir / f"{f.stem}_{i}{f.suffix}"
                # copy (safer on Windows than symlink)
                import shutil
                shutil.copy2(f, dest)

        stats["classes"][canon_cls] = {k: len(v) for k, v in parts.items()}

    return stats

def cap_per_class(root: str | Path, cap: int, seed: int = 42):
    """
    Limit images per class directory under `root` to `cap` by keeping a random subset.
    Mutates files in place (deletes extra files). Use with caution.
    """
    if cap is None or cap <= 0:
        return
    random.seed(seed)
    root_p = Path(root)
    if not root_p.exists():
        return
    for cls_dir in root_p.iterdir():
        if not cls_dir.is_dir():
            continue
        files = [p for p in cls_dir.iterdir() if is_image(p)]
        if len(files) > cap:
            keep = set(random.sample(files, cap))
            for p in files:
                if p not in keep:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

# ---------- dataloaders (grayscale OR rgb) ----------

def make_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    img_size: int = 100,
    num_workers: int = 0,         # 0 is safest on Windows; set >0 if you want speed
    seed: int = 42,
    mode: str = "grayscale",       # "grayscale" or "rgb"
    use_noise: bool = False,       # only relevant for RGB augmented
    verbose: bool = False,
):
    # global seeds for determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    g = torch.Generator().manual_seed(seed)

    if mode == "grayscale":
        tf_train = get_gray_transforms(img_size, augmented=True)
        tf_clean = get_gray_transforms(img_size, augmented=False)
        in_ch = 1
    elif mode == "rgb":
        tf_train = get_rgb_transforms(img_size, augmented=True, use_noise=use_noise)
        tf_clean = get_rgb_transforms(img_size, augmented=False)
        in_ch = 3
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")

    ds_train_aug   = datasets.ImageFolder(train_dir, transform=tf_train)
    ds_train_clean = datasets.ImageFolder(train_dir, transform=tf_clean)
    ds_test        = datasets.ImageFolder(test_dir,  transform=tf_clean)

    n_val = int(len(ds_train_aug) * val_split)
    n_train = len(ds_train_aug) - n_val
    idx_train, idx_val = random_split(range(len(ds_train_aug)), [n_train, n_val], generator=g)

    train_set = Subset(ds_train_aug,   idx_train.indices)
    val_set   = Subset(ds_train_clean, idx_val.indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, worker_init_fn=seed_worker
    )
    val_loader   = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=g, worker_init_fn=seed_worker
    )
    test_loader  = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=g, worker_init_fn=seed_worker
    )

    class_weights = compute_class_weights(ds_train_aug)
    class_names   = ds_train_aug.classes

    if verbose:
        print(f"[INFO] Seed: {seed}")
        print(f"[INFO] Mode: {mode} (in_channels={in_ch}, use_noise={use_noise})")
        print(f"[INFO] Classes ({len(class_names)}): {class_names}")
        print(f"[INFO] Sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(ds_test)}")
        print(f"[INFO] Class weights (mean≈1): {class_weights.numpy().round(3)}")

    return train_loader, val_loader, test_loader, class_weights, class_names, in_ch


# ---- Generic canonical loader (works for grayscale or RGB) ----
from typing import List, Dict, Optional
from torchvision import datasets

def make_canonical_loader(
    root: str,
    canonical_classes: List[str],
    rename_map: Optional[Dict[str, str]] = None,   # e.g. {"freshapples":"Apple Red 1", ...}
    img_size: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    augmented: bool = False,        # set True only if you'll TRAIN on this source
    shuffle: bool = False,          # usually False for eval
    mode: str = "grayscale",        # "grayscale" or "rgb"
    use_noise: bool = False,        # RGB-only extra aug (GaussianBlur/RandomErasing)
):
    # pick transform by mode; both paths already do Pad->Resize(100x100)->ToTensor->Normalize
    if mode == "grayscale":
        tf = get_gray_transforms(img_size=img_size, augmented=augmented)
    elif mode == "rgb":
        tf = get_rgb_transforms(img_size=img_size, augmented=augmented, use_noise=use_noise)
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")

    ds = datasets.ImageFolder(root, transform=tf)

    # map raw folder names -> canonical class indices
    idx2raw = {i: n for n, i in ds.class_to_idx.items()}
    canon_to_idx = {n: i for i, n in enumerate(canonical_classes)}

    kept = []
    for path, raw_idx in ds.samples:
        raw_name = idx2raw[raw_idx]
        canon_name = rename_map.get(raw_name, raw_name) if rename_map else raw_name
        if canon_name not in canon_to_idx:
            continue
        kept.append((path, canon_to_idx[canon_name]))

    ds.samples = kept
    ds.targets = [y for _, y in kept]
    ds.classes = canonical_classes  # keep canonical order for consistency

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, ds


# ---- Convenience wrapper: builds Fresh-only + Fruit-262 loaders in one shot ----
import os

def build_domain_loaders(
    canonical_classes: list,
    fresh_root: str,
    fresh_map: dict,
    f262_root: str,
    f262_map: dict,
    *,
    mode: str = "grayscale",        # "grayscale" or "rgb"
    batch_size: int = 32,
    img_size: int = 100,
    num_workers: int = 0,
    fresh_aug: bool = False,        # set True only if training on Fresh
    f262_train_aug: bool = False,   # set True only if training on Fruit-262
    shuffle_fresh: bool = False,
    shuffle_f262_train: bool = False,
):
    fresh_eval_loader, fresh_ds = make_canonical_loader(
        root=fresh_root,
        canonical_classes=canonical_classes,
        rename_map=fresh_map,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        augmented=fresh_aug, shuffle=shuffle_fresh, mode=mode
    )

    f262_train_loader, f262_train_ds = make_canonical_loader(
        root=os.path.join(f262_root, "train"),
        canonical_classes=canonical_classes,
        rename_map=f262_map,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        augmented=f262_train_aug, shuffle=shuffle_f262_train, mode=mode
    )
    f262_val_loader, f262_val_ds = make_canonical_loader(
        root=os.path.join(f262_root, "val"),
        canonical_classes=canonical_classes,
        rename_map=f262_map,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        augmented=False, shuffle=False, mode=mode
    )
    f262_test_loader, f262_test_ds = make_canonical_loader(
        root=os.path.join(f262_root, "test"),
        canonical_classes=canonical_classes,
        rename_map=f262_map,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        augmented=False, shuffle=False, mode=mode
    )

    return {
        "fresh_eval": fresh_eval_loader,
        "f262_train": f262_train_loader,
        "f262_val":   f262_val_loader,
        "f262_test":  f262_test_loader,
        "_fresh_ds": fresh_ds,
        "_f262_train_ds": f262_train_ds,
        "_f262_val_ds":   f262_val_ds,
        "_f262_test_ds":  f262_test_ds,
    }
