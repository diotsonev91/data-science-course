# src/data.py
import os
from typing import Tuple, Dict, List, Optional
from collections import Counter
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
from PIL import ImageOps, Image

# ---------- EDA helpers ----------

def count_images_per_class(directory: str) -> Dict[str, int]:
    return {
        cls: len(os.listdir(os.path.join(directory, cls)))
        for cls in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, cls))
    }

def quick_image_sanity_check(
    directory: str,
    expect_size: Optional[Tuple[int, int]] = (100, 100),
    expect_modes: Optional[List[str]] = None  # e.g. ["L"] or ["RGB"]
) -> Dict[str, int]:
    """Lightweight scan: counts images that DON'T match expectations."""
    mismatches = {"bad_size": 0, "bad_mode": 0}
    for cls in os.listdir(directory):
        cpath = os.path.join(directory, cls)
        if not os.path.isdir(cpath):
            continue
        for fname in os.listdir(cpath):
            fpath = os.path.join(cpath, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with Image.open(fpath) as im:
                    if expect_size and im.size != expect_size:
                        mismatches["bad_size"] += 1
                    if expect_modes and im.mode not in expect_modes:
                        mismatches["bad_mode"] += 1
            except Exception:
                mismatches["bad_size"] += 1
                mismatches["bad_mode"] += 1
    return mismatches

# ---------- transforms ----------

def equalize_grayscale(img):
    # grayscale + histogram equalization
    return ImageOps.equalize(img.convert("L"))

def get_transforms(img_size: int = 100, augmented: bool = False):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    if not augmented:
        # equalize -> resize -> to tensor -> normalize
        return transforms.Compose([transforms.Lambda(equalize_grayscale)] + base)

    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=10, shear=10),
    ]
    # equalize -> (augs) -> resize -> tensor -> normalize
    return transforms.Compose([transforms.Lambda(equalize_grayscale)] + aug + base)

# ---------- class weights ----------

def compute_class_weights(folder: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter([y for _, y in folder.samples])
    n_classes = len(folder.classes)
    weights = torch.tensor([1.0 / counts[i] for i in range(n_classes)], dtype=torch.float32)
    # normalize to mean 1.0 (nicer for LR tuning)
    return weights * (n_classes / weights.sum())

# ---------- dataloaders (deterministic) ----------

def make_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    img_size: int = 100,
    num_workers: int = 2,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Returns:
        train_loader, val_loader, test_loader, class_weights, class_names
    Deterministic split, shuffle, and worker RNGs for reproducibility.
    """
    # global seeds for reproducibility (affects torchvision augs too)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    g = torch.Generator().manual_seed(seed)

    # datasets (augmented train, clean val/test)
    ds_train_aug   = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augmented=True))
    ds_train_clean = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augmented=False))
    ds_test        = datasets.ImageFolder(test_dir,  transform=get_transforms(img_size, augmented=False))

    # split indices once (deterministic)
    n_val = int(len(ds_train_aug) * val_split)
    n_train = len(ds_train_aug) - n_val
    idx_train, idx_val = random_split(range(len(ds_train_aug)), [n_train, n_val], generator=g)

    # tie subsets to the two train datasets
    train_set = Subset(ds_train_aug, idx_train.indices)
    val_set   = Subset(ds_train_clean, idx_val.indices)

    # seed each worker so augs are reproducible
    def _worker_init_fn(worker_id):
        ws = seed + worker_id
        np.random.seed(ws)
        random.seed(ws)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, worker_init_fn=_worker_init_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=g, worker_init_fn=_worker_init_fn
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=g, worker_init_fn=_worker_init_fn
    )

    class_weights = compute_class_weights(ds_train_aug)
    class_names = ds_train_aug.classes

    if verbose:
        print(f"[INFO] Seed: {seed}")
        print(f"[INFO] Classes ({len(class_names)}): {class_names}")
        print(f"[INFO] Sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(ds_test)}")
        print(f"[INFO] Class weights (mean≈1): {class_weights.numpy()}")

    return train_loader, val_loader, test_loader, class_weights, class_names
