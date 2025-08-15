# src/data.py
import os, random, numpy as np
from typing import Tuple, Dict, List, Optional
from collections import Counter

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
    expect_modes: Optional[List[str]] = None
) -> Dict[str, int]:
    mismatches = {"bad_size": 0, "bad_mode": 0}
    for cls in os.listdir(directory):
        cpath = os.path.join(directory, cls)
        if not os.path.isdir(cpath): continue
        for fname in os.listdir(cpath):
            fpath = os.path.join(cpath, fname)
            if not os.path.isfile(fpath): continue
            try:
                with Image.open(fpath) as im:
                    if expect_size and im.size != expect_size:
                        mismatches["bad_size"] += 1
                    if expect_modes and im.mode not in expect_modes:
                        mismatches["bad_mode"] += 1
            except Exception:
                mismatches["bad_size"] += 1; mismatches["bad_mode"] += 1
    return mismatches

# ---------- transforms ----------
def equalize_grayscale(img):  # grayscale + histogram equalization
    return ImageOps.equalize(img.convert("L"))

def get_gray_transforms(img_size: int = 100, augmented: bool = False):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    if not augmented:
        return transforms.Compose([transforms.Lambda(equalize_grayscale)] + base)

    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=10, shear=10),
    ]
    return transforms.Compose([transforms.Lambda(equalize_grayscale)] + aug + base)

def get_rgb_transforms(img_size: int = 100, augmented: bool = False, use_noise: bool = False):
    """
    RGB transforms. RandomErasing must run on tensors (after ToTensor).
    """
    if not augmented:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    pil_augs = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    # GaussianBlur works on PIL or tensor; keep it with PIL augs
    if use_noise:
        pil_augs += [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]

    tensor_augs = []
    if use_noise:
        # RandomErasing MUST be after ToTensor (tensor-only)
        tensor_augs += [transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))]

    return transforms.Compose(
        pil_augs
        + [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        + tensor_augs
        + [transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    )

# ---------- class weights ----------
def compute_class_weights(folder: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter([y for _, y in folder.samples])
    n_classes = len(folder.classes)
    w = torch.tensor([1.0 / counts[i] for i in range(n_classes)], dtype=torch.float32)
    return w * (n_classes / w.sum())  # normalize to mean≈1

# ---------- Windows-safe worker seeding ----------
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

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
    # global seeds
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, worker_init_fn=seed_worker)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, generator=g, worker_init_fn=seed_worker)
    test_loader  = DataLoader(ds_test,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

    class_weights = compute_class_weights(ds_train_aug)
    class_names   = ds_train_aug.classes

    if verbose:
        print(f"[INFO] Seed: {seed}")
        print(f"[INFO] Mode: {mode} (in_channels={in_ch}, use_noise={use_noise})")
        print(f"[INFO] Classes ({len(class_names)}): {class_names}")
        print(f"[INFO] Sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(ds_test)}")
        print(f"[INFO] Class weights (mean≈1): {class_weights.numpy().round(3)}")

    return train_loader, val_loader, test_loader, class_weights, class_names, in_ch
