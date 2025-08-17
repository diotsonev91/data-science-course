from dataclasses import dataclass
from typing import Tuple, Dict, List
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# === Your existing TrainConfig can be reused; add batch_size if you haven't ===
@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    early_stop_patience: int = 5
    ckpt_path: str = "experiments/runs/mobilenet_best.pt"
    seed: int = 42
    # optional: keep LR fixed or scale with batch size
    lr_scale_with_bs: bool = False
    ref_batch: int = 32

# ---------- Data ----------
def get_mobilenet_loaders(
    train_dir: str,
    test_dir: str,
    *,
    input_type: str = "grayscale",         # "grayscale" | "rgb"
    use_noise: bool = False,
    img_size: Tuple[int, int] = (100, 100),
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    pin_memory: bool = True
):
    if input_type not in {"grayscale", "rgb"}:
        raise ValueError("input_type must be 'grayscale' or 'rgb'.")

    if input_type == "grayscale":
        tfm = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        in_channels = 1
    else:
        if use_noise:
            tfm = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),                 # must be before RandomErasing
                transforms.RandomErasing(p=0.5),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
        else:
            tfm = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
        in_channels = 3

    full_ds = ImageFolder(train_dir, transform=tfm)
    val_size = int(val_split * len(full_ds))
    train_size = len(full_ds) - val_size
    train_set, val_set = torch.utils.data.random_split(full_ds, [train_size, val_size])

    test_set = ImageFolder(test_dir, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, full_ds, in_channels

# ---------- Model ----------
def prepare_mobilenet_v2(
    in_channels: int,
    num_classes: int,
    *,
    pooling: str = "default",              # "default" | "max" | "adaptive"
    pretrained: bool = True,
):
    """
    Returns a MobileNetV2 model adjusted for in_channels and num_classes,
    and optionally injects an early pooling layer.
    """
    # Newer torchvision versions prefer 'weights' instead of 'pretrained'
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
    except Exception:
        mobilenet = models.mobilenet_v2(pretrained=pretrained)

    # Patch first conv for grayscale if needed
    if in_channels != 3:
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Optional pooling injection after the first block
    if pooling == "max":
        pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        mobilenet.features = nn.Sequential(
            mobilenet.features[0],
            pooling_layer,
            *mobilenet.features[1:]
        )
    elif pooling == "adaptive":
        # The size here is chosen to roughly match early spatial dims on 100x100 input.
        pooling_layer = nn.AdaptiveAvgPool2d(output_size=(56, 56))
        mobilenet.features = nn.Sequential(
            mobilenet.features[0],
            pooling_layer,
            *mobilenet.features[1:]
        )
    elif pooling != "default":
        raise ValueError("Invalid pooling type. Choose 'default', 'max', or 'adaptive'.")

    # Patch classifier head
    num_ftrs = mobilenet.classifier[1].in_features
    mobilenet.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return mobilenet

# ---------- Train/Eval (use your existing utilities if you prefer) ----------
def train_one_epoch(model, loader, criterion, optimizer, device)->float:
    model.train(); run=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        run += loss.item()
    return run / max(1, len(loader))

@torch.no_grad()
def eval_acc(model, loader, device)->float:
    model.eval(); corr=tot=0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        corr += (pred==yb).sum().item()
        tot  += yb.size(0)
    return corr / max(1, tot)

def train_loop(model, train_loader, val_loader, class_weights, device, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    lr = cfg.lr * (cfg.batch_size / cfg.ref_batch) if cfg.lr_scale_with_bs else cfg.lr
    opt = optim.Adam(model.parameters(), lr=lr)

    best, wait = 0.0, 0
    hist: Dict[str,List[float]] = {"train_loss":[], "val_acc":[]}
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    for e in range(cfg.epochs):
        tl = train_one_epoch(model, train_loader, criterion, opt, device)
        va = eval_acc(model, val_loader, device)
        hist["train_loss"].append(tl); hist["val_acc"].append(va)
        print(f"Epoch {e+1}/{cfg.epochs} | loss {tl:.4f} | val_acc {va:.4f} | lr {lr:g} | bs {cfg.batch_size}")
        if va > best:
            best, wait = va, 0
            torch.save(model.state_dict(), cfg.ckpt_path)
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                print("Early stopping."); break
    return hist, best

@torch.no_grad()
def test_accuracy(model, ckpt_path, test_loader, device)->float:
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return eval_acc(model, test_loader, device)

# ---------- One-call experiment wrapper ----------
def run_mobilenet_v2_experiment(
    *,
    train_dir: str,
    test_dir: str,
    num_classes: int,
    cfg: TrainConfig,
    input_type: str = "grayscale",
    use_noise: bool = False,
    pooling: str = "default",
    experiment_name: str = "MobileNetV2",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    class_weights: torch.Tensor = None,
    val_split: float = 0.2,
):
    # 1) Data
    train_loader, val_loader, test_loader, full_ds, in_channels = get_mobilenet_loaders(
        train_dir, test_dir,
        input_type=input_type, use_noise=use_noise,
        batch_size=cfg.batch_size, val_split=val_split
    )

    # 2) Model
    model = prepare_mobilenet_v2(
        in_channels=in_channels,
        num_classes=num_classes,
        pooling=pooling,
        pretrained=True
    )

    # 3) Class weights (if None, use uniform)
    if class_weights is None:
        class_weights = torch.ones(num_classes, dtype=torch.float32)

    # 4) Train
    hist, best_val = train_loop(model, train_loader, val_loader, class_weights, device, cfg)

    # 5) Evaluate on test
    test_acc = test_accuracy(model, cfg.ckpt_path, test_loader, device)

    # 6) Save & (optionally) log
    torch.save(model.state_dict(), cfg.ckpt_path)

    # If you have your own saver/summary, you can plug it here
    # save_experiment_result(experiment_name, hist["train_loss"], hist["val_acc"], test_acc)

    print(f"[{experiment_name}] best_val={best_val:.4f} | test_acc={test_acc:.4f}")
    return test_acc, model, test_loader, hist

