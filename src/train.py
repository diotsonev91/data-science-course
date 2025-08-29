# train.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np

@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    early_stop_patience: int = 4
    ckpt_path: str = "experiments/runs/baseline_best.pt"
    seed: int = 42
    monitor: str = "macro_f1"   # "macro_f1" or "val_acc"
    reduce_on_plateau: bool = True
    plateau_patience: int = 2
    plateau_factor: float = 0.5

def _macro_f1(pred: np.ndarray, true: np.ndarray, num_classes: int) -> Tuple[float, List[float]]:
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s)), f1s

@torch.no_grad()
def eval_metrics(model, loader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    all_y, all_p = [], []
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
        all_y.append(yb.cpu().numpy()); all_p.append(pred.cpu().numpy())
    if total == 0:
        return {"acc": 0.0, "macro_f1": 0.0}
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    macro, _ = _macro_f1(p, y, num_classes)
    acc = correct / total
    return {"acc": float(acc), "macro_f1": float(macro)}

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train(); run = 0.0; n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        run += loss.item() * yb.size(0); n += yb.size(0)
    return run / max(1, n)

def train_loop(model, train_loader, val_loader, class_weights, device, cfg: TrainConfig, num_classes: int):
    torch.manual_seed(cfg.seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg.plateau_factor, patience=cfg.plateau_patience
    ) if cfg.reduce_on_plateau else None

    best, wait = -1.0, 0
    hist: Dict[str, List[float]] = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    for e in range(cfg.epochs):
        tl = train_one_epoch(model, train_loader, criterion, opt, device)
        val_m = eval_metrics(model, val_loader, device, num_classes)
        score = val_m["macro_f1"] if cfg.monitor == "macro_f1" else val_m["acc"]
        if sch is not None: sch.step(score)

        hist["train_loss"].append(tl)
        hist["val_acc"].append(val_m["acc"])
        hist["val_macro_f1"].append(val_m["macro_f1"])

        print(f"Epoch {e+1:02d}/{cfg.epochs} | loss {tl:.4f} | "
              f"val_acc {val_m['acc']:.3f} | val_macroF1 {val_m['macro_f1']:.3f}")

        if score > best:
            best, wait = score, 0
            torch.save(model.state_dict(), cfg.ckpt_path)
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                print("Early stopping."); break
    return hist, best

@torch.no_grad()
def test_metrics(model, ckpt_path, loader, device, num_classes: int) -> Dict[str, float]:
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return eval_metrics(model, loader, device, num_classes)
