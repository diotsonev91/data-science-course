from dataclasses import dataclass
from typing import Dict, List
import os, torch, torch.nn as nn, torch.optim as optim

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    early_stop_patience: int = 5
    ckpt_path: str = "experiments/runs/baseline_best.pt"
    seed: int = 42

def train_one_epoch(model, loader, criterion, optimizer, device)->float:
    model.train(); run=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        run += loss.item()
    return run/max(1,len(loader))

@torch.no_grad()
def eval_acc(model, loader, device)->float:
    model.eval(); corr=tot=0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        corr += (pred==yb).sum().item()
        tot  += yb.size(0)
    return corr/max(1,tot)

def train_loop(model, train_loader, val_loader, class_weights, device, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    best, wait = 0.0, 0
    hist: Dict[str,List[float]] = {"train_loss":[], "val_acc":[]}
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    for e in range(cfg.epochs):
        tl = train_one_epoch(model, train_loader, criterion, opt, device)
        va = eval_acc(model, val_loader, device)
        hist["train_loss"].append(tl); hist["val_acc"].append(va)
        print(f"Epoch {e+1}/{cfg.epochs} | loss {tl:.4f} | val_acc {va:.4f}")
        if va>best:
            best, wait = va, 0
            torch.save(model.state_dict(), cfg.ckpt_path)
        else:
            wait += 1
            if wait>=cfg.early_stop_patience:
                print("Early stopping."); break
    return hist, best

@torch.no_grad()
def test_accuracy(model, ckpt_path, test_loader, device)->float:
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return eval_acc(model, test_loader, device)
