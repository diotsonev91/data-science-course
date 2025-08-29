from __future__ import annotations
import os, random, numpy as np, torch, json
from pathlib import Path
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For full CUDA determinism

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions may not have this

    print(f"[INFO] Global seed set to {seed}")



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def verify_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def save_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
