import os, random, numpy as np, torch

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
