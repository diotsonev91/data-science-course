# Fruit Classification — Data Science Course Project

This repository contains my course project for fruit image classification using **Convolutional Neural Networks (CNNs)**.  
The work explores **grayscale vs. RGB** inputs, **max vs. average pooling**, and prepares the model for **quantization** and **edge deployment**.

---

## 📦 Dataset

- **Fruits 360** (Kaggle). Images are 100×100, foldered by class and suitable for `torchvision.datasets.ImageFolder`.
- In the project we use a reduced subset of 8 classes, convert to **grayscale** for the baseline, and apply **light augmentations** on the training set only.

> You should place the dataset like this:
```
Dataset/
├─ Training/
│   ├─ ClassA/  image1.jpg ...
│   ├─ ClassB/  ...
└─ Test/
    ├─ ClassA/  ...
    ├─ ClassB/  ...
```

---

## 🗂 Repository Structure

```
data-science-course/
│
├── notebooks/
│   ├── index.ipynb                  # project overview & navigation
│   ├── 01_eda.ipynb                 # class distribution, sanity checks
│   ├── 02_preprocessing.ipynb       # transforms, loaders, class weights
│   ├── 03_train_cnn.ipynb           # baseline CNN + variants & comparison
│   ├── 04_eval_mobilenetv2.ipynb    # (optional) MobileNetV2 comparison
│   ├── 05_results.ipynb             # final metrics & visualization
│
├── src/
│   ├── data.py                      # transforms, dataloaders, class weights, EDA helpers
│   ├── models/
│   │   └── cnn_small.py             # compact CNN, max/avg pooling, 1/3 channels
│   ├── train.py                     # training loop, early stopping, test eval
│   └── viz.py                       # plots, misclassified viewer, curves
│
├── reports/
│   └── figures/                     # generated plots saved here
│
├── environment.yml                  # full Conda environment (portable version)
├── requirements.txt                 # lightweight pip install (CPU-friendly)
└── README.md                        # this file
```

---

## ⚙️ Environment Setup (choose ONE)

### Option A — Conda (full stack)
> Recommended on desktops/servers. On low‑RAM laptops Conda can be slow; use Option B instead.

```bash
conda env create -f environment.yml
conda activate ds_project
python -m ipykernel install --user --name=ds_project --display-name "Python (ds_project)"
```

**Notes**
- `environment.yml` is written to be cross‑platform (Windows/Linux).  
- If dependency solving is slow, install the faster solver:
  ```bash
  conda install -n base conda-libmamba-solver -c conda-forge -y
  conda config --set solver libmamba
  conda env create -f environment.yml
  ```
- If you see `ResolvePackageNotFound` for platform-specific builds or a `prefix:` path in a third‑party file, remove the `prefix:` line and any `=py311hXXXX_0` build IDs.

### Option B — Lightweight pip (works great on laptops)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=ds_project --display-name "Python (ds_project)"
```

This installs the **CPU** version of PyTorch by default.

### Option C — GPU (desktop with NVIDIA)
Use Conda and replace `cpuonly` with:
```yaml
channels: [pytorch, nvidia, conda-forge, defaults]
dependencies:
  - pytorch=2.5.*
  - torchvision=0.20.*
  - torchaudio=2.5.*
  - pytorch-cuda=12.4
  # ...rest of stack
```
Or with pip:
```bash
pip install torch==2.5.*+cu124 torchvision==0.20.*+cu124 torchaudio==2.5.*   -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 🚀 Quick Start

1. **Create environment** using Option A or B above.
2. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```
3. Open `notebooks/index.ipynb` and follow the links:
   - `01_eda.ipynb` — dataset sanity checks & class distribution
   - `02_preprocessing.ipynb` — transforms, loaders (train/val/test), class weights
   - `03_train_cnn.ipynb` — baseline CNN (grayscale), misclassifications, variants (avg pooling, RGB)
   - `05_results.ipynb` — comparison plots via `results_summary`

> Figures are saved automatically under `reports/figures/`.

---

## 🧪 Reproducibility

- We seed all randomness (`numpy`, `torch`, `DataLoader` generators, worker seeds).  
- On GPU, some ops (e.g., `AdaptiveAvgPool2d` backward) can be non‑deterministic; our “adaptive” variant uses deterministic `AvgPool2d` instead for exact reproducibility.
- For Windows, `num_workers=0` avoids spawn/pickling issues; increase if needed.

Minimal example in notebooks:
```python
from data import make_dataloaders
train_loader, val_loader, test_loader, class_weights, class_names, in_ch = make_dataloaders(
    "Dataset/Training", "Dataset/Test",
    batch_size=32, val_split=0.2, img_size=100,
    num_workers=0, seed=42, mode="grayscale", verbose=True
)
```

---

## 🧱 Experiments Overview

- **Baseline (Grayscale, MaxPool)**: compact CNN, 1 input channel, class‑weighted loss.  
- **Pooling Variant (AvgPool)**: deterministic average pooling vs max pooling.  
- **RGB Variants**: with/without light noise (blur + erasing); `in_channels=3`.  
- **Misclassified Viewer**: quick inspection of typical errors.  
- **Results Summary**: each experiment logs curves & test accuracy to `results_summary` and saves a plot.

Saved model weights (e.g., `fruit_cnn_baseline_with_val.pth`) are written to the project root unless changed.

---

## 🛠 Troubleshooting

- **Conda solve is slow / gets “Killed”**  
  Use the faster solver or micromamba:
  ```bash
  conda install -n base conda-libmamba-solver -c conda-forge -y
  conda config --set solver libmamba
  # or: micromamba create -n ds_project -f environment.yml -y
  ```

- **Bad channel URL like `https//conda.anaconda.org/...`**  
  Fix `.condarc`:
  ```bash
  conda config --remove-key channels
  conda config --add channels pytorch
  conda config --add channels conda-forge
  conda config --add channels defaults
  conda clean -a -y
  ```

- **`environment.yml` has `prefix:` or Windows‑only build hashes**  
  Delete `prefix:` and avoid build pins; or export cleanly next time:
  ```bash
  conda env export --from-history > environment.yml
  # or
  conda env export --no-builds > environment.yml
  ```

- **Restore the committed environment file**  
  ```bash
  git restore environment.yml
  # if staged:
  git restore --staged environment.yml && git restore environment.yml
  ```

- **Jupyter uses the wrong interpreter**  
  Re‑register the kernel **from the active env**:
  ```bash
  python -m ipykernel install --user --name=ds_project --display-name "Python (ds_project)"
  ```

---

## 👩‍🏫 Notes for the Instructor

- If Conda is problematic on your machine, please use `requirements.txt`:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Project targets **Python 3.11**, **PyTorch 2.5**, and a standard DS stack.  
- Reproducibility is enforced with fixed seeds; figures are saved to `reports/figures/`.

---

## 📄 License

MIT — feel free to use and adapt for educational purposes.
