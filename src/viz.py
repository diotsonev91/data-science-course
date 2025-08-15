import matplotlib.pyplot as plt
from typing import Dict, Optional
import torchvision
import numpy as np
from io import BytesIO
import torch
import pandas as pd

def display_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[str] = None
):
    print(f"\n{title}\n" + "-"*len(title))
    for cls, count in sorted(class_counts.items()):
        print(f"{cls}: {count} images")

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def show_batch(images, labels=None, class_names=None, grayscale=True, n: int = 8):
    grid = torchvision.utils.make_grid(images[:n])
    img = grid / 2 + 0.5  # unnormalize
    arr = img.numpy()
    if grayscale:
        arr = arr[0]
        plt.imshow(arr, cmap='gray')
    else:
        plt.imshow(np.transpose(arr, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    if labels is not None and class_names is not None:
        for i in range(min(n, len(labels))):
            print(f"Image {i+1}: {class_names[labels[i].item()]}")



def plot_training_curves(train_losses, val_accs):
    buf = BytesIO()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(train_losses); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2); plt.plot(val_accs);    plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc")
    plt.tight_layout(); plt.savefig(buf, format="png"); plt.close(); buf.seek(0)
    return buf


@torch.no_grad()
def plot_misclassified_images(model, test_loader, class_names, device, max_images=12, save_path=None):
    """Collect and plot up to `max_images` misclassified samples."""
    model.eval()
    mis = []

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        wrong = (pred != yb).nonzero(as_tuple=True)[0]
        for i in wrong.tolist():
            mis.append((xb[i].cpu(), yb[i].item(), pred[i].item()))
            if len(mis) >= max_images:
                break
        if len(mis) >= max_images:
            break

    if not mis:
        print("No misclassified samples found.")
        return mis  # return list even if empty

    cols, rows = 4, (len(mis) + 3) // 4
    plt.figure(figsize=(4 * cols, 3 * rows))
    for idx, (img, true_i, pred_i) in enumerate(mis):
        plt.subplot(rows, cols, idx + 1)
        arr = img.squeeze().numpy()
        if arr.ndim == 2:                       # grayscale
            plt.imshow(arr, cmap='gray')
        else:                                   # RGB
            plt.imshow(arr.transpose(1, 2, 0))
        plt.title(f"T: {class_names[true_i]}\nP: {class_names[pred_i]}", fontsize=9)
        plt.axis('off')

    plt.suptitle("Misclassified Test Images", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return mis


def misclassified_to_df(mis, class_names):
    rows = []
    for img, true_i, pred_i in mis:
        rows.append({"true": class_names[true_i], "pred": class_names[pred_i]})
    return pd.DataFrame(rows)


