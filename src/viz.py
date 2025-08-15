import matplotlib.pyplot as plt
from typing import Dict, Optional
import torchvision
import numpy as np

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


