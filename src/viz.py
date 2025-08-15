import matplotlib.pyplot as plt
from typing import Dict, Optional

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
