import os
from typing import Dict, Tuple, List, Optional
from PIL import Image

def count_images_per_class(directory: str) -> Dict[str, int]:
    return {
        cls: len(os.listdir(os.path.join(directory, cls)))
        for cls in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, cls))
    }

def quick_image_sanity_check(
    directory: str,
    expect_size: Optional[Tuple[int, int]] = (100, 100),
    expect_modes: Optional[List[str]] = None  # e.g. ["L"] for grayscale, ["RGB"] for color
) -> Dict[str, int]:
    """Lightweight scan: returns counts of images that DON'T match expectations."""
    mismatches = {"bad_size": 0, "bad_mode": 0}
    for cls in os.listdir(directory):
        cpath = os.path.join(directory, cls)
        if not os.path.isdir(cpath):
            continue
        for fname in os.listdir(cpath):
            fpath = os.path.join(cpath, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with Image.open(fpath) as im:
                    if expect_size and im.size != expect_size:
                        mismatches["bad_size"] += 1
                    if expect_modes and im.mode not in expect_modes:
                        mismatches["bad_mode"] += 1
            except Exception:
                # Treat unreadable images as size/mode mismatches implicitly
                mismatches["bad_size"] += 1
                mismatches["bad_mode"] += 1
    return mismatches
