"""Offline skeleton pre-computation utilities for Skeleton Recall Loss.

Reference: Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving
and Resource Efficient Segmentation of Thin Tubular Structures," ECCV 2024.
arXiv: 2404.03010
"""
from pathlib import Path

import numpy as np
from PIL import Image


def compute_tubed_skeleton(gt_float: np.ndarray, n_dilations: int = 2) -> np.ndarray:
    """Compute tubed skeleton from a float GT mask in [0, 1].

    Steps (Kirchhoff et al., ECCV 2024):
      1. Binarise the GT mask
      2. Extract 2-D medial axis via Lee's algorithm (skimage.morphology.skeletonize)
      3. Dilate n_dilations times to create a thin tube around the centreline
      4. Clip tube to GT boundary so skeleton never exceeds the vessel region
    Returns float32 array in [0, 1].
    """
    from skimage.morphology import skeletonize, dilation

    binary = gt_float > 0.5
    if not binary.any():
        return np.zeros_like(gt_float, dtype=np.float32)

    skel = skeletonize(binary).astype(np.uint8)
    for _ in range(n_dilations):
        skel = dilation(skel)
    skel = skel * binary.astype(np.uint8)
    return skel.astype(np.float32)


def build_skeleton_dir(labels_dir: str, skeletons_dir: str,
                       n_dilations: int = 2) -> int:
    """Pre-compute tubed skeletons for all PNG label files in labels_dir.

    Skeletons are saved as 8-bit PNG files in skeletons_dir with the same
    filename as the source label.  Existing files are overwritten.

    Returns the number of skeleton files written.
    """
    labels_path = Path(labels_dir)
    skel_path   = Path(skeletons_dir)
    skel_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in labels_path.iterdir()
        if f.suffix.lower() == '.png' and not f.name.startswith('.')
    )
    count = 0
    for f in files:
        gt   = np.array(Image.open(f).convert('L'), dtype=np.float32) / 255.0
        skel = compute_tubed_skeleton(gt, n_dilations)
        Image.fromarray((skel * 255).astype(np.uint8)).save(skel_path / f.name)
        count += 1
    return count
