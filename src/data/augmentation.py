import logging
import os
import random
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile
from tqdm import tqdm

from config import RANDOM_SEED

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


# ── Filename helpers ──────────────────────────────────────────────────────────

def _aug_fname(prefix: str, idx: int, orig_fname: str) -> str:
    """
    Build augmented filename following keras_dataAug.py convention:
    prefix + idx + original_basename (always saved as .png).
    Example: ('randomRotation', 0, '21_training.tif') -> 'randomRotation021_training.png'
    """
    name, _ = os.path.splitext(orig_fname)
    return f"{prefix}{idx}{name}.png"


# ── Augmentation operations ───────────────────────────────────────────────────

class DataAugmentation:
    """Static image augmentation operations (image + label pairs)."""

    @staticmethod
    def random_rotation(image: Image.Image, label: Image.Image):
        angle = np.random.randint(1, 360)
        return image.rotate(angle, Image.BICUBIC), label.rotate(angle, Image.NEAREST)

    @staticmethod
    def random_color(image: Image.Image, label: Image.Image):
        img = ImageEnhance.Color(image).enhance(np.random.randint(0, 31) / 10.0)
        img = ImageEnhance.Brightness(img).enhance(np.random.randint(10, 21) / 10.0)
        img = ImageEnhance.Contrast(img).enhance(np.random.randint(10, 21) / 10.0)
        img = ImageEnhance.Sharpness(img).enhance(np.random.randint(0, 31) / 10.0)
        return img, label

    @staticmethod
    def random_gaussian(image: Image.Image, label: Image.Image, mean: float = 0.2, sigma: float = 0.3):
        def _gaussian_noisy(channel_flat):
            for i in range(len(channel_flat)):
                channel_flat[i] += random.gauss(mean, sigma)
            return channel_flat

        img = np.array(image)  # writable copy (np.asarray gives read-only on PIL)
        h, w = img.shape[:2]
        img[:, :, 0] = _gaussian_noisy(img[:, :, 0].flatten()).reshape([h, w])
        img[:, :, 1] = _gaussian_noisy(img[:, :, 1].flatten()).reshape([h, w])
        img[:, :, 2] = _gaussian_noisy(img[:, :, 2].flatten()).reshape([h, w])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def h_flip(image: Image.Image, label: Image.Image):
        # cv2.flip(src, 1) — flip horizontally (matches flip.py)
        img_arr = cv2.flip(np.asarray(image), 1)
        lbl_arr = cv2.flip(np.asarray(label), 1)
        return Image.fromarray(img_arr), Image.fromarray(lbl_arr)

    @staticmethod
    def v_flip(image: Image.Image, label: Image.Image):
        # cv2.flip(src, 0) — flip vertically (matches flip.py)
        img_arr = cv2.flip(np.asarray(image), 0)
        lbl_arr = cv2.flip(np.asarray(label), 0)
        return Image.fromarray(img_arr), Image.fromarray(lbl_arr)

    @staticmethod
    def hv_flip(image: Image.Image, label: Image.Image):
        # cv2.flip(src, -1) — flip both axes simultaneously (matches flip.py)
        img_arr = cv2.flip(np.asarray(image), -1)
        lbl_arr = cv2.flip(np.asarray(label), -1)
        return Image.fromarray(img_arr), Image.fromarray(lbl_arr)


# ── SA-UNetV2 paper augmentation spec ─────────────────────────────────────────
# Per-image operations: (aug_type_name, repetitions, function)
AUG_OPS: List[Tuple[str, int, Callable]] = [
    ("randomRotation", 3, DataAugmentation.random_rotation),
    ("randomColor",    3, DataAugmentation.random_color),
    ("randomGaussian", 3, DataAugmentation.random_gaussian),
    ("h",              1, DataAugmentation.h_flip),
    ("v",              1, DataAugmentation.v_flip),
    ("hv",             1, DataAugmentation.hv_flip),
]


# ── Runner ────────────────────────────────────────────────────────────────────

class RetinalAugmentationRunner:
    """
    Augments retinal vessel datasets (DRIVE / STARE) following SA-UNetV2 paper.

    Split is performed at the ORIGINAL IMAGE level before augmentation to
    prevent data leakage between training and validation sets.

    Per original image generates 12 augmented versions:
        randomRotation ×3, randomColor ×3, randomGaussian ×3,
        h-flip ×1, v-flip ×1, hv-flip ×1
    Original images are also copied → 13 files per original image.

    Example with val_ratio=0.1 and 20 DRIVE training images:
        - 2 originals reserved for validate → 2 × 13 = 26 files
        - 18 originals reserved for train   → 18 × 13 = 234 files
        (No augmented variant of a validate-original appears in train.)

    Output structure inside aug_base_dir:
        aug_base/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── validate/
            ├── images/
            └── labels/
    """

    def __init__(self):
        pass

    def run(
        self,
        src_img_dir: str,
        src_lbl_dir: str,
        aug_base_dir: str,
        label_suffix_fn: Callable[[str], str],
        preprocessing_pipeline=None,
        val_ratio: float = 0.1,
    ) -> None:
        """
        src_img_dir           : original training images directory
        src_lbl_dir           : original training labels directory
        aug_base_dir          : output base (creates train/ and validate/ inside)
        label_suffix_fn       : callable(img_fname) -> label_fname
        preprocessing_pipeline: optional PreprocessingPipeline applied to each
                                 original image BEFORE augmentation so that all
                                 saved files already contain the preprocessed
                                 content (e.g. CLAHE-enhanced RGB).
        val_ratio             : fraction of ORIGINAL images reserved for validate
                                 (applied before augmentation; default 0.1 = 10%)
        """

        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)

        aug_base  = Path(aug_base_dir)
        train_img = aug_base / "train"    / "images"
        train_lbl = aug_base / "train"    / "labels"
        val_img   = aug_base / "validate" / "images"
        val_lbl   = aug_base / "validate" / "labels"

        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Collect & split original images ───────────────────────────
        img_files = sorted([
            f for f in os.listdir(src_img_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(src_img_dir, f))
        ])

        n_orig     = len(img_files)
        n_val_orig = max(1, round(n_orig * val_ratio))
        val_orig_indices   = set(random.sample(range(n_orig), n_val_orig))
        train_orig_indices = set(range(n_orig)) - val_orig_indices

        n_per_orig = sum(r for _, r, _ in AUG_OPS) + 1   # 12 augmented + 1 original
        n_train_est = len(train_orig_indices) * n_per_orig
        n_val_est   = len(val_orig_indices)   * n_per_orig

        _pre_desc = (preprocessing_pipeline.description
                     if preprocessing_pipeline is not None else "none (raw RGB)")
        print(f"\n  Sumber gambar        : {src_img_dir}")
        print(f"  Preprocessing        : {_pre_desc}")
        print(f"  Seed                 : {RANDOM_SEED}")
        print(f"  Jumlah gambar asli   : {n_orig}")
        print(f"  Split original       : {len(train_orig_indices)} train | "
              f"{len(val_orig_indices)} validate  (split sebelum augmentasi)")
        print(f"  Augmentasi per gambar: {n_per_orig - 1} + 1 (original) = {n_per_orig}")
        print(f"  Total estimasi       : {n_train_est} train  +  {n_val_est} validate\n")

        # ── Step 2: Augment each subset directly into its output directory ─────
        def _augment_subset(orig_indices: set, out_img: Path, out_lbl: Path,
                            desc: str) -> int:
            count = 0
            for i in tqdm(sorted(orig_indices), desc=f"  {desc}", unit="gambar"):
                img_fname = img_files[i]
                lbl_fname = label_suffix_fn(img_fname)
                lbl_path  = os.path.join(src_lbl_dir, lbl_fname)

                if not os.path.exists(lbl_path):
                    logger.warning("Label tidak ditemukan: %s — dilewati.", lbl_path)
                    continue

                img = Image.open(os.path.join(src_img_dir, img_fname)).convert('RGB')
                if preprocessing_pipeline is not None:
                    img = Image.fromarray(preprocessing_pipeline.apply(np.array(img)))
                lbl = Image.open(lbl_path).convert('L')

                # Save original as PNG (normalises format)
                img.save(str(out_img / (os.path.splitext(img_fname)[0] + '.png')))
                lbl.save(str(out_lbl / (os.path.splitext(lbl_fname)[0] + '.png')))
                count += 1

                # Generate and save augmented versions
                for aug_type, reps, op in AUG_OPS:
                    for j in range(reps):
                        aug_img, aug_lbl = op(img.copy(), lbl.copy())
                        aug_img.save(str(out_img / _aug_fname(aug_type, j, img_fname)))
                        aug_lbl.save(str(out_lbl / _aug_fname(aug_type, j, lbl_fname)))
                        count += 1
            return count

        n_train_actual = _augment_subset(train_orig_indices, train_img, train_lbl,
                                         "Augmentasi train   ")
        n_val_actual   = _augment_subset(val_orig_indices,   val_img,   val_lbl,
                                         "Augmentasi validate")

        print(f"\n  Selesai!")
        print(f"  Train    → {train_img}  ({n_train_actual} file)")
        print(f"  Validate → {val_img}  ({n_val_actual} file)")
