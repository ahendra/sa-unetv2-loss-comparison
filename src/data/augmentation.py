import logging
import os
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile
from tqdm import tqdm

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

# Validate sample counts per augmentation type (for 20 original train images → 260 total)
VALIDATE_COUNTS: Dict[str, int] = {
    "original":       2,
    "h":              2,
    "v":              2,
    "hv":             2,
    "randomColor":    6,
    "randomGaussian": 6,
    "randomRotation": 6,
}


# ── Runner ────────────────────────────────────────────────────────────────────

class RetinalAugmentationRunner:
    """
    Augments retinal vessel datasets (DRIVE / STARE) following SA-UNetV2 paper.

    Per original image generates 12 augmented versions:
        randomRotation ×3, randomColor ×3, randomGaussian ×3,
        h-flip ×1, v-flip ×1, hv-flip ×1

    Original images are also copied. Final output is split proportionally:
        - validate : 10% (2 original, 2h, 2v, 2hv, 6 randomColor, 6 randomGaussian, 6 randomRotation)
        - train    : remaining 90%

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
    ) -> None:
        """
        src_img_dir     : original training images directory
        src_lbl_dir     : original training labels directory
        aug_base_dir    : output base (creates train/ and validate/ inside)
        label_suffix_fn : callable(img_fname) -> label_fname
        """

        aug_base  = Path(aug_base_dir)
        pool_base = aug_base / "_pool"
        train_img = aug_base / "train"    / "images"
        train_lbl = aug_base / "train"    / "labels"
        val_img   = aug_base / "validate" / "images"
        val_lbl   = aug_base / "validate" / "labels"

        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        for aug_type in VALIDATE_COUNTS:
            (pool_base / aug_type / "images").mkdir(parents=True, exist_ok=True)
            (pool_base / aug_type / "labels").mkdir(parents=True, exist_ok=True)

        # ── Step 1: Generate augmented images into pool ───────────────────────
        img_files = sorted([
            f for f in os.listdir(src_img_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(src_img_dir, f))
        ])

        reps_total = sum(r for _, r, _ in AUG_OPS)
        n_orig = len(img_files)
        n_total = n_orig * (reps_total + 1)

        print(f"\n  Sumber gambar        : {src_img_dir}")
        print(f"  Jumlah gambar asli   : {n_orig}")
        print(f"  Augmentasi per gambar: {reps_total} + 1 (original) = {reps_total + 1}")
        print(f"  Total estimasi       : {n_total} gambar\n")

        # Collect (img_pool_path, lbl_pool_path) per aug type for the split
        type_pairs: Dict[str, List[Tuple[str, str]]] = {t: [] for t in VALIDATE_COUNTS}

        for img_fname in tqdm(img_files, desc="  Augmentasi", unit="gambar"):
            lbl_fname = label_suffix_fn(img_fname)
            lbl_path  = os.path.join(src_lbl_dir, lbl_fname)

            if not os.path.exists(lbl_path):
                logger.warning("Label tidak ditemukan: %s — dilewati.", lbl_path)
                continue

            img = Image.open(os.path.join(src_img_dir, img_fname)).convert('RGB')
            lbl = Image.open(lbl_path).convert('L')

            # Convert original to PNG (normalizes format to match augmented outputs)
            orig_img_fname = os.path.splitext(img_fname)[0] + '.png'
            orig_lbl_fname = os.path.splitext(lbl_fname)[0] + '.png'
            orig_img_dst = str(pool_base / "original" / "images" / orig_img_fname)
            orig_lbl_dst = str(pool_base / "original" / "labels" / orig_lbl_fname)
            img.save(orig_img_dst)
            lbl.save(orig_lbl_dst)
            type_pairs["original"].append((orig_img_dst, orig_lbl_dst))

            # Generate and save augmented versions
            for aug_type, reps, op in AUG_OPS:
                for i in range(reps):
                    aug_img, aug_lbl = op(img.copy(), lbl.copy())
                    aug_img_fname = _aug_fname(aug_type, i, img_fname)
                    aug_lbl_fname = _aug_fname(aug_type, i, lbl_fname)
                    aug_img_dst   = str(pool_base / aug_type / "images" / aug_img_fname)
                    aug_lbl_dst   = str(pool_base / aug_type / "labels" / aug_lbl_fname)
                    aug_img.save(aug_img_dst)
                    aug_lbl.save(aug_lbl_dst)
                    type_pairs[aug_type].append((aug_img_dst, aug_lbl_dst))

        # ── Step 2: Proportional random split ─────────────────────────────────
        print("\n  Membagi data secara proporsional (train / validate)...\n")
        train_pairs: List[Tuple[str, str]] = []
        val_pairs:   List[Tuple[str, str]] = []

        for aug_type, val_count in VALIDATE_COUNTS.items():
            pairs    = type_pairs[aug_type]
            n_val    = min(val_count, len(pairs))
            val_idx  = set(random.sample(range(len(pairs)), n_val))

            n_to_train = len(pairs) - n_val
            print(f"  [{aug_type:>16}]  total={len(pairs):>3}  "
                  f"validate={n_val:>2}  train={n_to_train:>3}")

            for i, pair in enumerate(pairs):
                if i in val_idx:
                    val_pairs.append(pair)
                else:
                    train_pairs.append(pair)

        print(f"\n  Train    : {len(train_pairs)} gambar")
        print(f"  Validate : {len(val_pairs)} gambar")
        print(f"  Total    : {len(train_pairs) + len(val_pairs)} gambar\n")

        # ── Step 3: Copy to final train / validate directories ────────────────
        for img_src, lbl_src in tqdm(train_pairs, desc="  Simpan train   ", unit="gambar"):
            shutil.copy2(img_src, train_img / Path(img_src).name)
            shutil.copy2(lbl_src, train_lbl / Path(lbl_src).name)

        for img_src, lbl_src in tqdm(val_pairs, desc="  Simpan validate", unit="gambar"):
            shutil.copy2(img_src, val_img / Path(img_src).name)
            shutil.copy2(lbl_src, val_lbl / Path(lbl_src).name)

        # ── Cleanup temporary pool ────────────────────────────────────────────
        shutil.rmtree(pool_base)

        print(f"\n  Selesai!")
        print(f"  Train    → {train_img}")
        print(f"  Validate → {val_img}")
