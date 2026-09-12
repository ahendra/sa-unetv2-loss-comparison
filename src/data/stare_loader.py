import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import StareConfig
from src.preprocessing import IdentityStep, PreprocessingPipeline, build_pipeline


def _pad_symmetric(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h, pad_w = max(target_h - h, 0), max(target_w - w, 0)
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    if img.ndim == 3:
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')
    return np.pad(img, ((top, bottom), (left, right)), mode='constant')


class StareDataLoader:
    """Loads and preprocesses STARE retinal vessel dataset."""

    def __init__(self, cfg: StareConfig, pipeline: Optional[PreprocessingPipeline] = None):
        self.cfg = cfg
        self.target_h = cfg.input_size[0]
        self.target_w = cfg.input_size[1]
        self._original_test_shapes: List[Tuple[int, int]] = []
        # Test pipeline: applied to raw original test images (preprocessing not pre-baked)
        self._test_pipeline = pipeline or build_pipeline(
            cfg.preprocessing_mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid
        )
        # Train/val: preprocessing is already baked into aug files at generation time
        self._train_pipeline = PreprocessingPipeline([IdentityStep()])

    # ── Public API ──────────────────────────────────────────────────────────

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load augmented training split (aug/train).

        Split is performed at the original-image level during augmentation,
        so this directory contains only data from training-set originals.
        """
        return self._load_split(self.cfg.aug_train_images, self.cfg.aug_train_labels)

    def load_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load augmented validation split (aug/validate)."""
        return self._load_split(self.cfg.aug_val_images, self.cfg.aug_val_labels)

    def load_train_skeleton(self) -> np.ndarray:
        """Load pre-computed tubed skeletons for the training split."""
        return self._load_skeleton_split(
            self.cfg.aug_train_images, self.cfg.aug_train_labels,
            self.cfg.aug_train_skeletons,
        )

    def load_validate_skeleton(self) -> np.ndarray:
        """Load pre-computed tubed skeletons for the validation split."""
        return self._load_skeleton_split(
            self.cfg.aug_val_images, self.cfg.aug_val_labels,
            self.cfg.aug_val_skeletons,
        )

    def load_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test images (padded to input_size) and labels."""
        self._original_test_shapes = []
        test_dir  = self.cfg.test_images
        label_dir = self.cfg.test_labels

        files = sorted(
            f for f in os.listdir(test_dir)
            if not f.startswith('.')
            and os.path.isfile(os.path.join(test_dir, f))
        )
        x_list, y_list = [], []

        for fname in files:
            base = os.path.splitext(fname)[0]
            label_path = None
            for ext in ('.ppm', '.png', '.gif', '.tif'):
                candidate = os.path.join(label_dir, f"{base}.ah{ext}")
                if os.path.exists(candidate):
                    label_path = candidate
                    break
            if label_path is None:
                continue

            im    = np.array(Image.open(os.path.join(test_dir, fname)).convert('RGB'))
            im    = self._test_pipeline.apply(im)
            label = np.array(Image.open(label_path).convert('L'))

            self._original_test_shapes.append(im.shape[:2])

            _, label_bin = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            label_bin = np.expand_dims(label_bin, axis=-1)

            h, w = im.shape[:2]
            c = im.shape[2] if im.ndim == 3 else 1
            if im.ndim == 2:
                im = np.expand_dims(im, axis=-1)
            padded = np.zeros((self.target_h, self.target_w, c), dtype=np.float32)
            padded[:h, :w, :] = im

            x_list.append(padded)
            y_list.append(label_bin)

        x_test = np.array(x_list, dtype=np.float32) / 255.0
        y_test = np.array(y_list, dtype=np.float32) / 255.0
        return x_test, y_test

    def restore_predictions(self, y_pred_padded: np.ndarray) -> List[np.ndarray]:
        """Restore per-image predictions to their original spatial sizes."""
        return [
            y_pred_padded[i, :h, :w, :]
            for i, (h, w) in enumerate(self._original_test_shapes)
        ]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_split(self, img_dir: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all image/label pairs from a STARE split directory.

        Label name convention: {base}.ah.png where base = filename without extension.
        Works for both augmented filenames (e.g. randomRotation0im0001.png) and
        original filenames (e.g. im0001.png).
        """
        if not os.path.isdir(img_dir):
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith('.png') and not f.startswith('.'))
        x_list, y_list = [], []
        for fname in files:
            base       = os.path.splitext(fname)[0]
            label_name = f"{base}.ah.png"
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                continue

            img   = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB'))
            img   = self._train_pipeline.apply(img)
            label = np.array(Image.open(label_path).convert('L'))

            img_pad = _pad_symmetric(img,   self.target_h, self.target_w)
            lbl_pad = _pad_symmetric(label, self.target_h, self.target_w)
            _, lbl_bin = cv2.threshold(lbl_pad.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

            x_list.append(img_pad)
            y_list.append(np.expand_dims(lbl_bin, axis=-1))

        x = np.array(x_list, dtype=np.float32) / 255.0
        y = np.array(y_list, dtype=np.float32) / 255.0
        return x, y

    def _load_skeleton_split(self, img_dir: str, label_dir: str,
                              skel_dir: str) -> np.ndarray:
        """Load skeleton PNGs aligned with _load_split ordering."""
        if not os.path.isdir(img_dir):
            return np.zeros((0,), dtype=np.float32)
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith('.png') and not f.startswith('.'))
        skel_list = []
        for fname in files:
            base       = os.path.splitext(fname)[0]
            label_name = f"{base}.ah.png"
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                continue
            skel_path = os.path.join(skel_dir, label_name)
            if os.path.exists(skel_path):
                skel = np.array(
                    Image.open(skel_path).convert('L'), dtype=np.float32
                ) / 255.0
            else:
                lbl  = np.array(Image.open(label_path).convert('L'))
                skel = np.zeros_like(lbl, dtype=np.float32)
            skel_pad = _pad_symmetric(skel, self.target_h, self.target_w)
            skel_list.append(np.expand_dims(skel_pad, axis=-1))
        return np.array(skel_list, dtype=np.float32)
