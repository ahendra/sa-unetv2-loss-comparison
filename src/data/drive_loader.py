import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import DriveConfig
from src.preprocessing import IdentityStep, PreprocessingPipeline, build_pipeline


def _read_image(path: str, mode: str = 'RGB') -> np.ndarray:
    """Read any image format (TIFF/LZW, GIF, PNG, …) via PIL."""
    with Image.open(path) as img:
        return np.array(img.convert(mode))


def _find_label(label_dir: str, stem: str, extensions=('.png', '.gif')) -> Optional[str]:
    for ext in extensions:
        candidate = os.path.join(label_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _pad_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Zero-pad image at bottom-right corner to reach target spatial size."""
    h, w = img.shape[:2]
    if img.ndim == 3:
        out = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        out[:h, :w, :] = img
    else:
        out = np.zeros((target_h, target_w), dtype=img.dtype)
        out[:h, :w] = img
    return out


class DriveDataLoader:
    """Loads and preprocesses DRIVE retinal vessel dataset."""

    def __init__(self, cfg: DriveConfig, pipeline: Optional[PreprocessingPipeline] = None):
        self.cfg = cfg
        self.target_h = cfg.input_size[0]
        self.target_w = cfg.input_size[1]
        # Test pipeline: applied to raw original test images (preprocessing not pre-baked)
        self._test_pipeline = pipeline or build_pipeline(
            cfg.preprocessing_mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid
        )
        # Train/val: preprocessing is already baked into aug files at generation time
        self._train_pipeline = PreprocessingPipeline([IdentityStep()])

    # ── Public API ──────────────────────────────────────────────────────────

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load augmented training split (aug/train)."""
        return self._load_split(self.cfg.aug_train_images, self.cfg.aug_train_labels)

    def load_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load augmented validation split (aug/validate)."""
        return self._load_split(self.cfg.aug_val_images, self.cfg.aug_val_labels)

    def load_test(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load test images (padded), labels and optional FOV masks at original size.
        Returns (x_test_padded, y_test_orig, masks_orig).
        """
        test_dir  = self.cfg.test_images
        label_dir = self.cfg.test_labels
        mask_dir  = self.cfg.test_masks

        files = sorted(f for f in os.listdir(test_dir) if not f.startswith('.'))
        x_list, y_list, m_list = [], [], []

        for fname in files:
            stem = fname.split('.')[0].split('_')[0]
            label_path = _find_label(label_dir, f"{stem}_manual1")
            if label_path is None:
                continue

            im    = _read_image(os.path.join(test_dir, fname), mode='RGB')
            im    = self._test_pipeline.apply(im)
            label = _read_image(label_path, mode='L')

            im    = cv2.resize(im,    (self.cfg.original_w, self.cfg.original_h))
            label = cv2.resize(label, (self.cfg.original_w, self.cfg.original_h))
            # cv2.resize squeezes (H,W,1) → (H,W); restore channel dim if needed
            if im.ndim == 2:
                im = np.expand_dims(im, axis=-1)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

            x_list.append(im)
            y_list.append(np.expand_dims(label, axis=-1))

            mask_path = (
                _find_label(mask_dir, f"{stem}_test_mask") or
                _find_label(mask_dir, f"{stem}_mask")
            )
            if mask_path:
                mask = _read_image(mask_path, mode='L')
                mask = cv2.resize(mask, (self.cfg.original_w, self.cfg.original_h))
                m_list.append(mask)

        x_test = np.array(x_list, dtype=np.float32) / 255.0
        y_test = np.array(y_list, dtype=np.float32) / 255.0
        masks  = np.array(m_list, dtype=np.float32) / 255.0 if m_list else None

        if self.cfg.use_mask_eval and masks is None:
            print(
                f"  [WARN] use_mask_eval=True tetapi tidak ada file mask yang ditemukan "
                f"di: {mask_dir}\n"
                f"         Evaluasi akan dilakukan TANPA FOV mask. "
                f"Hasil metrik mungkin berbeda dari paper."
            )
        elif masks is not None:
            print(f"  FOV mask dimuat: {len(m_list)} file — evaluasi WITH FOV mask.")

        x_padded = self._pad_batch(x_test)
        return x_padded, y_test, masks

    def restore_predictions(self, y_pred_padded: np.ndarray) -> np.ndarray:
        """Crop padding from model output to recover original spatial size."""
        n = y_pred_padded.shape[0]
        restored = np.zeros((n, self.cfg.original_h, self.cfg.original_w, 1), dtype=np.float32)
        for i in range(n):
            restored[i] = y_pred_padded[i, :self.cfg.original_h, :self.cfg.original_w, :]
        return restored

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_split(self, img_dir: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all image/label pairs from a DRIVE split directory.
        Label stem is derived as fname.split('_')[0] + '_manual1'.
        Works for both original files ('21_training.tif' → stem '21')
        and augmented files ('randomRotation021_training.png' → stem 'randomRotation021').
        """
        files = sorted(f for f in os.listdir(img_dir) if not f.startswith('.'))
        x_list, y_list = [], []

        for fname in files:
            stem       = fname.split('_')[0]
            label_path = _find_label(label_dir, f"{stem}_manual1")
            if label_path is None:
                continue

            im    = _read_image(os.path.join(img_dir, fname), mode='RGB')
            im    = self._train_pipeline.apply(im)
            label = _read_image(label_path, mode='L')

            im_pad    = _pad_to(im,    self.target_h, self.target_w)
            lbl_pad   = _pad_to(label, self.target_h, self.target_w)
            _, lbl_bin = cv2.threshold(
                lbl_pad.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )

            x_list.append(im_pad)
            y_list.append(np.expand_dims(lbl_bin, axis=-1))

        x = np.array(x_list, dtype=np.float32) / 255.0
        y = np.array(y_list, dtype=np.float32) / 255.0
        return x, y

    def _pad_batch(self, batch: np.ndarray) -> np.ndarray:
        n, h, w, c = batch.shape
        out = np.zeros((n, self.target_h, self.target_w, c), dtype=batch.dtype)
        out[:, :h, :w, :] = batch
        return out
