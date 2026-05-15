import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import StareConfig
from src.preprocessing import PreprocessingPipeline, build_pipeline


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
        self._pipeline = pipeline or build_pipeline(
            cfg.preprocessing_mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training split.
        Matches original notebook: load all augmented images then split 90/10
        with train_test_split(random_state=42).
        """
        x_all, y_all = self._load_all_augmented()
        x_train, _, y_train, _ = train_test_split(
            x_all, y_all, test_size=0.1, shuffle=True, random_state=42
        )
        return x_train, y_train

    def load_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation split (10% of augmented data, random_state=42)."""
        x_all, y_all = self._load_all_augmented()
        _, x_val, _, y_val = train_test_split(
            x_all, y_all, test_size=0.1, shuffle=True, random_state=42
        )
        return x_val, y_val

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
            im    = self._pipeline.apply(im)
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

    def _load_all_augmented(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all augmented images from both train and validate pool dirs."""
        x_list, y_list = [], []
        for img_dir, label_dir in (
            (self.cfg.aug_train_images, self.cfg.aug_train_labels),
            (self.cfg.aug_val_images,   self.cfg.aug_val_labels),
        ):
            if not os.path.isdir(img_dir):
                continue
            files = sorted(f for f in os.listdir(img_dir)
                           if f.lower().endswith('.png') and not f.startswith('.'))
            for fname in files:
                base       = os.path.splitext(fname)[0]
                label_name = f"{base}.ah.png"
                label_path = os.path.join(label_dir, label_name)
                if not os.path.exists(label_path):
                    continue

                img   = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB'))
                img   = self._pipeline.apply(img)
                label = np.array(Image.open(label_path).convert('L'))

                img_pad = _pad_symmetric(img, self.target_h, self.target_w)
                lbl_pad = _pad_symmetric(label, self.target_h, self.target_w)
                _, lbl_bin = cv2.threshold(
                    lbl_pad.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
                )

                x_list.append(img_pad)
                y_list.append(np.expand_dims(lbl_bin, axis=-1))

        x = np.array(x_list, dtype=np.float32) / 255.0
        y = np.array(y_list, dtype=np.float32) / 255.0
        return x, y
