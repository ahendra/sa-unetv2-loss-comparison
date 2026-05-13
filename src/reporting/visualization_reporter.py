import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, RESULTS_DIR, StareConfig


class VisualizationReporter:
    """Build a segmentation comparison grid for Section 4.5.

    Grid layout:
      rows = [Original Image, Ground Truth, Binary Prediction, Probability Map]
      cols = one per loss function

    Reads prediction images already saved by ModelEvaluator from:
      results/<dataset>/predictions/<loss_key>/pred_NNN.png
      results/<dataset>/predictions/<loss_key>/prob_NNN.png

    Single Responsibility: visualisation grid only; no inference or file I/O
    beyond reading existing prediction images.
    """

    _ROW_LABELS = ["Original", "Ground Truth", "Prediction", "Prob Map"]

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        output_dir: Path,
        n_samples: int = 3,
    ):
        self._cfg      = cfg
        self._out_dir  = output_dir
        self._n_samples = n_samples

    def generate(
        self,
        x_test: np.ndarray,
        y_test,
        sample_indices: Optional[List[int]] = None,
    ) -> Path:
        """Build and save the grid PNG.

        Args:
            x_test:         padded test images float32 (N, H, W, C), range [0,1]
            y_test:         ground truth, list or array, each (H, W, 1) float32
            sample_indices: which test images to include (default: first n_samples)

        Returns:
            Path to saved PNG.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip visualisasi.")
            return None

        n_imgs = len(x_test)
        if sample_indices is None:
            sample_indices = list(range(min(self._n_samples, n_imgs)))

        loss_keys   = list(LOSS_FUNCTIONS.keys())
        loss_labels = list(LOSS_FUNCTIONS.values())
        n_rows = 4 * len(sample_indices)
        n_cols = len(loss_keys)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.5 * n_cols, 2.8 * n_rows),
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for col_idx, (loss_key, loss_label) in enumerate(zip(loss_keys, loss_labels)):
            pred_dir = (RESULTS_DIR / self._cfg.name.lower()
                        / "predictions" / loss_key)

            for sample_offset, img_idx in enumerate(sample_indices):
                base_row = sample_offset * 4

                # ── Row 0: original test image ────────────────────────────
                orig = x_test[img_idx]
                if orig.shape[-1] == 1:
                    orig_disp = orig.squeeze()
                    cmap_orig = "gray"
                else:
                    orig_disp = orig
                    cmap_orig = None
                ax = axes[base_row, col_idx]
                ax.imshow(orig_disp, cmap=cmap_orig)
                ax.axis("off")
                if sample_offset == 0:
                    short = loss_label.replace(" (Baseline)", "")
                    ax.set_title(short, fontsize=7, pad=3)
                if col_idx == 0:
                    ax.set_ylabel(f"#{img_idx+1} Original", fontsize=7)

                # ── Row 1: ground truth ───────────────────────────────────
                gt = y_test[img_idx] if hasattr(y_test, "__getitem__") else y_test[img_idx]
                gt_disp = gt.squeeze()
                ax = axes[base_row + 1, col_idx]
                ax.imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel("Ground Truth", fontsize=7)

                # ── Row 2: binary prediction ──────────────────────────────
                pred_path = pred_dir / f"pred_{img_idx + 1:03d}.png"
                ax = axes[base_row + 2, col_idx]
                if pred_path.exists():
                    pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                    ax.imshow(pred_img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel("Prediction", fontsize=7)

                # ── Row 3: probability map ────────────────────────────────
                prob_path = pred_dir / f"prob_{img_idx + 1:03d}.png"
                ax = axes[base_row + 3, col_idx]
                if prob_path.exists():
                    prob_img = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
                    im = ax.imshow(prob_img, cmap="hot", vmin=0, vmax=255)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel("Prob Map", fontsize=7)

        fig.suptitle(
            f"Segmentation Results — {self._cfg.name}  "
            f"(samples: {[i+1 for i in sample_indices]})",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / f"segmentation_grid_{self._cfg.name.lower()}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grid visualisasi: {path}")
        return path
