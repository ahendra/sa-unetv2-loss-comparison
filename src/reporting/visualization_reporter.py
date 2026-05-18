import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, RESULTS_DIR, StareConfig


# Row descriptions — shown as rotated labels on the left side of each figure
_ROW_LABELS = [
    "Gambar Input",
    "Ground Truth\n(Manual Annotation)",
    "Prediksi Biner\n(Threshold 0.5)",
    "Peta Kesalahan\n(TP / TN / FP / FN)",
]


class VisualizationReporter:
    """Build per-sample segmentation comparison grids for Section 4.5.

    Each sample image is saved as a separate PNG so the output fits on an A4
    page when printed.  Grid layout per file (4 rows × n_loss_functions cols):

      Row 0 — Gambar Input         : padded test image
      Row 1 — Ground Truth         : manual annotation (grayscale)
      Row 2 — Prediksi Biner       : thresholded prediction (grayscale)
      Row 3 — Peta Kesalahan       : colour-coded TP / TN / FP / FN

    Colour coding (Peta Kesalahan):
      TP = Hijau  (#00B400)  — vessel terdeteksi benar
      TN = Putih             — background terdeteksi benar
      FP = Merah  (#DC0000)  — background salah diprediksi sebagai vessel
      FN = Biru   (#0000DC)  — vessel tidak terdeteksi

    Output filenames:
      segmentation_grid_<dataset>_sample_<N>.png
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        output_dir: Path,
        n_samples: int = 3,
    ):
        self._cfg       = cfg
        self._out_dir   = output_dir
        self._n_samples = n_samples

    def generate(
        self,
        x_test: np.ndarray,
        y_test,
        sample_indices: Optional[List[int]] = None,
    ) -> List[Path]:
        """Build and save one PNG per sample.

        Args:
            x_test:         padded test images float32 (N, H, W, C), range [0,1]
            y_test:         ground truth, list or array, each (H, W, 1) float32
            sample_indices: which test images to include (default: first n_samples)

        Returns:
            List of saved PNG paths, one per sample.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip visualisasi.")
            return []

        n_imgs = len(x_test)
        if sample_indices is None:
            sample_indices = list(range(min(self._n_samples, n_imgs)))

        loss_keys   = list(LOSS_FUNCTIONS.keys())
        loss_labels = list(LOSS_FUNCTIONS.values())
        n_cols = len(loss_keys)

        # Colour legend patches for Peta Kesalahan row
        legend_patches = [
            Patch(facecolor=(0/255, 180/255,   0/255), edgecolor="#888",
                  label="TP — Vessel terdeteksi benar"),
            Patch(facecolor="white",                   edgecolor="#888",
                  label="TN — Background terdeteksi benar"),
            Patch(facecolor=(220/255,   0/255,   0/255), edgecolor="#888",
                  label="FP — Background diprediksi sebagai vessel"),
            Patch(facecolor=(0/255,   0/255, 220/255), edgecolor="#888",
                  label="FN — Vessel tidak terdeteksi"),
        ]

        self._out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        for img_idx in sample_indices:
            fig, axes = plt.subplots(
                4, n_cols,
                figsize=(2.6 * n_cols, 3.3 * 4),
                gridspec_kw={"hspace": 0.10, "wspace": 0.05},
            )
            # Normalise axes shape so indexing is always [row, col]
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            fig.suptitle(
                f"Segmentation Results — {self._cfg.name}   "
                f"(Sampel #{img_idx + 1})",
                fontsize=11, fontweight="bold",
            )

            gt      = y_test[img_idx]
            gt_disp = gt.squeeze()   # (H, W) float32

            for col_idx, (loss_key, loss_label) in enumerate(
                zip(loss_keys, loss_labels)
            ):
                pred_dir  = (
                    RESULTS_DIR / self._cfg.name.lower()
                    / "predictions" / loss_key
                )
                pred_path = pred_dir / f"pred_{img_idx + 1:03d}.png"
                pred_img  = (
                    cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                    if pred_path.exists() else None
                )

                # ── Row 0: gambar input ───────────────────────────────────
                orig = x_test[img_idx]
                if orig.shape[-1] == 1:
                    orig_disp, cmap_orig = orig.squeeze(), "gray"
                else:
                    orig_disp, cmap_orig = orig, None

                ax = axes[0, col_idx]
                ax.imshow(orig_disp, cmap=cmap_orig)
                ax.axis("off")
                short_label = loss_label.replace(" (Baseline)", "")
                ax.set_title(short_label, fontsize=8, pad=4, fontweight="bold")

                # ── Row 1: ground truth ───────────────────────────────────
                ax = axes[1, col_idx]
                ax.imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

                # ── Row 2: prediksi biner ─────────────────────────────────
                ax = axes[2, col_idx]
                if pred_img is not None:
                    ax.imshow(pred_img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "N/A\n(belum dievaluasi)",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="#666")
                ax.axis("off")

                # ── Row 3: peta kesalahan (TP / TN / FP / FN) ────────────
                ax = axes[3, col_idx]
                if pred_img is not None:
                    pred_bin = (pred_img > 127).astype(np.uint8)
                    gt_bin   = (gt_disp  > 0.5).astype(np.uint8)

                    if pred_bin.shape != gt_bin.shape:
                        pred_bin = cv2.resize(
                            pred_bin,
                            (gt_bin.shape[1], gt_bin.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # TN = white background; colour only the non-TN classes
                    err_map = np.full((*gt_bin.shape, 3), 255, dtype=np.uint8)
                    err_map[(pred_bin == 1) & (gt_bin == 1)] = [0,   180,   0]  # TP hijau
                    err_map[(pred_bin == 1) & (gt_bin == 0)] = [220,   0,   0]  # FP merah
                    err_map[(pred_bin == 0) & (gt_bin == 1)] = [0,     0, 220]  # FN biru
                    ax.imshow(err_map)
                else:
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="#666")
                ax.axis("off")

            # ── Row labels on the left of the first column ───────────────
            for row_idx, row_label in enumerate(_ROW_LABELS):
                axes[row_idx, 0].text(
                    -0.10, 0.5, row_label,
                    transform=axes[row_idx, 0].transAxes,
                    fontsize=8, ha="right", va="center",
                    rotation=90, fontweight="bold",
                    clip_on=False,
                )

            # ── Colour legend for Peta Kesalahan ─────────────────────────
            fig.legend(
                handles=legend_patches,
                loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.03),
                fontsize=8, framealpha=0.95,
                title="Keterangan Warna — Peta Kesalahan (Baris 4)",
                title_fontsize=8,
            )

            path = (
                self._out_dir
                / f"segmentation_grid_{self._cfg.name.lower()}"
                  f"_sample_{img_idx + 1}.png"
            )
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Grid visualisasi sampel #{img_idx + 1}: {path}")
            saved_paths.append(path)

        return saved_paths
