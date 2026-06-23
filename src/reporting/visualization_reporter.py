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
    "Detail Pembuluh\n(Zoom Pusat Retina)",
]

_N_IMG_ROWS = 5


def _draw_zoom_box(ax, x1: int, y1: int, x2: int, y2: int) -> None:
    """Overlay a red rectangle that marks the zoomed crop on a given axes."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=1.8, edgecolor="#FF0000", facecolor="none", zorder=10,
    ))


class VisualizationReporter:
    """Build per-sample segmentation comparison grids for Section 4.5.

    Grid layout per file (5 rows × n_loss_functions cols):

      Row 0 — Gambar Input              : padded test image
      Row 1 — Ground Truth              : manual annotation (grayscale)
      Row 2 — Prediksi Biner            : thresholded prediction (grayscale)
      Row 3 — Peta Kesalahan            : colour-coded TP / TN / FP / FN
      Row 4 — Detail Pembuluh (Zoom)    : centre-crop of binary prediction

    Rows 0–3 each carry a red rectangle that marks the zoomed crop region.
    The colour legend is rendered in a dedicated gridspec row directly below
    Row 4 with no extra blank space.

    Colour coding (Peta Kesalahan):
      TP = Hijau  (#00B400)  — vessel terdeteksi benar
      TN = Putih             — background terdeteksi benar
      FP = Merah  (#DC0000)  — background salah diprediksi sebagai vessel
      FN = Biru   (#0000DC)  — vessel tidak terdeteksi
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        output_dir: Path,
        n_samples: int = 3,
        zoom_fraction: float = 0.40,
    ):
        self._cfg        = cfg
        self._out_dir    = output_dir
        self._n_samples  = n_samples
        self._zoom_frac  = zoom_fraction

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
            import matplotlib.gridspec as gridspec
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

        legend_patches = [
            Patch(facecolor=(  0/255, 180/255,   0/255), edgecolor="#888",
                  label="TP — Vessel terdeteksi benar"),
            Patch(facecolor="white",                     edgecolor="#888",
                  label="TN — Background terdeteksi benar"),
            Patch(facecolor=(220/255,   0/255,   0/255), edgecolor="#888",
                  label="FP — Background diprediksi sebagai vessel"),
            Patch(facecolor=(  0/255,   0/255, 220/255), edgecolor="#888",
                  label="FN — Vessel tidak terdeteksi"),
        ]

        self._out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        for img_idx in sample_indices:
            # ── Figure dimensions ─────────────────────────────────────────
            col_w     = 2.55   # inches per column
            img_row_h = 2.75   # inches per image row (tighter than original 3.3)
            legend_h  = 0.46   # inches for the legend row

            fig_w = col_w * n_cols
            fig_h = img_row_h * _N_IMG_ROWS + legend_h

            fig = plt.figure(figsize=(fig_w, fig_h))

            # ── GridSpec: 5 image rows + 1 legend row ─────────────────────
            leg_ratio = legend_h / img_row_h   # relative to one image row
            gs = gridspec.GridSpec(
                _N_IMG_ROWS + 1, n_cols,
                height_ratios=[1.0] * _N_IMG_ROWS + [leg_ratio],
                hspace=0.06,
                wspace=0.05,
            )

            # Build [n_img_rows × n_cols] axes array
            axes = np.empty((_N_IMG_ROWS, n_cols), dtype=object)
            for r in range(_N_IMG_ROWS):
                for c in range(n_cols):
                    axes[r, c] = fig.add_subplot(gs[r, c])

            # Dedicated axes spanning all columns for the colour legend
            ax_legend = fig.add_subplot(gs[_N_IMG_ROWS, :])
            ax_legend.axis("off")

            fig.suptitle(
                f"Segmentation Results — {self._cfg.name}   "
                f"(Sampel #{img_idx + 1})",
                fontsize=11, fontweight="bold",
            )

            gt      = y_test[img_idx]
            gt_disp = gt.squeeze()   # (H, W) float32

            # ── Zoom crop region (same for every column) ──────────────────
            H, W = gt_disp.shape
            ch = int(H * self._zoom_frac)
            cw = int(W * self._zoom_frac)
            cy, cx = H // 2, W // 2
            zy1 = max(0, cy - ch // 2)
            zy2 = min(H, cy + ch // 2)
            zx1 = max(0, cx - cw // 2)
            zx2 = min(W, cx + cw // 2)

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
                _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                ax.axis("off")
                ax.set_title(
                    loss_label.replace(" (Baseline)", ""),
                    fontsize=8, pad=3, fontweight="bold",
                )

                # ── Row 1: ground truth ───────────────────────────────────
                ax = axes[1, col_idx]
                ax.imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
                _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                ax.axis("off")

                # ── Row 2: prediksi biner ─────────────────────────────────
                ax = axes[2, col_idx]
                if pred_img is not None:
                    ax.imshow(pred_img, cmap="gray", vmin=0, vmax=255)
                    _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
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

                    err_map = np.full((*gt_bin.shape, 3), 255, dtype=np.uint8)
                    err_map[(pred_bin == 1) & (gt_bin == 1)] = [  0, 180,   0]  # TP hijau
                    err_map[(pred_bin == 1) & (gt_bin == 0)] = [220,   0,   0]  # FP merah
                    err_map[(pred_bin == 0) & (gt_bin == 1)] = [  0,   0, 220]  # FN biru
                    ax.imshow(err_map)
                    _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                else:
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="#666")
                ax.axis("off")

                # ── Row 4: zoomed binary prediction ──────────────────────
                ax = axes[4, col_idx]
                if pred_img is not None:
                    ax.imshow(
                        pred_img[zy1:zy2, zx1:zx2],
                        cmap="gray", vmin=0, vmax=255,
                    )
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

            # ── Colour legend in the dedicated bottom row ─────────────────
            ax_legend.legend(
                handles=legend_patches,
                loc="center", ncol=2,
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
