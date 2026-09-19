import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, RESULTS_DIR, StareConfig


# ── Language-dependent text ───────────────────────────────────────────────────

_ROW_LABELS = {
    "id": [
        "Gambar Input",
        "Ground Truth\n(Manual Annotation)",
        "Peta Probabilitas\n(Sebelum Threshold)",
        "Prediksi Biner\n(Threshold 0.5)",
        "Peta Kesalahan\n(TP / TN / FP / FN)",
        "Detail Ground Truth\n(Zoom Pusat Retina)",
        "Detail Prediksi Biner\n(Zoom Pusat Retina)",
        "Detail Peta Kesalahan\n(Zoom Pusat Retina)",
    ],
    "en": [
        "Input Image",
        "Ground Truth\n(Manual Annotation)",
        "Probability Map\n(Pre-threshold)",
        "Binary Prediction\n(Threshold 0.5)",
        "Error Map\n(TP / TN / FP / FN)",
        "Ground Truth Detail\n(Centre Crop)",
        "Prediction Detail\n(Centre Crop)",
        "Error Map Detail\n(Centre Crop)",
    ],
}

_LEGEND_TEXT = {
    "id": {
        "tp":        "TP — Vessel terdeteksi benar",
        "tn":        "TN — Background terdeteksi benar",
        "fp":        "FP — Background diprediksi sebagai vessel",
        "fn":        "FN — Vessel tidak terdeteksi",
        "title":     "Keterangan Warna — Peta Kesalahan (Baris 5)",
        "na_detail": "N/A\n(belum dievaluasi)",
        "na":        "N/A",
    },
    "en": {
        "tp":        "TP — Vessel correctly detected",
        "tn":        "TN — Background correctly detected",
        "fp":        "FP — Background predicted as vessel",
        "fn":        "FN — Vessel not detected",
        "title":     "Colour Legend — Error Map (Row 5)",
        "na_detail": "N/A\n(not yet evaluated)",
        "na":        "N/A",
    },
}

_N_IMG_ROWS    = 8
_DEFAULT_SEEDS = [42, 123, 456, 789, 2026]

# A4 two-column full-width: text area = (210 − 13 − 13) mm = 184 mm
_FIG_W_EN_IN = 184 / 25.4   # 7.244 in


def _draw_zoom_box(ax, x1: int, y1: int, x2: int, y2: int) -> None:
    """Overlay a red rectangle that marks the zoomed crop region."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=1.8, edgecolor="#FF0000", facecolor="none", zorder=10,
    ))


def _find_median_seeds(
    cfg: Union[DriveConfig, StareConfig],
    seeds: List[int],
) -> Dict[str, str]:
    """Return {loss_key: seed_tag} where seed_tag is the seed whose F1 is
    closest to the mean F1 across all seeds (most representative result).
    Falls back to the first seed if multiseed_summary.json is missing.
    """
    summary_path = (
        RESULTS_DIR / cfg.name.lower()
        / "reports" / "section_4_10_multiseed"
        / cfg.name.lower() / "multiseed_summary.json"
    )
    fallback_tag = f"seed{seeds[0]}"
    fallback     = {k: fallback_tag for k in LOSS_FUNCTIONS}

    if not summary_path.exists():
        print(f"  [WARN] multiseed_summary.json tidak ditemukan — "
              f"menggunakan {fallback_tag} untuk semua loss.")
        return fallback

    with open(summary_path) as f:
        data = json.load(f)

    # Support both flat {loss_key: {...}} and nested {"summary": {loss_key: {...}}}
    summary = data.get("summary", data)

    result: Dict[str, str] = {}
    print(f"\n  Median seed per loss function (F1 paling dekat ke mean):")
    for loss_key in LOSS_FUNCTIONS:
        f1_info  = summary.get(loss_key, {}).get("f1", {})
        raw_vals = f1_info.get("raw", [])
        mean_f1  = f1_info.get("mean", None)

        if not raw_vals or mean_f1 is None or len(raw_vals) != len(seeds):
            result[loss_key] = fallback_tag
            print(f"    {loss_key:<12} → {fallback_tag} (fallback)")
            continue

        best_idx = min(range(len(seeds)), key=lambda i: abs(raw_vals[i] - mean_f1))
        seed_tag = f"seed{seeds[best_idx]}"
        result[loss_key] = seed_tag
        print(f"    {loss_key:<12} → {seed_tag}  "
              f"(F1={raw_vals[best_idx]:.2f}%, mean={mean_f1:.2f}%)")

    return result


class VisualizationReporter:
    """Build per-sample segmentation comparison grids for Section 4.5.

    Grid layout per file (8 image rows + 1 label row + 1 legend row):

      Row 0 — Input image               : padded test image
      Row 1 — Ground Truth              : manual annotation (grayscale)
      Row 2 — Probability Map           : raw sigmoid output (pre-threshold)
      Row 3 — Binary Prediction         : thresholded prediction (grayscale)
      Row 4 — Error Map                 : colour-coded TP / TN / FP / FN
      Row 5 — Ground Truth Detail (Zoom): centre-crop of ground truth
      Row 6 — Prediction Detail (Zoom)  : centre-crop of binary prediction
      Row 7 — Error Map Detail (Zoom)   : centre-crop of error map
      [col-label row]                   : loss-function names
      [legend row]                      : colour coding key

    Rows 0–4 carry a red rectangle marking the zoom region.

    Colour coding (Error Map) — colorblind-safe palette (Wong 2011):
      TP = Bluish-green (#009E73)
      TN = White
      FP = Orange       (#E69F00)
      FN = Blue         (#0072B2)

    Predictions are sourced from the median seed per loss function — the seed
    whose F1 score is closest to the mean F1 across all seeds — giving the
    most representative visual result for each loss function.

    Parameters
    ----------
    lang  : "id" (default) or "en"
    seeds : list of experiment seeds; used to locate the median seed
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        output_dir: Path,
        n_samples: int = 3,
        zoom_fraction: float = 0.40,
        lang: str = "id",
        seeds: Optional[List[int]] = None,
    ):
        self._cfg          = cfg
        self._out_dir      = output_dir
        self._n_samples    = n_samples
        self._zoom_frac    = zoom_fraction
        self._lang         = lang
        self._seeds        = seeds or _DEFAULT_SEEDS
        self._median_seeds = _find_median_seeds(cfg, self._seeds)

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
        n_cols      = len(loss_keys)

        txt        = _LEGEND_TEXT[self._lang]
        row_labels = _ROW_LABELS[self._lang]

        legend_patches = [
            Patch(facecolor=(  0/255, 158/255, 115/255), edgecolor="#888",
                  label=txt["tp"]),
            Patch(facecolor="white",                     edgecolor="#888",
                  label=txt["tn"]),
            Patch(facecolor=(230/255, 159/255,   0/255), edgecolor="#888",
                  label=txt["fp"]),
            Patch(facecolor=(  0/255, 114/255, 178/255), edgecolor="#888",
                  label=txt["fn"]),
        ]

        # ── Layout parameters ─────────────────────────────────────────────────
        if self._lang == "en":
            fig_w         = _FIG_W_EN_IN
            col_w         = fig_w / n_cols
            _min_row_h    = 1.12
            if sample_indices:
                _ref           = np.squeeze(y_test[sample_indices[0]])
                _img_h, _img_w = _ref.shape[:2]
                _natural_h     = col_w * (_img_h / _img_w)
            else:
                _natural_h = col_w
            img_row_h     = max(_natural_h, _min_row_h)
            col_label_h   = 0.14
            legend_h      = 0.55
            _dpi          = 300
            _legend_ncol  = 2
            _top          = 0.99
            _row_label_fs = 7
            _legend_fs    = 8
        else:
            col_w         = 2.55
            img_row_h     = 2.50
            col_label_h   = 0.28
            legend_h      = 0.46
            fig_w         = col_w * n_cols
            _dpi          = 300
            _legend_ncol  = 2
            _top          = 0.97
            _row_label_fs = 8
            _legend_fs    = 8

        col_label_ratio = col_label_h / img_row_h
        leg_ratio       = legend_h    / img_row_h
        fig_h           = img_row_h * _N_IMG_ROWS + col_label_h + legend_h

        _rc_params = {}
        if self._lang == "en":
            _rc_params = {
                "font.family":     "sans-serif",
                "font.sans-serif": ["Helvetica", "Arial",
                                    "Helvetica Neue", "DejaVu Sans"],
                "font.size":       8,
                "axes.titlesize":  8,
                "axes.labelsize":  8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
            }

        self._out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        with matplotlib.rc_context(_rc_params):
            for img_idx in sample_indices:
                fig = plt.figure(figsize=(fig_w, fig_h))

                gs = gridspec.GridSpec(
                    _N_IMG_ROWS + 2, n_cols,
                    height_ratios=[1.0] * _N_IMG_ROWS + [col_label_ratio, leg_ratio],
                    hspace=0.06,
                    wspace=0.05,
                    top=_top,
                    bottom=0.01,
                )

                axes = np.empty((_N_IMG_ROWS, n_cols), dtype=object)
                for r in range(_N_IMG_ROWS):
                    for c in range(n_cols):
                        axes[r, c] = fig.add_subplot(gs[r, c])

                axes_labels = np.empty(n_cols, dtype=object)
                for c in range(n_cols):
                    axes_labels[c] = fig.add_subplot(gs[_N_IMG_ROWS, c])
                    axes_labels[c].axis("off")

                ax_legend = fig.add_subplot(gs[_N_IMG_ROWS + 1, :])
                ax_legend.axis("off")

                if self._lang == "id":
                    fig.suptitle(
                        f"Segmentation Results — {self._cfg.name}   "
                        f"(Sampel #{img_idx + 1})",
                        fontsize=11, fontweight="bold", y=0.99,
                    )

                gt      = y_test[img_idx]
                gt_disp = gt.squeeze()   # (H, W) float32

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
                    seed_tag  = self._median_seeds.get(loss_key, f"seed{self._seeds[0]}")
                    pred_dir  = (
                        RESULTS_DIR / self._cfg.name.lower()
                        / "predictions" / f"{loss_key}_{seed_tag}"
                    )
                    pred_path = pred_dir / f"pred_{img_idx + 1:03d}.png"
                    prob_path = pred_dir / f"prob_{img_idx + 1:03d}.png"

                    pred_img = (
                        cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                        if pred_path.exists() else None
                    )
                    prob_img = (
                        cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
                        if prob_path.exists() else None
                    )
                    short_label = loss_label.replace(" (Baseline)", "")

                    # Build error map once — reused for row 4 and row 7
                    err_map = None
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
                        err_map[(pred_bin == 1) & (gt_bin == 1)] = [  0, 158, 115]  # TP
                        err_map[(pred_bin == 1) & (gt_bin == 0)] = [230, 159,   0]  # FP
                        err_map[(pred_bin == 0) & (gt_bin == 1)] = [  0, 114, 178]  # FN

                    # ── Row 0: Input image ────────────────────────────────────
                    orig = x_test[img_idx]
                    if orig.shape[-1] == 1:
                        orig_disp, cmap_orig = orig.squeeze(), "gray"
                    else:
                        orig_disp, cmap_orig = orig, None

                    ax = axes[0, col_idx]
                    ax.imshow(orig_disp, cmap=cmap_orig)
                    _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                    ax.axis("off")
                    ax.set_title(short_label, fontsize=8, pad=3, fontweight="bold")

                    # ── Row 1: Ground truth ───────────────────────────────────
                    ax = axes[1, col_idx]
                    ax.imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
                    _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                    ax.axis("off")

                    # ── Row 2: Probability map (pre-threshold) ────────────────
                    ax = axes[2, col_idx]
                    if prob_img is not None:
                        ax.imshow(prob_img, cmap="gray", vmin=0, vmax=255)
                        _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                    else:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, txt["na_detail"],
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#666")
                    ax.axis("off")

                    # ── Row 3: Binary prediction ──────────────────────────────
                    ax = axes[3, col_idx]
                    if pred_img is not None:
                        ax.imshow(pred_img, cmap="gray", vmin=0, vmax=255)
                        _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                    else:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, txt["na_detail"],
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#666")
                    ax.axis("off")

                    # ── Row 4: Error map ──────────────────────────────────────
                    ax = axes[4, col_idx]
                    if err_map is not None:
                        ax.imshow(err_map)
                        _draw_zoom_box(ax, zx1, zy1, zx2, zy2)
                    else:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, txt["na"],
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#666")
                    ax.axis("off")

                    # ── Row 5: Zoomed ground truth ────────────────────────────
                    ax = axes[5, col_idx]
                    ax.imshow(gt_disp[zy1:zy2, zx1:zx2], cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")

                    # ── Row 6: Zoomed binary prediction ──────────────────────
                    ax = axes[6, col_idx]
                    if pred_img is not None:
                        ax.imshow(pred_img[zy1:zy2, zx1:zx2], cmap="gray",
                                  vmin=0, vmax=255)
                    else:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, txt["na"],
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#666")
                    ax.axis("off")

                    # ── Row 7: Zoomed error map ───────────────────────────────
                    ax = axes[7, col_idx]
                    if err_map is not None:
                        ax.imshow(err_map[zy1:zy2, zx1:zx2])
                    else:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, txt["na"],
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#666")
                    ax.axis("off")

                    # ── Column label ──────────────────────────────────────────
                    axes_labels[col_idx].text(
                        0.5, 0.5, short_label,
                        transform=axes_labels[col_idx].transAxes,
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                    )

                # ── Row labels on left edge ───────────────────────────────────
                for row_idx, row_label in enumerate(row_labels):
                    axes[row_idx, 0].text(
                        -0.10, 0.5, row_label,
                        transform=axes[row_idx, 0].transAxes,
                        fontsize=_row_label_fs, ha="right", va="center",
                        rotation=90, fontweight="bold",
                        clip_on=False,
                    )

                # ── Colour legend ─────────────────────────────────────────────
                ax_legend.legend(
                    handles=legend_patches,
                    loc="center", ncol=_legend_ncol,
                    fontsize=_legend_fs, framealpha=0.95,
                    title=txt["title"],
                    title_fontsize=_legend_fs,
                )

                _lang_suffix = "_en" if self._lang == "en" else ""
                path = (
                    self._out_dir
                    / f"segmentation_grid_{self._cfg.name.lower()}"
                      f"_sample_{img_idx + 1}{_lang_suffix}.png"
                )
                fig.savefig(str(path), dpi=_dpi, bbox_inches="tight",
                            facecolor="white", edgecolor="none")
                plt.close(fig)
                print(f"  Grid visualisasi sampel #{img_idx + 1}: {path}")
                saved_paths.append(path)

        return saved_paths
