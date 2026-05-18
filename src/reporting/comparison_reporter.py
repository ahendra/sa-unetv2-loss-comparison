import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LOSS_FUNCTIONS, RESULTS_DIR


_PRIMARY_METRICS = ["f1", "sensitivity", "specificity", "auc", "mcc", "jaccard"]
_ALL_METRICS     = _PRIMARY_METRICS + ["cldice", "betti0_error", "betti1_error"]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}

# Category mapping for Section 4.6.2
_LOSS_CATEGORIES = {
    "bce_mcc":   "Pixel-wise",
    "dice":      "Overlap",
    "focal":     "Imbalance-aware",
    "cldice":    "Topology-aware",
    "dice_ssim": "Hybrid (Area+Texture)",
    "bce_ssim":  "Hybrid (BCE+Texture)",
}


class ComparisonReporter:
    """Generate all comparison artifacts for Section 4.6.

    Outputs per dataset:
      - radar_chart_<dataset>.png   (spider chart, 6 metrics × 6 losses)
      - bar_chart_<dataset>.png     (grouped bar chart per metric)
      - ranking_table_<dataset>.json (ranked loss functions per metric)

    Single Responsibility: reads existing *_results.json files and produces
    visual + tabular comparisons. Does not perform training or inference.
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    def generate_all(self, dataset: str = "drive") -> Dict[str, Path]:
        """Load results and generate all comparison outputs.

        Args:
            dataset: 'drive' or 'stare'

        Returns:
            Dict mapping output type to saved Path.
        """
        results = self._load_results(dataset)
        if not results:
            print(f"  [WARN] Tidak ada hasil evaluasi untuk dataset '{dataset}'.")
            return {}

        self._out_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        paths["radar"]     = self._plot_radar(results, dataset)
        paths["bar"]       = self._plot_bar(results, dataset)
        paths["confusion"] = self._plot_confusion_matrices(results, dataset)
        paths["ranking"]   = self._save_ranking(results, dataset)
        return paths

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_results(self, dataset: str) -> Dict[str, Dict]:
        results = {}
        for loss_key in LOSS_FUNCTIONS:
            path = RESULTS_DIR / dataset / f"{loss_key}_results.json"
            if path.exists():
                with open(path) as f:
                    results[loss_key] = json.load(f)
        return results

    # ── Radar chart ───────────────────────────────────────────────────────────

    def _plot_radar(self, results: Dict, dataset: str) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip radar chart.")
            return None

        metrics     = _PRIMARY_METRICS
        n_m         = len(metrics)
        angles      = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles     += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 9), subplot_kw=dict(polar=True))
        colors  = plt.cm.tab10(np.linspace(0, 1, len(LOSS_FUNCTIONS)))

        for (loss_key, loss_label), color in zip(LOSS_FUNCTIONS.items(), colors):
            if loss_key not in results:
                continue
            vals  = [results[loss_key].get(m, 0) / 100.0 for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, lw=1.8,
                    label=loss_label.replace(" (Baseline)", ""))
            ax.fill(angles, vals, color=color, alpha=0.08)

        # Dynamic radial range — zoom in so small differences are visible
        all_vals_norm = [
            results[k].get(m, 0) / 100.0
            for k in results for m in metrics
        ]
        v_min = min(all_vals_norm) if all_vals_norm else 0.0
        v_max = max(all_vals_norm) if all_vals_norm else 1.0
        span  = max(v_max - v_min, 0.01)
        r_min = max(0.0, v_min - max(0.03, span * 0.25))
        r_max = min(1.0, v_max + max(0.01, span * 0.05))
        ticks = np.linspace(r_min, r_max, 6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
        ax.set_ylim(r_min, r_max)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t * 100:.2f}%" for t in ticks], fontsize=7)
        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}\n(6 primary metrics)",
            fontsize=11, fontweight="bold", pad=20,
        )
        # Legend placed below the polar axes — avoids overlap with spokes/labels
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=9, framealpha=0.9,
        )

        path = self._out_dir / f"radar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Radar chart: {path}")
        return path

    # ── Bar chart ─────────────────────────────────────────────────────────────

    def _plot_bar(self, results: Dict, dataset: str) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        metrics       = _PRIMARY_METRICS
        metric_labels = ["F1", "Sensitivity", "Specificity", "AUC", "MCC", "Jaccard"]
        loss_keys     = [k for k in LOSS_FUNCTIONS if k in results]
        loss_labels   = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        n_losses = len(loss_keys)
        colors   = plt.cm.tab10(np.linspace(0, 1, n_losses))

        # One subplot per metric — each gets its own zoomed y-axis so even
        # sub-1% differences between loss functions are clearly visible.
        n_cols = 3
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, 5 * n_rows),
            gridspec_kw={"hspace": 0.60, "wspace": 0.35},
        )
        fig.suptitle(
            f"Loss Function Comparison — {dataset.upper()}",
            fontsize=12, fontweight="bold",
        )
        axes_flat = np.array(axes).flatten()

        for m_idx, (mkey, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax   = axes_flat[m_idx]
            vals = [results[k].get(mkey, 0) for k in loss_keys]

            # Per-metric y-axis zoom
            v_min = min(vals)
            v_max = max(vals)
            span  = max(v_max - v_min, 0.05)
            y_min = max(0.0,   v_min - max(0.3, span * 0.8))
            y_max = min(100.0, v_max + max(0.3, span * 1.5))
            label_offset = (y_max - y_min) * 0.025

            for i, (k, label, color) in enumerate(zip(loss_keys, loss_labels, colors)):
                v = results[k].get(mkey, 0)
                ax.bar(i, v, width=0.65, color=color, alpha=0.85,
                       edgecolor="white", label=label)
                ax.text(i, v + label_offset, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

            ax.set_xticks([])
            ax.set_ylabel("Score (%)", fontsize=9)
            ax.set_title(mlabel, fontsize=10, fontweight="bold")
            ax.set_xlim(-0.6, n_losses - 0.4)
            ax.set_ylim(y_min, y_max)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused panels
        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].axis("off")

        # Single figure-level legend below all subplots — no overlap risk
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=min(n_losses, 3),
            bbox_to_anchor=(0.5, -0.03),
            fontsize=9, framealpha=0.9,
        )

        path = self._out_dir / f"bar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart: {path}")
        return path

    # ── Confusion matrix grid ─────────────────────────────────────────────────

    def _plot_confusion_matrices(self, results: Dict, dataset: str) -> Optional[Path]:
        """Plot one 2×2 confusion matrix per loss function using actual pixel counts.

        Pixel counts (TP, TN, FP, FN) are read from *_results.json.
        If a file predates this feature (no count keys), falls back to
        deriving approximate rates from Sensitivity / Specificity.

        Layout — rows = Actual class, cols = Predicted class:
          [TN  FP]   actual = Background (non-vessel)
          [FN  TP]   actual = Vessel

        Colour scheme matches segmentation_grid Peta Kesalahan:
          TP = Hijau    FP = Merah    FN = Biru    TN = Hijau muda
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip confusion matrix.")
            return None

        loss_keys   = [k for k in LOSS_FUNCTIONS if k in results]
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        n = len(loss_keys)

        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4.8, n_rows * 4.6),
            gridspec_kw={"hspace": 0.60, "wspace": 0.40},
        )
        axes_flat = np.array(axes).flatten()

        # Cell colours (RGB 0–1) — consistent with visualization_reporter
        _CELL_COLOR = {
            "TN": np.array([0.55, 0.88, 0.55]),   # hijau muda
            "FP": np.array([0.92, 0.40, 0.40]),   # merah
            "FN": np.array([0.40, 0.40, 0.92]),   # biru
            "TP": np.array([0.10, 0.68, 0.10]),   # hijau tua
        }
        # [row][col] mapping: row=Actual, col=Predicted
        _CELL_KEY = [["TN", "FP"],   # Actual = Background
                     ["FN", "TP"]]   # Actual = Vessel

        for idx, (loss_key, label) in enumerate(zip(loss_keys, loss_labels)):
            ax = axes_flat[idx]
            m  = results[loss_key]

            # --- Retrieve per-image average pixel counts -----------------
            has_counts = all(k in m for k in
                             ("tp_count", "tn_count", "fp_count", "fn_count"))
            n_imgs = max(int(m.get("num_images", 1)), 1)

            if has_counts:
                # Opsi B: average pixel count per image
                tp = m["tp_count"] / n_imgs
                tn = m["tn_count"] / n_imgs
                fp = m["fp_count"] / n_imgs
                fn = m["fn_count"] / n_imgs
            else:
                # Fallback: derive approximate rates from averaged metrics
                # (shown as float rates, flagged for re-evaluation)
                tpr = m.get("sensitivity", 0) / 100.0
                tnr = m.get("specificity", 0) / 100.0
                tp, tn, fp, fn = tpr, tnr, 1.0 - tnr, 1.0 - tpr

            total = tp + tn + fp + fn if (tp + tn + fp + fn) > 0 else 1

            # counts[row][col]: row=Actual, col=Predicted
            counts = [[tn, fp],
                      [fn, tp]]

            # --- Cell background: intensity ∝ count / row maximum -------
            row_maxes = [max(counts[r][0], counts[r][1]) for r in range(2)]
            rgb_img = np.ones((2, 2, 3), dtype=np.float64)
            for r in range(2):
                for c in range(2):
                    v    = counts[r][c] / (row_maxes[r] + 1e-9)
                    base = _CELL_COLOR[_CELL_KEY[r][c]]
                    rgb_img[r, c] = 1.0 - v * (1.0 - base)

            ax.imshow(rgb_img, interpolation="nearest", aspect="auto",
                      extent=[-0.5, 1.5, 1.5, -0.5])
            ax.axhline(0.5, color="white", lw=3.0)
            ax.axvline(0.5, color="white", lw=3.0)

            # --- Cell text: label / avg pixel count / percentage ---------
            for r in range(2):
                for c in range(2):
                    cnt      = counts[r][c]
                    cell_key = _CELL_KEY[r][c]
                    pct      = cnt / total * 100
                    bg_val   = cnt / (row_maxes[r] + 1e-9)
                    txt_col  = "white" if bg_val > 0.55 else "black"

                    if has_counts:
                        count_str = f"{cnt:,.0f} px"
                        pct_str   = f"({pct:.2f}%)"
                    else:
                        count_str = f"{cnt * 100:.2f}%"
                        pct_str   = "(~estimasi)"

                    ax.text(c, r - 0.18, cell_key,
                            ha="center", va="center",
                            fontsize=12, fontweight="bold", color=txt_col)
                    ax.text(c, r + 0.10, count_str,
                            ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_col)
                    ax.text(c, r + 0.35, pct_str,
                            ha="center", va="center",
                            fontsize=8, color=txt_col)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Prediksi\nBackground", "Prediksi\nVessel"],
                                fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Aktual\nBackground", "Aktual\nVessel"],
                                fontsize=9)
            ax.tick_params(length=0)
            ax.set_title(label, fontsize=10, fontweight="bold", pad=10)
            if not has_counts:
                ax.set_xlabel("⚠ Re-run evaluasi untuk pixel counts",
                              fontsize=7, color="gray")

        # Hide unused panels
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis("off")

        subtitle = (
            "Rata-rata jumlah pixel per gambar test (total ÷ jumlah gambar)\n"
            "Persentase relatif terhadap total pixel rata-rata per gambar"
        )
        fig.suptitle(
            f"Confusion Matrix — {dataset.upper()}\n{subtitle}",
            fontsize=11, fontweight="bold",
        )

        path = self._out_dir / f"confusion_matrix_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Confusion matrix: {path}")
        return path

    # ── Ranking table ─────────────────────────────────────────────────────────

    def _save_ranking(self, results: Dict, dataset: str) -> Path:
        ranking: Dict = {}
        for metric in _ALL_METRICS:
            reverse = metric not in _LOWER_IS_BETTER
            items = [
                (k, results[k].get(metric, float("nan")))
                for k in LOSS_FUNCTIONS
                if k in results
            ]
            items.sort(key=lambda x: (x[1] != x[1], x[1]),
                       reverse=reverse)
            ranking[metric] = [
                {
                    "rank":       i + 1,
                    "loss_key":   k,
                    "loss_label": LOSS_FUNCTIONS[k],
                    "category":   _LOSS_CATEGORIES.get(k, ""),
                    "value":      round(v, 2) if v == v else None,
                }
                for i, (k, v) in enumerate(items)
            ]

        path = self._out_dir / f"ranking_table_{dataset}.json"
        with open(path, "w") as f:
            json.dump(ranking, f, indent=2)
        print(f"  Ranking table: {path}")
        return path
