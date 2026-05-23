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

        metrics = _PRIMARY_METRICS
        n_m     = len(metrics)
        angles  = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=dict(polar=True))
        colors       = plt.cm.tab10(np.linspace(0, 1, len(LOSS_FUNCTIONS)))
        linestyles   = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
        markerstyles = ['o', 's', '^', 'D', 'v', 'P']

        # Per-metric min-max normalization: each spoke independently stretched
        # to [_LO, _HI] so even sub-0.1% differences become clearly visible.
        # Actual value ranges are shown in the spoke labels below.
        _LO, _HI = 0.15, 0.90
        metric_stats: Dict[str, tuple] = {}
        for m in metrics:
            vals_m = [results[k].get(m, 0) / 100.0
                      for k in LOSS_FUNCTIONS if k in results]
            metric_stats[m] = (min(vals_m), max(vals_m))

        def _norm(val_pct: float, m: str) -> float:
            raw = val_pct / 100.0
            lo, hi = metric_stats[m]
            rng = (hi - lo) if (hi - lo) > 1e-9 else 1e-9
            return _LO + (raw - lo) / rng * (_HI - _LO)

        for idx, ((loss_key, loss_label), color) in enumerate(
                zip(LOSS_FUNCTIONS.items(), colors)):
            if loss_key not in results:
                continue
            norm_vals  = [_norm(results[loss_key].get(m, 0), m) for m in metrics]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals,
                    color=color, lw=2.0,
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markerstyles[idx % len(markerstyles)],
                    markersize=6,
                    label=loss_label.replace(" (Baseline)", ""))
            ax.fill(angles, norm_vals, color=color, alpha=0.04)

        # Radial ticks: relative labels (min → max per spoke)
        r_ticks = np.linspace(_LO, _HI, 5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(r_ticks.tolist())
        ax.set_yticklabels(["min", "25%", "50%", "75%", "max"],
                           fontsize=7, color="gray")

        # Spoke labels: metric name + actual [min–max] range in the data
        spoke_labels = [
            f"{m.upper()}\n"
            f"[{metric_stats[m][0]*100:.2f}–{metric_stats[m][1]*100:.2f}%]"
            for m in metrics
        ]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spoke_labels, fontsize=8)

        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}\n"
            f"Skala relatif per metrik  |  [min – max] = rentang nilai aktual",
            fontsize=11, fontweight="bold", pad=25,
        )
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
            f"Loss Function Comparison — {dataset.upper()}\n"
            f"(sumbu-Y diperbesar per subplot  |  ╱╲ = axis tidak mulai dari nol)",
            fontsize=12, fontweight="bold",
        )
        axes_flat = np.array(axes).flatten()

        for m_idx, (mkey, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax   = axes_flat[m_idx]
            vals = [results[k].get(mkey, 0) for k in loss_keys]

            v_min = min(vals)
            v_max = max(vals)
            span  = (v_max - v_min) if (v_max - v_min) > 1e-9 else 0.02

            # Tight zoom: pad only 40% below and 60% above the data range
            # so bars fill most of the subplot and small differences are visible
            y_min = max(0.0,   v_min - span * 0.4)
            y_max = min(100.0, v_max + span * 0.6)
            if (y_max - y_min) < 0.01:
                y_min = max(0.0,   v_min - 0.05)
                y_max = min(100.0, v_max + 0.05)
            label_offset = (y_max - y_min) * 0.02

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

            # Axis-break indicator when y-axis doesn't start from zero
            if y_min > 0:
                ax.spines["bottom"].set_linewidth(0)
                ax.tick_params(bottom=False)
                d  = 0.012
                kw = dict(transform=ax.transAxes, color="black",
                          clip_on=False, lw=1.5)
                ax.plot((-d, +d), (-2 * d, +2 * d), **kw)
                ax.plot((1 - d, 1 + d), (-2 * d, +2 * d), **kw)

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
