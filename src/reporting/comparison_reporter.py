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

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
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
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}\n(6 primary metrics)",
            fontsize=11, fontweight="bold", pad=20,
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
        x        = np.arange(len(metrics))
        width    = 0.8 / n_losses
        colors   = plt.cm.tab10(np.linspace(0, 1, n_losses))

        # Dynamic y-axis range — zoom in so small differences are visible
        all_vals = [results[k].get(m, 0) for k in loss_keys for m in metrics]
        v_min = min(all_vals) if all_vals else 0.0
        v_max = max(all_vals) if all_vals else 100.0
        span  = max(v_max - v_min, 1.0)
        y_min = max(0.0,   v_min - max(2.0, span * 0.30))
        y_max = min(100.0, v_max + max(1.0, span * 0.15))

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, (loss_key, label, color) in enumerate(zip(loss_keys, loss_labels, colors)):
            vals = [results[loss_key].get(m, 0) for m in metrics]
            offset = (i - n_losses / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, alpha=0.85, edgecolor="white")
            label_offset = (y_max - y_min) * 0.008
            for bar in bars:
                h = bar.get_height()
                if h > y_min:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + label_offset,
                        f"{h:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(y_min, y_max)

        path = self._out_dir / f"bar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart: {path}")
        return path

    # ── Confusion matrix grid ─────────────────────────────────────────────────

    def _plot_confusion_matrices(self, results: Dict, dataset: str) -> Optional[Path]:
        """Plot one normalized 2×2 confusion matrix per loss function.

        Values are derived from averaged Sensitivity (TPR) and Specificity (TNR)
        stored in the results JSON:
          TPR = sensitivity / 100      FNR = 1 − TPR
          TNR = specificity / 100      FPR = 1 − TNR

        Row = Actual class, Col = Predicted class:
          [TN  FP]   actual = Background
          [FN  TP]   actual = Vessel

        Cell colours match the visualization peta kesalahan:
          TP = hijau, TN = hijau muda, FP = merah, FN = biru
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
            figsize=(n_cols * 4.2, n_rows * 4.0),
            gridspec_kw={"hspace": 0.55, "wspace": 0.35},
        )
        axes_flat = np.array(axes).flatten()

        # Base colours (RGB 0–1) — match peta kesalahan in visualization_reporter
        _BASE = {
            "TN": np.array([0.55, 0.88, 0.55]),   # hijau muda
            "FP": np.array([0.92, 0.45, 0.45]),   # merah
            "FN": np.array([0.45, 0.45, 0.92]),   # biru
            "TP": np.array([0.10, 0.68, 0.10]),   # hijau tua
        }
        _CELL_LABEL = [["TN", "FP"], ["FN", "TP"]]

        for idx, (loss_key, label) in enumerate(zip(loss_keys, loss_labels)):
            ax = axes_flat[idx]
            m  = results[loss_key]

            tpr = m.get("sensitivity", 0) / 100.0
            tnr = m.get("specificity", 0) / 100.0
            fnr = 1.0 - tpr
            fpr = 1.0 - tnr

            # vals[row][col]: rows = Actual, cols = Predicted
            vals = [[tnr, fpr],   # actual=Background: TN | FP
                    [fnr, tpr]]   # actual=Vessel:      FN | TP

            # Build RGB image — white blended toward base colour by value intensity
            rgb = np.ones((2, 2, 3), dtype=np.float64)
            for r in range(2):
                for c in range(2):
                    v = vals[r][c]
                    base = _BASE[_CELL_LABEL[r][c]]
                    rgb[r, c] = 1.0 - v * (1.0 - base)   # lerp white → base

            ax.imshow(rgb, interpolation="nearest", aspect="auto",
                      extent=[-0.5, 1.5, 1.5, -0.5])

            # White grid lines between cells
            ax.axhline(0.5, color="white", lw=2.5)
            ax.axvline(0.5, color="white", lw=2.5)

            # Cell text: label + percentage
            for r in range(2):
                for c in range(2):
                    v        = vals[r][c]
                    cell_lbl = _CELL_LABEL[r][c]
                    txt_col  = "white" if v > 0.50 else "black"
                    ax.text(c, r,
                            f"{cell_lbl}\n{v * 100:.2f}%",
                            ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color=txt_col)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred.\nBackground", "Pred.\nVessel"], fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Actual\nBackground", "Actual\nVessel"], fontsize=8)
            ax.tick_params(length=0)
            ax.set_title(label, fontsize=9, fontweight="bold", pad=8)

        # Hide unused subplot panels
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(
            f"Confusion Matrix (Normalized Rate) — {dataset.upper()}\n"
            "Nilai diturunkan dari rata-rata Sensitivity & Specificity per gambar test",
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
