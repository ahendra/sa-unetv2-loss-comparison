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
    "bce_ssim":  "Hybrid (Area+Texture)",
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
        paths["radar"]   = self._plot_radar(results, dataset)
        paths["bar"]     = self._plot_bar(results, dataset)
        paths["ranking"] = self._save_ranking(results, dataset)
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

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
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

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, (loss_key, label, color) in enumerate(zip(loss_keys, loss_labels, colors)):
            vals = [results[loss_key].get(m, 0) for m in metrics]
            offset = (i - n_losses / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title(
            f"Loss Function Comparison — {dataset.upper()}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

        path = self._out_dir / f"bar_chart_{dataset}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart: {path}")
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
                    "value":      round(v, 4) if v == v else None,
                }
                for i, (k, v) in enumerate(items)
            ]

        path = self._out_dir / f"ranking_table_{dataset}.json"
        with open(path, "w") as f:
            json.dump(ranking, f, indent=2)
        print(f"  Ranking table: {path}")
        return path
