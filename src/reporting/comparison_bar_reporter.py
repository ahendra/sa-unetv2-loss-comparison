import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LOSS_FUNCTIONS, RESULTS_DIR
from .palette import LOSS_COLORS, METRIC_COLORS, METRIC_LABELS


_ALL_METRICS     = [
    "accuracy", "f1", "sensitivity", "specificity",
    "auc", "mcc", "jaccard", "cldice",
    "betti0_error", "betti1_error",
]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}
_Y_AXIS_LABEL    = {"betti0_error": "Error Count", "betti1_error": "Error Count"}

_DATASETS = [("drive", "DRIVE"), ("stare", "STARE")]

_RC = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
    "axes.linewidth":  0.6,
}


class ComparisonBarReporter:
    """Improved grouped bar chart comparison for journal publication.

    Generates bar_chart_drive.png and bar_chart_stare.png with:
      - Larger individual panels (figsize 20 × 7.5·n_rows inches)
      - Tighter inter-panel spacing (wspace 0.28, hspace 0.55)
      - Helvetica / Arial 8 pt throughout
      - Value labels on bars at 8 pt
      - Last chart row horizontally centred
      - 300 DPI output
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> List[Path]:
        """Generate bar charts for DRIVE and STARE. Returns list of saved paths."""
        paths: List[Path] = []
        for ds_key, ds_label in _DATASETS:
            results = self._load_results(ds_key)
            if not results:
                print(f"  [WARN] Tidak ada hasil evaluasi untuk dataset '{ds_key}'.")
                continue
            self._out_dir.mkdir(parents=True, exist_ok=True)
            p = self._plot_bar(results, ds_key, ds_label)
            if p:
                paths.append(p)
        return paths

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_results(self, dataset: str) -> Dict:
        results = {}
        for loss_key in LOSS_FUNCTIONS:
            p = RESULTS_DIR / dataset / f"{loss_key}_results.json"
            if p.exists():
                with open(p) as f:
                    results[loss_key] = json.load(f)
        return results

    def _plot_bar(
        self, results: Dict, dataset: str, dataset_label: str
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip bar chart.")
            return None

        metrics     = _ALL_METRICS
        loss_keys   = [k for k in LOSS_FUNCTIONS if k in results]
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        n_losses    = len(loss_keys)
        colors      = [LOSS_COLORS.get(k, "#aaaaaa") for k in loss_keys]

        n_cols        = 3
        n_metrics     = len(metrics)
        n_rows        = (n_metrics + n_cols - 1) // n_cols
        n_in_last_row = n_metrics - (n_rows - 1) * n_cols
        last_row_off  = (n_cols - n_in_last_row) // 2 if n_in_last_row < n_cols else 0

        with matplotlib.rc_context(_RC):
            fig = plt.figure(figsize=(20, 7.5 * n_rows))
            fig.patch.set_facecolor("#ffffff")

            gs = gridspec.GridSpec(
                n_rows, n_cols,
                figure=fig,
                hspace=0.55,
                wspace=0.28,
                left=0.05, right=0.97,
                top=0.96, bottom=0.06,
            )

            fig.suptitle(
                f"Loss Function Comparison — {dataset_label}",
                fontsize=11, fontweight="bold",
            )

            # Build axes with centering for last row
            axes_list: List = []
            for m_idx in range(n_metrics):
                row = m_idx // n_cols
                if row == n_rows - 1 and n_in_last_row < n_cols:
                    col = last_row_off + (m_idx % n_cols)
                else:
                    col = m_idx % n_cols
                axes_list.append(fig.add_subplot(gs[row, col]))

            for m_idx, mkey in enumerate(metrics):
                ax           = axes_list[m_idx]
                mlabel       = METRIC_LABELS.get(mkey, mkey.upper())
                mcolor       = METRIC_COLORS.get(mkey, "#111111")
                vals         = [results[k].get(mkey, 0) for k in loss_keys]
                lower_better = mkey in _LOWER_IS_BETTER

                v_min = min(vals)
                v_max = max(vals)
                span  = (v_max - v_min) if (v_max - v_min) > 1e-9 else 0.02

                y_min = max(0.0, v_min - span * 0.4)
                y_max = v_max + span * 0.6
                if not lower_better:
                    y_max = min(100.0, y_max)
                if (y_max - y_min) < 0.01:
                    y_min = max(0.0, v_min - 0.05)
                    y_max = v_max + 0.05
                label_offset = (y_max - y_min) * 0.02

                for i, (k, color) in enumerate(zip(loss_keys, colors)):
                    v = results[k].get(mkey, 0)
                    ax.bar(i, v, width=0.65, color=color, alpha=0.85,
                           edgecolor="white", label=loss_labels[i])
                    ax.text(i, v + label_offset, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8, rotation=45)

                ax.set_xticks(range(n_losses))
                ax.set_xticklabels(loss_labels, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(_Y_AXIS_LABEL.get(mkey, "Score (%)"), fontsize=8)
                ax.set_title(mlabel, fontsize=9, fontweight="bold", color=mcolor)
                ax.set_xlim(-0.6, n_losses - 0.4)
                ax.set_ylim(y_min, y_max)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            handles, labels_leg = axes_list[0].get_legend_handles_labels()
            fig.legend(
                handles, labels_leg,
                loc="lower center",
                ncol=min(n_losses, 6),
                bbox_to_anchor=(0.5, 0.01),
                fontsize=8, framealpha=0.9,
                edgecolor="#dddddd",
            )

            path = self._out_dir / f"bar_chart_{dataset}.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Bar chart ({dataset_label}): {path}")
            return path
