import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.0,
    "axes.linewidth":  0.6,
}


def _ms_extract(summary: Dict) -> Tuple[Dict, Dict, int]:
    """Extract (vals, stds, n_seeds) from a multiseed summary dict."""
    s = summary.get("summary", {})
    vals = {lk: {m: info["mean"] for m, info in pm.items() if isinstance(info, dict)}
            for lk, pm in s.items()}
    stds = {lk: {m: info["std"]  for m, info in pm.items() if isinstance(info, dict)}
            for lk, pm in s.items()}
    n = summary.get("n_seeds", len(summary.get("seeds", [])))
    return vals, stds, n


class ComparisonBarReporter:
    """Grouped bar chart comparison for journal publication.

    When *multiseed_summaries* is provided, bar heights are per-loss means
    across all seeds and black error bars show ±1 SD (N=5 seeds). Without
    multiseed data, falls back to single-seed scalar values.

    Parameters
    ----------
    output_dir         : directory for output PNG files
    multiseed_summaries: optional {dataset_key: multiseed_summary_dict}
                         loaded from multiseed_summary.json (MultiSeedReporter).
    """

    def __init__(
        self,
        output_dir: Path,
        multiseed_summaries: Optional[Dict[str, Dict]] = None,
    ):
        self._out_dir = output_dir
        self._ms      = multiseed_summaries or {}

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> List[Path]:
        """Generate bar charts for DRIVE and STARE. Returns list of saved paths."""
        paths: List[Path] = []
        for ds_key, ds_label in _DATASETS:
            vals, stds, n_seeds = self._resolve_data(ds_key)
            if not vals:
                print(f"  [WARN] Tidak ada hasil evaluasi untuk dataset '{ds_key}'.")
                continue
            self._out_dir.mkdir(parents=True, exist_ok=True)
            p = self._plot_bar(vals, stds, n_seeds, ds_key, ds_label)
            if p:
                paths.append(p)
        return paths

    # ── Private ──────────────────────────────────────────────────────────────

    def _resolve_data(
        self, dataset: str
    ) -> Tuple[Dict, Optional[Dict], Optional[int]]:
        """Return (vals, stds, n_seeds). stds/n_seeds are None for single-seed."""
        ms = self._ms.get(dataset)
        if ms:
            vals, stds, n_seeds = _ms_extract(ms)
            return vals, stds, n_seeds
        vals = self._load_results(dataset)
        return vals, None, None

    def _load_results(self, dataset: str) -> Dict:
        results = {}
        for loss_key in LOSS_FUNCTIONS:
            p = RESULTS_DIR / dataset / f"{loss_key}_results.json"
            if p.exists():
                with open(p) as f:
                    results[loss_key] = json.load(f)
        return results

    def _plot_bar(
        self,
        results:     Dict,
        stds:        Optional[Dict],
        n_seeds:     Optional[int],
        dataset:     str,
        dataset_label: str,
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
        multiseed   = stds is not None

        n_cols        = 3
        n_metrics     = len(metrics)
        n_rows        = (n_metrics + n_cols - 1) // n_cols
        n_in_last_row = n_metrics - (n_rows - 1) * n_cols
        last_row_off  = (n_cols - n_in_last_row) // 2 if n_in_last_row < n_cols else 0

        subtitle = (f"Mean ± SD  (N={n_seeds} seeds)"
                    if multiseed else "Single-seed evaluation")

        with matplotlib.rc_context(_RC):
            fig = plt.figure(figsize=(14, 5.5 * n_rows))
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
                f"Loss Function Comparison — {dataset_label}\n{subtitle}",
                fontsize=13, fontweight="bold",
            )

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
                errs         = ([stds[k].get(mkey, 0) for k in loss_keys]
                                if multiseed else None)
                lower_better = mkey in _LOWER_IS_BETTER

                # Compute y range accounting for error bars
                if errs:
                    effective_max = max(v + e for v, e in zip(vals, errs))
                    effective_min = min(v - e for v, e in zip(vals, errs))
                else:
                    effective_max = max(vals)
                    effective_min = min(vals)

                span  = (effective_max - effective_min) if (effective_max - effective_min) > 1e-9 else 0.02
                y_min = max(0.0, effective_min - span * 0.35)
                y_max = effective_max + span * 0.65
                if not lower_better:
                    y_max = min(100.0, y_max)
                if (y_max - y_min) < 0.01:
                    y_min = max(0.0, effective_min - 0.05)
                    y_max = effective_max + 0.05
                label_offset = (y_max - y_min) * 0.02

                for i, (k, color) in enumerate(zip(loss_keys, colors)):
                    v   = results[k].get(mkey, 0)
                    err = errs[i] if errs else None
                    ax.bar(i, v, width=0.65, color=color, alpha=0.85,
                           edgecolor="white", label=loss_labels[i])
                    if err is not None:
                        ax.errorbar(i, v, yerr=err, fmt="none",
                                    color="#333333", capsize=5, capthick=1.3,
                                    lw=1.5, zorder=5)
                    top = (v + err + label_offset) if err is not None else (v + label_offset)
                    txt = f"{v:.2f}\n±{err:.2f}" if err is not None else f"{v:.2f}"
                    ax.text(i, top, txt,
                            ha="center", va="bottom", fontsize=10, rotation=0)

                ax.set_xticks(range(n_losses))
                ax.set_xticklabels(loss_labels, rotation=45, ha="right",
                                   rotation_mode="anchor", fontsize=12)
                ax.set_ylabel(_Y_AXIS_LABEL.get(mkey, "Score (%)"), fontsize=12)
                ax.set_title(mlabel, fontsize=13, fontweight="bold", color=mcolor)
                ax.set_xlim(-0.6, n_losses - 0.4)
                ax.set_ylim(y_min, y_max)
                ax.tick_params(axis="y", labelsize=12)
                ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            handles, labels_leg = axes_list[0].get_legend_handles_labels()
            fig.legend(
                handles, labels_leg,
                loc="lower center",
                ncol=min(n_losses, 6),
                bbox_to_anchor=(0.5, 0.01),
                fontsize=12, framealpha=0.9,
                edgecolor="#dddddd",
            )

            path = self._out_dir / f"bar_chart_{dataset}.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            mode = "multi-seed Mean±SD" if multiseed else "single-seed"
            print(f"  Bar chart ({dataset_label}, {mode}): {path}")
            return path
