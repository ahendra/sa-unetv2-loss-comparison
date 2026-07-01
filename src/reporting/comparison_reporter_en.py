import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LOSS_FUNCTIONS, RESULTS_DIR
from .palette import LOSS_COLORS


# ── Metrics ───────────────────────────────────────────────────────────────────

_ALL_METRICS     = [
    "accuracy", "f1", "sensitivity", "specificity",
    "auc", "mcc", "jaccard", "cldice",
    "betti0_error", "betti1_error",
]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}
_LO, _HI         = 0.08, 0.92

# English spoke labels — compact enough for 7 pt font on a small polar axis
_SPOKE_LABELS_EN = {
    "accuracy":     "Accuracy",
    "f1":           "F1 Score",
    "sensitivity":  "Sensitivity",
    "specificity":  "Specificity",
    "auc":          "AUC",
    "mcc":          "MCC",
    "jaccard":      "Jaccard",
    "cldice":       "clDice",
    "betti0_error": "β₀ Error\n(↓ better)",
    "betti1_error": "β₁ Error\n(↓ better)",
}

_DATASETS = [("drive", "DRIVE"), ("stare", "STARE")]
_MARKERS  = ["o", "s", "^", "D", "v", "P"]

# A4 two-column full-width: text area = (210 − 13 − 13) mm = 184 mm
_FIG_W_IN = 184 / 25.4   # 7.244 in
_FIG_H_IN = 5.5           # inches — provides clearance for all 10 spoke labels


class ComparisonReporterEN:
    """Combined DRIVE + STARE radar chart in English for journal publication.

    Layout : single figure — DRIVE radar (left) + STARE radar (right),
             shared legend below.
    Size   : A4 two-column full-width = 184 mm (7.244 in) × 5.5 in.
    Font   : Helvetica / Arial; axes titles 9 pt, spoke labels 7 pt,
             radial tick labels 6 pt, legend 7 pt.
    DPI    : 300.
    Output : radar_chart_combined_en.png
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> Optional[Path]:
        """Build and save the combined radar chart. Returns saved path or None."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip radar chart.")
            return None

        all_results = {ds: self._load_results(ds) for ds, _ in _DATASETS}
        if not any(all_results.values()):
            print("  [WARN] Tidak ada hasil evaluasi untuk DRIVE maupun STARE.")
            return None

        with matplotlib.rc_context({
            "font.family":     "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial",
                                "Helvetica Neue", "DejaVu Sans"],
            "font.size":       8,
            "axes.titlesize":  9,
            "axes.labelsize":  8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.2,
            "axes.linewidth":  0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }):
            fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN))
            fig.patch.set_facecolor("#ffffff")

            # Row 0: two polar subplots; Row 1: shared legend strip
            gs = gridspec.GridSpec(
                2, 2,
                figure=fig,
                height_ratios=[5.0, 0.5],
                hspace=0.08,
                wspace=0.28,
                left=0.03, right=0.97,
                top=0.93, bottom=0.03,
            )

            ax_leg = fig.add_subplot(gs[1, :])
            ax_leg.axis("off")

            legend_handles: List = []
            for col_idx, (ds_key, ds_label) in enumerate(_DATASETS):
                ax = fig.add_subplot(gs[0, col_idx], projection="polar")
                ax.set_facecolor("#f8f9fa")
                ds_res = all_results[ds_key]
                if not ds_res:
                    ax.set_title(f"{ds_label}\n(no data available)",
                                 fontsize=9, fontweight="bold", pad=18)
                    continue
                handles = self._draw_radar(ax, ds_res, ds_label)
                if not legend_handles:
                    legend_handles = handles   # use first non-empty dataset

            if legend_handles:
                ax_leg.legend(
                    handles=legend_handles,
                    loc="center", ncol=3,
                    fontsize=7, framealpha=0.95,
                    edgecolor="#dddddd",
                    columnspacing=1.2, handlelength=2.0,
                )

            self._out_dir.mkdir(parents=True, exist_ok=True)
            path = self._out_dir / "radar_chart_combined_en.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Combined radar chart (EN): {path}")
            return path

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_results(self, dataset: str) -> Dict:
        results = {}
        for loss_key in LOSS_FUNCTIONS:
            p = RESULTS_DIR / dataset / f"{loss_key}_results.json"
            if p.exists():
                with open(p) as f:
                    results[loss_key] = json.load(f)
        return results

    def _draw_radar(self, ax, results: Dict, dataset_label: str) -> List:
        """Plot one radar chart onto *ax*. Returns line handles for shared legend."""
        metrics = _ALL_METRICS
        n_m     = len(metrics)
        angles  = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles += angles[:1]

        # Per-metric min-max normalisation → [_LO, _HI]
        metric_stats: Dict[str, tuple] = {}
        for m in metrics:
            vals = [results[k].get(m, 0.0) for k in LOSS_FUNCTIONS if k in results]
            metric_stats[m] = (min(vals), max(vals)) if vals else (0.0, 1.0)

        def _norm(val: float, m: str) -> float:
            lo, hi = metric_stats[m]
            rng = (hi - lo) if (hi - lo) > 1e-9 else 1e-9
            t   = (val - lo) / rng
            if m in _LOWER_IS_BETTER:
                t = 1.0 - t
            return _LO + t * (_HI - _LO)

        handles: List = []
        for idx, (loss_key, loss_label) in enumerate(LOSS_FUNCTIONS.items()):
            if loss_key not in results:
                continue
            color  = LOSS_COLORS.get(loss_key, "#aaaaaa")
            marker = _MARKERS[idx % len(_MARKERS)]
            short  = loss_label.replace(" (Baseline)", "")
            nv     = [_norm(results[loss_key].get(m, 0.0), m) for m in metrics]
            nv    += nv[:1]
            (line,) = ax.plot(
                angles, nv,
                color=color, lw=1.4, ls="-",
                marker=marker, markersize=4,
                markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.5,
                label=short, zorder=3,
            )
            ax.fill(angles, nv, color=color, alpha=0.05, zorder=2)
            handles.append(line)

        # Radial ticks: only label innermost ("Worst") and outermost ("Best")
        r_ticks = np.linspace(_LO, _HI, 5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(r_ticks.tolist())
        ax.set_yticklabels(["Worst", "", "", "", "Best"],
                           fontsize=6, color="#888888")
        ax.yaxis.set_tick_params(pad=4)

        # Spoke labels
        spoke = [_SPOKE_LABELS_EN.get(m, m.upper()) for m in metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spoke, fontsize=7, color="#222222")

        # Grid & spine styling
        ax.grid(color="#cccccc", linestyle="-", linewidth=0.5, alpha=0.8)
        ax.spines["polar"].set_color("#cccccc")
        ax.spines["polar"].set_linewidth(0.5)

        # Dataset title above the radar
        ax.set_title(dataset_label, fontsize=9, fontweight="bold",
                     pad=18, color="#111111")

        return handles
