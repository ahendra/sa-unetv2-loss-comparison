import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LOSS_FUNCTIONS, RESULTS_DIR


_SHORT_LABELS: Dict[str, str] = {
    "bce_mcc":   "BCE+MCC",
    "dice":      "Dice Loss",
    "focal":     "Focal Loss",
    "cldice":    "clDice Loss",
    "dice_ssim": "Dice+SSIM",
    "bce_ssim":  "BCE+SSIM",
}

_DATASETS = [("drive", "DRIVE"), ("stare", "STARE")]

# A4 two-column full-width: text area = (210 − 13 − 13) mm = 184 mm
_FIG_W_IN = 184 / 25.4   # 7.244 in
_FIG_H_IN = 3.80         # 2 subplot rows + legend row

_X_LIM   = (-2.0, 153.0)
_X_TICKS = [0, 50, 100, 150]

_C_TRAIN = "#1A5FA8"   # deep blue  — training loss
_C_VAL   = "#C0392B"   # deep red   — validation loss
_C_BEST  = "#7F8C8D"   # slate gray — best epoch marker


class CombinedHistoryReporter:
    """Generate a single figure with 12 training curves: 6 loss functions × 2 datasets.

    When *seeds* is provided, each subplot overlays all per-seed curves (thin,
    alpha=0.18) plus a thick cross-seed mean with ±1 SD shading band.
    Falls back to single-seed curves when seeds is None.

    Layout : 2 rows (DRIVE, STARE) × 6 columns (one per loss function),
             plus a shared legend row at the bottom.
    Output : 300 dpi PNG sized for full-width two-column A4 paper (184 mm wide).
    """

    def __init__(
        self,
        output_dir: Path,
        seeds: Optional[List[int]] = None,
    ):
        self._out_dir = output_dir
        self._seeds   = seeds or []

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> Optional[Path]:
        """Build and save combined_training_history[_multiseed].png."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib.ticker as ticker
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip plot.")
            return None

        multiseed = bool(self._seeds)

        with matplotlib.rc_context({
                "font.family":        "sans-serif",
                "font.sans-serif":    ["Helvetica", "Arial", "Helvetica Neue",
                                       "DejaVu Sans"],
                "font.size":          8,
                "axes.titlesize":     8,
                "axes.labelsize":     8,
                "xtick.labelsize":    7,
                "ytick.labelsize":    7,
                "legend.fontsize":    7,
                "lines.linewidth":    1.2,
                "axes.linewidth":     0.6,
                "xtick.major.width":  0.6,
                "ytick.major.width":  0.6,
                "xtick.major.size":   2.5,
                "ytick.major.size":   2.5,
            }):

            loss_keys = list(LOSS_FUNCTIONS.keys())   # 6
            n_cols    = len(loss_keys)
            n_rows    = len(_DATASETS)                # 2

            fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN))

            gs = gridspec.GridSpec(
                n_rows + 1, n_cols,
                figure=fig,
                height_ratios=[1.0] * n_rows + [0.20],
                hspace=0.38,
                wspace=0.42,
                left=0.062, right=0.997,
                top=0.91,   bottom=0.02,
            )

            axes = [
                [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                for r in range(n_rows)
            ]

            for row_idx, (ds_key, ds_label) in enumerate(_DATASETS):
                for col_idx, loss_key in enumerate(loss_keys):
                    ax = axes[row_idx][col_idx]

                    if multiseed:
                        histories = self._load_histories_multiseed(ds_key, loss_key)
                        if len(histories) >= 2:
                            self._draw_curves_multiseed(ax, histories)
                        elif len(histories) == 1:
                            self._draw_curves(ax, histories[0])
                        else:
                            ax.text(0.5, 0.5, "—", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=11, color="#bbb")
                    else:
                        history = self._load_history(ds_key, loss_key)
                        if history is None:
                            ax.text(0.5, 0.5, "—", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=11, color="#bbb")
                        else:
                            self._draw_curves(ax, history)

                    ax.set_xlim(_X_LIM)
                    ax.set_xticks(_X_TICKS)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
                    ax.tick_params(axis="both", length=2.5, pad=2)
                    ax.grid(True, alpha=0.2, lw=0.5, color="#888888")
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6)
                        sp.set_color("#888888")

                    if row_idx == 0:
                        ax.set_title(
                            _SHORT_LABELS.get(loss_key, loss_key),
                            pad=3, fontsize=8, fontweight="bold",
                        )

                    if col_idx == 0:
                        ax.text(
                            0.04, 0.96, ds_label,
                            transform=ax.transAxes,
                            fontsize=7.5, fontweight="bold",
                            ha="left", va="top",
                            bbox=dict(boxstyle="square,pad=0.25",
                                      facecolor="white", edgecolor="#aaaaaa",
                                      alpha=0.85, linewidth=0.5),
                            zorder=5,
                        )

            # ── Shared legend row ─────────────────────────────────────────────
            ax_leg = fig.add_subplot(gs[n_rows, :])
            ax_leg.axis("off")

            if multiseed:
                n = len(self._seeds)
                legend_handles = [
                    Line2D([0], [0], color=_C_TRAIN, lw=1.6, ls="-",
                           label=f"Train Mean (N={n})"),
                    Patch(facecolor=_C_TRAIN, alpha=0.20, edgecolor="none",
                          label="Train ± SD"),
                    Line2D([0], [0], color=_C_VAL, lw=1.6, ls="-",
                           label=f"Val Mean (N={n})"),
                    Patch(facecolor=_C_VAL, alpha=0.20, edgecolor="none",
                          label="Val ± SD"),
                    Line2D([0], [0], color=_C_BEST, lw=1.0, ls=":",
                           label="Best Mean Epoch"),
                ]
            else:
                legend_handles = [
                    Line2D([0], [0], color=_C_TRAIN, lw=1.4, ls="-",
                           label="Training Loss"),
                    Line2D([0], [0], color=_C_VAL,   lw=1.4, ls="-",
                           label="Validation Loss"),
                    Line2D([0], [0], color=_C_BEST,  lw=1.0, ls=":",
                           label="Best Epoch"),
                ]

            ax_leg.legend(
                handles=legend_handles,
                loc="upper center", bbox_to_anchor=(0.5, 1.02),
                ncol=len(legend_handles), fontsize=7, framealpha=0.9,
                edgecolor="#cccccc",
                handlelength=2.0, columnspacing=1.2,
            )

            self._out_dir.mkdir(parents=True, exist_ok=True)
            fname = ("combined_training_history_multiseed.png"
                     if multiseed else "combined_training_history.png")
            path = self._out_dir / fname
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            mode = f"multi-seed (N={len(self._seeds)})" if multiseed else "single-seed"
            print(f"  Combined training history ({mode}): {path}")
            return path

    # ── Private: multi-seed ───────────────────────────────────────────────────

    def _load_histories_multiseed(
        self, dataset: str, loss_key: str
    ) -> List[Dict]:
        histories = []
        for s in self._seeds:
            p = (RESULTS_DIR / dataset / "history"
                 / f"{loss_key}_seed{s}_history.json")
            if p.exists():
                with open(p) as f:
                    histories.append(json.load(f))
        return histories

    def _draw_curves_multiseed(self, ax, histories: List[Dict]) -> None:
        """Overlay per-seed thin curves + mean ± SD band on ax."""
        train_lists = [h.get("loss", [])     for h in histories]
        val_lists   = [h.get("val_loss", []) for h in histories]

        min_t = min((len(t) for t in train_lists if t), default=0)
        min_v = min((len(v) for v in val_lists   if v), default=0)
        if min_t == 0 and min_v == 0:
            return

        ep_t = list(range(1, min_t + 1))
        ep_v = list(range(1, min_v + 1))

        train_arr = np.array([t[:min_t] for t in train_lists if len(t) >= min_t])
        val_arr   = np.array([v[:min_v] for v in val_lists   if len(v) >= min_v])

        m_train = train_arr.mean(axis=0) if len(train_arr) else np.array([])
        s_train = (train_arr.std(axis=0, ddof=1)
                   if len(train_arr) > 1 else np.zeros(min_t))
        m_val   = val_arr.mean(axis=0)   if len(val_arr) else np.array([])
        s_val   = (val_arr.std(axis=0, ddof=1)
                   if len(val_arr) > 1 else np.zeros(min_v))

        # Mean curves + SD band
        if len(m_train):
            ax.plot(ep_t, m_train, color=_C_TRAIN, lw=1.4, ls="-")
            ax.fill_between(ep_t,
                            m_train - s_train,
                            m_train + s_train,
                            alpha=0.14, color=_C_TRAIN)
        if len(m_val):
            ax.plot(ep_v, m_val, color=_C_VAL, lw=1.4, ls="-")
            ax.fill_between(ep_v,
                            m_val - s_val,
                            m_val + s_val,
                            alpha=0.14, color=_C_VAL)
            best_ep  = int(np.argmin(m_val)) + 1
            best_val = float(m_val.min())
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")
            ax.text(
                0.97, 0.50,
                f"Best: {best_ep}\n(val={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=6, ha="right", va="center",
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.22",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.90, linewidth=0.4),
                zorder=6,
            )

        # Y-axis: zoom to mean-curve range
        all_m = list(m_train) + list(m_val)
        if all_m:
            vmin, vmax = min(all_m), max(all_m)
            span = max(vmax - vmin, 1e-6)
            ax.set_ylim(max(0.0, vmin - span * 0.06), vmax + span * 0.06)

    # ── Private: single-seed ──────────────────────────────────────────────────

    def _load_history(self, dataset: str, loss_key: str) -> Optional[Dict]:
        path = RESULTS_DIR / dataset / "history" / f"{loss_key}_history.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _draw_curves(self, ax, history: dict) -> None:
        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, color=_C_TRAIN, lw=1.2, ls="-")
        ax.plot(epochs, val_loss,   color=_C_VAL,   lw=1.2, ls="-")

        all_vals = [v for v in train_loss + val_loss if v == v]
        if all_vals:
            vmin, vmax = min(all_vals), max(all_vals)
            span = max(vmax - vmin, 1e-6)
            ax.set_ylim(max(0.0, vmin - span * 0.06), vmax + span * 0.06)

        if val_loss:
            best_ep  = val_loss.index(min(val_loss)) + 1
            best_val = min(val_loss)
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")
            ax.text(
                0.97, 0.50,
                f"Best epoch: {best_ep}\n(val_loss={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=6, ha="right", va="center",
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.22",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.90, linewidth=0.4),
                zorder=6,
            )
