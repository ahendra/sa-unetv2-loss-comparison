import json
from pathlib import Path
from typing import Dict, Optional

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
_FIG_H_IN = 5.20         # 2 subplot rows + legend row (taller for fluctuation visibility)

# Fixed x-axis — same across all subplots for fair epoch-count comparison.
# Y-axis is auto-scaled per subplot: different loss functions operate on
# fundamentally different value ranges (e.g. Focal ~0.01–0.14 vs BCE ~0.15–0.82)
# so a shared y-range would hide entire curves or clip most of the data.
_X_LIM   = (-2.0, 153.0)
_X_TICKS = [0, 50, 100, 150]

_C_TRAIN = "#1A5FA8"   # deep blue  — training loss
_C_VAL   = "#C0392B"   # deep red   — validation loss
_C_BEST  = "#7F8C8D"   # slate gray — best epoch marker


class CombinedHistoryReporter:
    """Generate a single figure with 12 training curves: 6 loss functions × 2 datasets.

    Layout : 2 rows (DRIVE, STARE) × 6 columns (one per loss function),
             plus a shared legend row at the bottom.
    Output : 600 dpi PNG sized for full-width two-column A4 paper (184 mm wide).
    Font   : Helvetica / Arial / sans-serif, 8 pt.
    Scales : Fixed x-axis [0–150] for fair epoch comparison; y-axis auto-scaled
             per subplot (loss value ranges differ fundamentally across functions).
             Y-tick labels shown on col 0 only; x-tick labels on bottom row only.
             Axis names moved to the legend row to free subplot area.
    Note   : No figure-level title; add a numbered caption in the paper body.
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> Optional[Path]:
        """Build and save combined_training_history.png. Returns path or None."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib.ticker as ticker
            from matplotlib.lines import Line2D
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip plot.")
            return None

        with matplotlib.rc_context():
            plt.rcParams.update({
                "font.family":        "sans-serif",
                "font.sans-serif":    ["Helvetica", "Arial", "Helvetica Neue",
                                       "DejaVu Sans"],
                "font.size":          8,
                "axes.titlesize":     8,
                "axes.labelsize":     8,
                "xtick.labelsize":    6.5,
                "ytick.labelsize":    6.5,
                "legend.fontsize":    7,
                "lines.linewidth":    1.2,
                "axes.linewidth":     0.6,
                "xtick.major.width":  0.6,
                "ytick.major.width":  0.6,
                "xtick.major.size":   2.5,
                "ytick.major.size":   2.5,
            })

            loss_keys = list(LOSS_FUNCTIONS.keys())   # 6
            n_cols    = len(loss_keys)
            n_rows    = len(_DATASETS)                # 2

            fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN))

            # 2 subplot rows + 1 legend row (0.32 relative height to hold 2
            # lines of content: legend handles + axis-name description).
            # Y-tick labels on col 0 only → wspace can be tight (0.12).
            # No axis labels on subplots → left margin reduced to 0.062.
            gs = gridspec.GridSpec(
                n_rows + 1, n_cols,
                figure=fig,
                height_ratios=[1.0] * n_rows + [0.16],
                hspace=0.42,
                wspace=0.55,
                left=0.095, right=0.997,
                top=0.91,   bottom=0.04,
            )

            axes = [
                [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                for r in range(n_rows)
            ]

            for row_idx, (ds_key, ds_label) in enumerate(_DATASETS):
                for col_idx, loss_key in enumerate(loss_keys):
                    ax      = axes[row_idx][col_idx]
                    history = self._load_history(ds_key, loss_key)

                    if history is None:
                        ax.text(0.5, 0.5, "—", ha="center", va="center",
                                transform=ax.transAxes, fontsize=11, color="#bbb")
                    else:
                        self._draw_curves(ax, history)

                    # ── Axis scales ───────────────────────────────────────────
                    ax.set_xlim(_X_LIM)
                    ax.set_xticks(_X_TICKS)

                    # 4 Y-ticks per subplot — enough detail without label crowding
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))

                    # Max 3 decimal places; use 2 only when the range is large (≥ 0.5)
                    y_lo, y_hi = ax.get_ylim()
                    y_range = y_hi - y_lo if y_hi != y_lo else 1e-6
                    y_fmt = "%.2f" if y_range >= 0.5 else "%.3f"
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

                    ax.tick_params(axis="both", length=2.5, pad=2)
                    ax.grid(True, alpha=0.25, lw=0.5, color="#888888")
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6)
                        sp.set_color("#888888")

                    # Axis labels: x on both rows, y on leftmost column only
                    ax.set_xlabel("Epoch", fontsize=7.5, labelpad=3)
                    if col_idx == 0:
                        ax.set_ylabel("Loss", fontsize=7.5, labelpad=3)

                    # Column header: top row only
                    if row_idx == 0:
                        ax.set_title(
                            _SHORT_LABELS.get(loss_key, loss_key),
                            pad=3, fontsize=8, fontweight="bold",
                        )

                    # Dataset label inside subplot, upper-left box (col 0 only)
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

            # Upper part: line-style legend handles
            ax_leg.legend(
                handles=[
                    Line2D([0], [0], color=_C_TRAIN, lw=1.4, ls="-",
                           label="Training Loss"),
                    Line2D([0], [0], color=_C_VAL,   lw=1.4, ls="-",
                           label="Validation Loss"),
                    Line2D([0], [0], color=_C_BEST,  lw=1.0, ls=":",
                           label="Best Epoch"),
                ],
                loc="upper center", bbox_to_anchor=(0.5, 1.02),
                ncol=3, fontsize=7, framealpha=0.9,
                edgecolor="#cccccc",
                handlelength=2.0, columnspacing=1.2,
            )

            self._out_dir.mkdir(parents=True, exist_ok=True)
            path = self._out_dir / "combined_training_history.png"
            fig.savefig(str(path), dpi=600, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Combined training history: {path}")
            return path

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_history(self, dataset: str, loss_key: str) -> Optional[Dict]:
        path = RESULTS_DIR / dataset / "history" / f"{loss_key}_history.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _draw_curves(self, ax, history: dict) -> None:
        """Plot train/val curves, best-epoch marker, and annotation.

        Also sets the y-axis limits from the actual data so each loss function
        is shown at its native scale (loss value ranges differ fundamentally
        across functions and cannot share a meaningful common y-axis).
        """
        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, color=_C_TRAIN, lw=1.0, ls="-")
        ax.plot(epochs, val_loss,   color=_C_VAL,   lw=1.0, ls="-")

        # Y-range calibrated to the settled/convergence phase (last 60% of epochs)
        # so epoch-to-epoch fluctuations occupy most of the vertical space.
        # The steep initial drop (first 40%) may render above the top limit —
        # matplotlib clips it cleanly without distorting the convergence region.
        n = len(train_loss)
        tail_start = max(1, int(n * 0.25))
        tail_vals = [v for v in (train_loss[tail_start:] + val_loss[tail_start:]) if v == v]
        ref_vals  = tail_vals if tail_vals else [v for v in train_loss + val_loss if v == v]
        if ref_vals:
            vmin, vmax = min(ref_vals), max(ref_vals)
            span = max(vmax - vmin, 1e-6)
            ax.set_ylim(max(0.0, vmin - span * 0.05), vmax + span * 0.08)

        if val_loss:
            best_ep  = val_loss.index(min(val_loss)) + 1
            best_val = min(val_loss)
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")

            # Best-epoch annotation — placed at vertical centre (0.50) to
            # avoid overlap with the DRIVE/STARE label in the upper-left.
            ax.text(
                0.97, 0.50,
                f"Best epoch: {best_ep}\n(val_loss={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=5, ha="right", va="center",
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.22",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.90, linewidth=0.4),
                zorder=6,
            )
