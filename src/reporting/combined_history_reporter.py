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
_FIG_H_IN = 3.80         # balanced for 2 subplot rows + legend row

_C_TRAIN = "#1A5FA8"   # deep blue  — training loss
_C_VAL   = "#C0392B"   # deep red   — validation loss
_C_BEST  = "#7F8C8D"   # slate gray — best epoch marker


class CombinedHistoryReporter:
    """Generate a single figure with 12 training curves: 6 loss functions × 2 datasets.

    Layout : 2 rows (DRIVE, STARE) × 6 columns (one per loss function),
             plus a shared legend row at the bottom.
    Output : 300 dpi PNG sized for full-width two-column A4 paper (184 mm wide).
    Font   : Helvetica / Arial / sans-serif, 8 pt throughout.
    Note   : No figure-level title; insert a numbered caption in the paper body.
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
                "xtick.labelsize":    7,
                "ytick.labelsize":    7,
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

            # 2 subplot rows + 1 thin legend row.
            # left=0.100 provides enough room for the shared "Loss" fig.text +
            # y-axis tick labels without competing row-label annotations.
            gs = gridspec.GridSpec(
                n_rows + 1, n_cols,
                figure=fig,
                height_ratios=[1.0] * n_rows + [0.22],
                hspace=0.52,
                wspace=0.42,
                left=0.100, right=0.997,
                top=0.91,   bottom=0.02,
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
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_color("#cccccc")
                            sp.set_linewidth(0.4)
                    else:
                        self._draw_curves(ax, history, ticker)

                    # Column header: top subplot row only
                    if row_idx == 0:
                        ax.set_title(
                            _SHORT_LABELS.get(loss_key, loss_key),
                            pad=3, fontsize=8, fontweight="bold",
                        )
                    # x-axis label: bottom subplot row only
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("Epoch", labelpad=2)
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)

                    # Dataset label inside the first column subplot (upper-left box).
                    # Placed inside the axes to avoid competing with the y-axis area.
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

            # Shared "Loss" y-axis label — single rotated text on the figure left,
            # vertically centred over both subplot rows.  x=0.008 sits between the
            # figure edge and the tick labels (which start at ≈ x=0.055).
            fig.text(
                0.008, 0.53, "Loss",
                rotation=90, va="center", ha="center",
                fontsize=8,
            )

            # Shared legend row spanning all columns
            ax_leg = fig.add_subplot(gs[n_rows, :])
            ax_leg.axis("off")
            ax_leg.legend(
                handles=[
                    Line2D([0], [0], color=_C_TRAIN, lw=1.4, ls="-",
                           label="Training Loss"),
                    Line2D([0], [0], color=_C_VAL,   lw=1.4, ls="-",
                           label="Validation Loss"),
                    Line2D([0], [0], color=_C_BEST,  lw=1.0, ls=":",
                           label="Best Epoch"),
                ],
                loc="center", ncol=3,
                fontsize=7, framealpha=0.9,
                edgecolor="#cccccc",
                handlelength=2.0, columnspacing=1.2,
            )

            self._out_dir.mkdir(parents=True, exist_ok=True)
            path = self._out_dir / "combined_training_history.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
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

    def _draw_curves(self, ax, history: dict, ticker) -> None:
        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, color=_C_TRAIN, lw=1.2, ls="-")
        ax.plot(epochs, val_loss,   color=_C_VAL,   lw=1.2, ls="-")

        if val_loss:
            best_ep  = val_loss.index(min(val_loss)) + 1
            best_val = min(val_loss)
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")

            # Best epoch annotation — top-right corner with white background box
            ax.text(
                0.97, 0.97,
                f"Best epoch: {best_ep}\n(val_loss={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=6, ha="right", va="top",
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.25",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.88, linewidth=0.4),
                zorder=6,
            )

        all_vals = [v for v in train_loss + val_loss if v == v]
        if all_vals:
            vmin, vmax = min(all_vals), max(all_vals)
            span = max(vmax - vmin, 1e-4)
            ax.set_ylim(max(0.0, vmin - span * 0.06), vmax + span * 0.06)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="both", length=2.5, pad=2)
        ax.grid(True, alpha=0.2, lw=0.5, color="#888888")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#888888")
