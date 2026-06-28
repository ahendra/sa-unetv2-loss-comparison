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
_FIG_H_IN = 3.80         # 2 subplot rows + legend row

# Fixed axis ranges — same across all 12 subplots for fair visual comparison
_X_LIM   = (-2.0, 153.0)      # slight padding beyond [0, 150]
_X_TICKS = [0, 50, 100, 150]
_Y_LIM   = (0.145, 0.855)     # padding so 0.2 and 0.8 ticks sit clearly inside
_Y_TICKS = [0.2, 0.4, 0.6, 0.8]

_C_TRAIN = "#1A5FA8"   # deep blue  — training loss
_C_VAL   = "#C0392B"   # deep red   — validation loss
_C_BEST  = "#7F8C8D"   # slate gray — best epoch marker


class CombinedHistoryReporter:
    """Generate a single figure with 12 training curves: 6 loss functions × 2 datasets.

    Layout : 2 rows (DRIVE, STARE) × 6 columns (one per loss function),
             plus a shared legend row at the bottom.
    Output : 300 dpi PNG sized for full-width two-column A4 paper (184 mm wide).
    Font   : Helvetica / Arial / sans-serif, 8 pt.
    Scales : Fixed shared x [0–150] and y [0.2–0.8] for fair cross-subplot comparison.
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

            # 2 subplot rows + 1 legend row (0.32 relative height to hold 2
            # lines of content: legend handles + axis-name description).
            # Y-tick labels on col 0 only → wspace can be tight (0.12).
            # No axis labels on subplots → left margin reduced to 0.062.
            gs = gridspec.GridSpec(
                n_rows + 1, n_cols,
                figure=fig,
                height_ratios=[1.0] * n_rows + [0.32],
                hspace=0.38,
                wspace=0.12,
                left=0.062, right=0.997,
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
                    else:
                        self._draw_curves(ax, history)

                    # ── Fixed shared scales (applied to every subplot) ────────
                    ax.set_xlim(_X_LIM)
                    ax.set_xticks(_X_TICKS)
                    ax.set_ylim(_Y_LIM)
                    ax.set_yticks(_Y_TICKS)
                    ax.yaxis.set_major_formatter(
                        ticker.FormatStrFormatter("%.1f")
                    )
                    ax.tick_params(axis="both", length=2.5, pad=2)
                    ax.grid(True, alpha=0.2, lw=0.5, color="#888888")
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6)
                        sp.set_color("#888888")

                    # Column header: top row only
                    if row_idx == 0:
                        ax.set_title(
                            _SHORT_LABELS.get(loss_key, loss_key),
                            pad=3, fontsize=8, fontweight="bold",
                        )

                    # X-tick labels: bottom row only
                    if row_idx != n_rows - 1:
                        plt.setp(ax.get_xticklabels(), visible=False)

                    # Y-tick labels: leftmost column only — saves wspace
                    if col_idx != 0:
                        plt.setp(ax.get_yticklabels(), visible=False)

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
            # Lower part: axis-name description (replaces subplot axis labels)
            ax_leg.text(
                0.5, 0.07,
                "x-axis: Epoch  |  y-axis: Loss",
                transform=ax_leg.transAxes,
                ha="center", va="center",
                fontsize=6.5, color="#444444", style="italic",
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

    def _draw_curves(self, ax, history: dict) -> None:
        """Plot train/val curves and best-epoch marker. Axis scaling done by caller."""
        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, color=_C_TRAIN, lw=1.2, ls="-")
        ax.plot(epochs, val_loss,   color=_C_VAL,   lw=1.2, ls="-")

        if val_loss:
            best_ep  = val_loss.index(min(val_loss)) + 1
            best_val = min(val_loss)
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")

            # Best-epoch annotation at (0.97, 0.93) so the bbox top stays
            # safely inside the axes top spine; fontsize=5 keeps it compact.
            ax.text(
                0.97, 0.93,
                f"Best epoch: {best_ep}\n(val_loss={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=5, ha="right", va="top",
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.22",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.90, linewidth=0.4),
                zorder=6,
            )
