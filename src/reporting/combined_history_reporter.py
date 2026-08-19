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
                        self._draw_curves(ax, history, col_idx)

                    # ── Axis scales ───────────────────────────────────────────
                    ax.set_xlim(_X_LIM)
                    ax.set_xticks(_X_TICKS)

                    # Fixed ticks — same on every subplot now that Y is normalized [0,1]
                    ax.yaxis.set_major_locator(
                        ticker.FixedLocator([0.00, 0.25, 0.50, 0.75, 1.00])
                    )
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

                    ax.tick_params(axis="both", length=2.5, pad=2)
                    ax.grid(True, alpha=0.25, lw=0.5, color="#888888")
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6)
                        sp.set_color("#888888")

                    # Axis labels: x on both rows, y on leftmost column only
                    ax.set_xlabel("Epoch", fontsize=7.5, labelpad=3)
                    if col_idx == 0:
                        ax.set_ylabel("Normalized Loss", fontsize=7.5, labelpad=3)

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

    def _draw_curves(self, ax, history: dict, col_idx: int = 1) -> None:
        """Plot train/val curves, best-epoch marker, and annotation.

        col_idx is used to avoid overlapping the Best-epoch box with the
        DRIVE/STARE dataset label that sits in the upper-left of column 0.
        """
        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        def _minmax(data: list) -> list:
            lo, hi = min(data), max(data)
            span = max(hi - lo, 1e-9)
            return [(v - lo) / span for v in data]

        # Normalize each curve independently to [0, 1] so all 12 subplots
        # share the same scale — convergence dynamics are directly comparable.
        train_norm = _minmax(train_loss) if len(train_loss) > 1 else train_loss
        val_norm   = _minmax(val_loss)   if len(val_loss)   > 1 else val_loss

        ax.plot(epochs, train_norm, color=_C_TRAIN, lw=1.0, ls="-")
        ax.plot(epochs, val_norm,   color=_C_VAL,   lw=1.0, ls="-")

        # Fixed Y range for all subplots — slight padding so curves don't
        # touch the top/bottom spine.
        ax.set_ylim(-0.04, 1.08)

        if val_loss:
            best_ep  = val_loss.index(min(val_loss)) + 1
            best_val = min(val_loss)          # original value for annotation
            ax.axvline(best_ep, color=_C_BEST, lw=0.9, ls=":")

            # Best-epoch annotation.
            # Col 0: DRIVE/STARE label at upper-left → place box at center-right.
            # Col 1+: no conflicting label → place box at upper-right.
            if col_idx == 0:
                ann_y, ann_va = 0.50, "center"
            else:
                ann_y, ann_va = 0.96, "top"

            ax.text(
                0.97, ann_y,
                f"Best: ep.{best_ep}\n(val={best_val:.5f})",
                transform=ax.transAxes,
                fontsize=5, ha="right", va=ann_va,
                color="#222222",
                bbox=dict(boxstyle="square,pad=0.22",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.90, linewidth=0.4),
                zorder=6,
            )
