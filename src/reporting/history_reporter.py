from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, StareConfig

_RC = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.linewidth":  0.6,
}


class HistoryReporter:
    """Generate training curve plots from saved history JSON files (Section 4.4).

    When *seeds* is provided, overlays all per-seed curves (thin, semi-transparent)
    on top of the cross-seed mean ± 1 SD band. Falls back to single-seed plot
    when seeds is None or no seed-tagged histories are found.
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        trainer,
        output_dir: Path,
        seeds: Optional[List[int]] = None,
    ):
        self._cfg     = cfg
        self._trainer = trainer
        self._out_dir = output_dir
        self._seeds   = seeds or []

    def generate_all(self) -> List[Path]:
        paths = []
        for loss_key, loss_label in LOSS_FUNCTIONS.items():
            p = self.generate_single(loss_key, loss_label)
            if p:
                paths.append(p)
        return paths

    def generate_single(
        self, loss_key: str, loss_label: Optional[str] = None
    ) -> Optional[Path]:
        label = loss_label or LOSS_FUNCTIONS.get(loss_key, loss_key)

        # Multi-seed path
        if self._seeds:
            histories = []
            for s in self._seeds:
                h = self._trainer.load_history(loss_key, seed_tag=f"seed{s}")
                if h is not None:
                    histories.append(h)
            if len(histories) >= 2:
                return self._plot_multi_seed(loss_key, label, histories)

        # Single-seed fallback
        history = self._trainer.load_history(loss_key)
        if history is None:
            print(f"  [SKIP] History tidak ditemukan untuk '{loss_key}'.")
            return None
        return self._plot(loss_key, label, history)

    # ── Multi-seed plot ───────────────────────────────────────────────────────

    def _plot_multi_seed(
        self, loss_key: str, loss_label: str, histories: list
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib not installed, skipping plot.")
            return None

        train_lists = [h.get("loss", [])     for h in histories]
        val_lists   = [h.get("val_loss", []) for h in histories]

        # Align to minimum length across seeds
        min_t = min((len(t) for t in train_lists if t), default=0)
        min_v = min((len(v) for v in val_lists   if v), default=0)
        if min_t == 0 and min_v == 0:
            return None

        ep_t = list(range(1, min_t + 1))
        ep_v = list(range(1, min_v + 1))

        train_arr = np.array([t[:min_t] for t in train_lists if len(t) >= min_t])
        val_arr   = np.array([v[:min_v] for v in val_lists   if len(v) >= min_v])

        m_train = train_arr.mean(axis=0) if len(train_arr) else np.array([])
        s_train = (train_arr.std(axis=0, ddof=1)
                   if len(train_arr) > 1 else np.zeros(min_t))
        m_val   = val_arr.mean(axis=0)   if len(val_arr)   else np.array([])
        s_val   = (val_arr.std(axis=0, ddof=1)
                   if len(val_arr) > 1 else np.zeros(min_v))

        _C_TRAIN = "#1A5FA8"
        _C_VAL   = "#C0392B"

        with matplotlib.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor("#ffffff")

            # Individual seed curves — thin, semi-transparent
            n = len(histories)
            for t_curve in train_lists:
                if len(t_curve) >= min_t:
                    ax.plot(ep_t, t_curve[:min_t],
                            color=_C_TRAIN, alpha=0.18, lw=0.8)
            for v_curve in val_lists:
                if len(v_curve) >= min_v:
                    ax.plot(ep_v, v_curve[:min_v],
                            color=_C_VAL, alpha=0.18, lw=0.8)

            # Mean ± SD
            if len(m_train):
                ax.plot(ep_t, m_train, color=_C_TRAIN, lw=2.0,
                        label=f"Train Loss (Mean, N={n})")
                ax.fill_between(ep_t,
                                m_train - s_train,
                                m_train + s_train,
                                alpha=0.18, color=_C_TRAIN, label="Train ±1 SD")
            if len(m_val):
                ax.plot(ep_v, m_val, color=_C_VAL, lw=2.0,
                        label=f"Val Loss (Mean, N={n})")
                ax.fill_between(ep_v,
                                m_val - s_val,
                                m_val + s_val,
                                alpha=0.18, color=_C_VAL, label="Val ±1 SD")
                best_ep = int(np.argmin(m_val)) + 1
                best_v  = float(m_val.min())
                ax.axvline(best_ep, color="#7F8C8D", ls="--", lw=1.2,
                           label=f"Best mean epoch: {best_ep} (val={best_v:.5f})")

            # Y-axis zoom to mean range
            all_m = list(m_train) + list(m_val)
            if all_m:
                vmin, vmax = min(all_m), max(all_m)
                span = max(vmax - vmin, 1e-4)
                ax.set_ylim(max(0.0, vmin - span * 0.10), vmax + span * 0.10)

            ax.set_title(
                f"Training History — {loss_label} ({self._cfg.name})\n"
                f"Mean ± SD across {n} seeds"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
            ax.grid(alpha=0.3, linewidth=0.5)

            self._out_dir.mkdir(parents=True, exist_ok=True)
            path = self._out_dir / f"{loss_key}_history_multiseed.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
        print(f"  Multi-seed training curve: {path}")
        return path

    # ── Single-seed plot ──────────────────────────────────────────────────────

    def _plot(self, loss_key: str, loss_label: str, history: dict) -> Path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib not installed, skipping plot.")
            return None

        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        _C_TRAIN = "#1A5FA8"
        _C_VAL   = "#C0392B"

        with matplotlib.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor("#ffffff")
            ax.plot(epochs, train_loss, color=_C_TRAIN, lw=1.8, label="Train Loss")
            ax.plot(epochs, val_loss,   color=_C_VAL,   lw=1.8, label="Val Loss")

            if val_loss:
                best_epoch = val_loss.index(min(val_loss)) + 1
                ax.axvline(best_epoch, color="#7F8C8D", ls="--", lw=1.2,
                           label=f"Best epoch: {best_epoch} (val_loss={min(val_loss):.5f})")

            all_vals = [v for v in train_loss + val_loss if v == v]
            if all_vals:
                v_min = min(all_vals)
                v_max = max(all_vals)
                span  = max(v_max - v_min, 1e-4)
                ax.set_ylim(max(0.0, v_min - span * 0.08), v_max + span * 0.08)

            ax.set_title(f"Training History — {loss_label} ({self._cfg.name})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
            ax.grid(alpha=0.3, linewidth=0.5)

            self._out_dir.mkdir(parents=True, exist_ok=True)
            path = self._out_dir / f"{loss_key}_history.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
        print(f"  Training curve: {path}")
        return path
