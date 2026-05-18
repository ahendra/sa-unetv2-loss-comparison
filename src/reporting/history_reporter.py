from pathlib import Path
from typing import List, Optional, Union

from config import DriveConfig, LOSS_FUNCTIONS, StareConfig


class HistoryReporter:
    """Generate training curve plots from saved history JSON files (Section 4.4).

    Single Responsibility: read history JSON written by ModelTrainer and
    produce loss/val_loss curve PNG files.
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        trainer,
        output_dir: Path,
    ):
        self._cfg     = cfg
        self._trainer = trainer
        self._out_dir = output_dir

    def generate_all(self) -> List[Path]:
        """Generate curve PNG for every loss function that has a history file."""
        paths = []
        for loss_key, loss_label in LOSS_FUNCTIONS.items():
            p = self.generate_single(loss_key, loss_label)
            if p:
                paths.append(p)
        return paths

    def generate_single(self, loss_key: str, loss_label: Optional[str] = None) -> Optional[Path]:
        history = self._trainer.load_history(loss_key)
        if history is None:
            print(f"  [SKIP] History tidak ditemukan untuk '{loss_key}'.")
            return None
        label = loss_label or LOSS_FUNCTIONS.get(loss_key, loss_key)
        return self._plot(loss_key, label, history)

    # ── Private ──────────────────────────────────────────────────────────────

    def _plot(self, loss_key: str, loss_label: str, history: dict) -> Path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip plot.")
            return None

        train_loss = history.get("loss", [])
        val_loss   = history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_loss, "b-",  lw=1.5, label="Train Loss")
        ax.plot(epochs, val_loss,   "r-",  lw=1.5, label="Val Loss")

        if val_loss:
            best_epoch = val_loss.index(min(val_loss)) + 1
            ax.axvline(best_epoch, color="gray", ls="--", lw=1.2,
                       label=f"Best epoch: {best_epoch} (val_loss={min(val_loss):.5f})")

        # Zoom y-axis to the actual loss range (exclude initial spike if extreme)
        all_vals = [v for v in train_loss + val_loss if v == v]
        if all_vals:
            v_min = min(all_vals)
            v_max = max(all_vals)
            span  = max(v_max - v_min, 1e-4)
            ax.set_ylim(max(0.0, v_min - span * 0.08), v_max + span * 0.08)

        ax.set_title(f"Training History — {loss_label} ({self._cfg.name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        # Legend placed just outside the right edge of the axes — no overlap with curves.
        # bbox_inches="tight" captures it in the saved PNG.
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0, framealpha=0.9)
        ax.grid(alpha=0.3)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / f"{loss_key}_history.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Kurva training: {path}")
        return path
