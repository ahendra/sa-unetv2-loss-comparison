from pathlib import Path
from typing import List

from config import LOSS_FUNCTIONS, RESULTS_DIR, DriveConfig, StareConfig
from src.tuning import LossTuner


class TuningPlotReporter:
    """Regenerate hyperparameter tuning charts from existing Optuna SQLite DBs.

    Does not retrain or re-tune — reads existing .db files and rewrites the
    3 PNG chart files (_tuning_history, _tuning_importance, _tuning_scatter)
    with the current _save_plots styling (suptitle fixed, 300 DPI).
    """

    _DATASETS = [
        ("drive", DriveConfig),
        ("stare", StareConfig),
    ]

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir  # unused: charts go to RESULTS_DIR/<ds>/tuning/

    def regenerate_all(self) -> List[Path]:
        """Regenerate charts for all loss functions across both datasets."""
        regenerated: List[Path] = []

        for ds_name, cfg_cls in self._DATASETS:
            print(f"\n  ── Dataset: {ds_name.upper()} ──")
            cfg = cfg_cls()
            for loss_key in LOSS_FUNCTIONS:
                print(f"  {loss_key} ... ", end="", flush=True)
                ok = LossTuner.regenerate_plots_from_db(cfg, loss_key)
                if ok:
                    regen_dir = RESULTS_DIR / ds_name / "tuning" / "regenerated"
                    for suffix in (
                        "_tuning_history.png",
                        "_tuning_importance.png",
                        "_tuning_scatter.png",
                    ):
                        p = regen_dir / f"{loss_key}{suffix}"
                        if p.exists():
                            regenerated.append(p)

        return regenerated
