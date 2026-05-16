import gc
import json
import time
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.metrics import f1_score

from config import DriveConfig, StareConfig, RESULTS_DIR

# Early-stopping patience inside each trial (separate from main training)
_TRIAL_PATIENCE = 7

# ── Parameter search spaces ───────────────────────────────────────────────────
#
# Setiap range didasarkan pada literatur ilmiah sebagai berikut:
#
# [1] Lin et al. (2017) "Focal Loss for Dense Object Detection." ICCV.
#     → Focal alpha default 0.25, tested {0.25, 0.5, 0.75}; gamma default 2.0,
#       tested {0, 0.5, 1, 2, 5}.
#
# [2] Shit et al. (2021) "clDice – A Novel Topology-Preserving Loss Function
#     for Tubular Structure Segmentation." CVPR.
#     → alpha=0.5 default; "alpha is a per-dataset tunable hyperparameter";
#       smooth=1.0 hardcoded; iters=10 for 2D experiments.
#
# [3] Yeung et al. (2022) "Unified Focal Loss: Generalising Dice and
#     Cross Entropy-based Losses to Handle Class Imbalanced Medical Image
#     Segmentation." CMIG, 46, 101979.
#     → Lambda weight default 0.5 (balanced); delta in (0,1);
#       gamma range [0.1, 0.9] for 2D, fixed 0.5 for 3D.
#
# [4] Bertels et al. (2022) "The Dice Loss in the Context of Missing or
#     Empty Labels: Introducing Φ and ε." MICCAI, LNCS 13438.
#     → Smooth ε analyzed in range [1e-7, 1.0].
#
# [5] Jadon et al. (2020) "A Survey of Loss Functions for Semantic
#     Segmentation." IEEE BIBM.
#     → Combined losses (BCE+Dice, Dice+SSIM, etc.) typically weighted
#       in [0.3, 0.7] for medical segmentation.
#
# [6] Ni et al. (2025) "Retinal Vascular Segmentation Network Based on
#     Dual-scale Morphological Enhancement." Springer, DOI 10.1007/s44443-025-00191-3.
#     → BCE+SSIM: alpha=beta=0.5.

SEARCH_SPACES = {
    # ── BCE + MCC ────────────────────────────────────────────────────────────
    # lambda_bce ∈ [0.3, 0.7]:
    #   SA-UNetV2 original paper menggunakan 0.5/0.5 sebagai baseline.
    #   Yeung et al. [3] merekomendasikan lambda=0.5 sebagai titik awal
    #   untuk combined losses; Jadon et al. [5] menunjukkan range [0.3, 0.7]
    #   adalah tipikal untuk medical image segmentation.
    "bce_mcc": {
        "lambda_bce": ("float", 0.3, 0.7),
    },

    # ── Dice Loss ────────────────────────────────────────────────────────────
    # smooth ∈ [1e-7, 1.0] (log scale):
    #   Bertels et al. [4] menganalisis epsilon dalam range ini secara
    #   sistematis. SMP (segmentation_models_pytorch) menggunakan 1e-7
    #   sebagai default; implementasi umum memakai 1e-6 hingga 1.0.
    "dice": {
        "smooth": ("float_log", 1e-7, 1.0),
    },

    # ── Focal Loss ───────────────────────────────────────────────────────────
    # alpha ∈ [0.1, 0.9]:
    #   Lin et al. [1] menguji alpha ∈ {0.25, 0.5, 0.75}; default 0.25.
    #   Yeung et al. [3] merekomendasikan alpha ∈ [0.25, 0.75] untuk
    #   medical imaging dengan class imbalance; range diperlebar ke [0.1, 0.9]
    #   untuk eksplorasi komprehensif pada retinal vessel (imbalance ~10%).
    #
    # gamma ∈ [0.5, 5.0]:
    #   Lin et al. [1] menguji gamma ∈ {0, 0.5, 1, 2, 5}; default gamma=2.0
    #   memberikan hasil terbaik. Yeung et al. [3] menggunakan gamma ∈ [0.1, 0.9]
    #   untuk 2D (dengan delta terpisah). Range [0.5, 5.0] mencakup semua
    #   nilai yang diuji Lin et al. (menghindari gamma=0 yang setara BCE biasa).
    #
    # smooth ∈ [1e-7, 1e-3] (log scale):
    #   Digunakan untuk clipping prediksi (menghindari log(0)).
    #   Praktik umum dalam implementasi focal loss.
    "focal": {
        "alpha":  ("float", 0.1, 0.9),
        "gamma":  ("float", 0.5, 5.0),
        "smooth": ("float_log", 1e-7, 1e-3),
    },

    # ── clDice Loss ──────────────────────────────────────────────────────────
    # alpha ∈ [0.3, 0.7]:
    #   Shit et al. [2] menggunakan alpha=0.5 sebagai default dan menyatakan
    #   "alpha is a per-dataset tunable hyperparameter." Paper juga menyatakan
    #   "increasing alpha generally improves the clDice measure."
    #   Range [0.3, 0.7] mencakup nilai default dan variasi di sekitarnya.
    #
    # iters ∈ [5, 30]:
    #   Shit et al. [2] menggunakan iter_=10 untuk eksperimen 2D (Table 4).
    #   Paper asli menggunakan iter_=50 untuk volume 3D. Ablation study
    #   menunjukkan konvergensi sekitar k=20-30 untuk gambar 2D; perbedaan
    #   marginal di atas k=30. Range [5, 30] mencakup nilai eksperimental
    #   untuk retinal vessel segmentation 2D.
    #
    # smooth ∈ [1e-7, 1.0] (log scale):
    #   Shit et al. [2] menghardcode smooth=1.0, namun Bertels et al. [4]
    #   menganalisis epsilon dalam range [1e-7, 1.0] secara sistematis.
    #   Di-tune untuk menemukan nilai optimal per dataset.
    "cldice": {
        "alpha":  ("float",     0.3,  0.7),
        "iters":  ("int",       5,    30),
        "smooth": ("float_log", 1e-7, 1.0),
    },

    # ── Dice + SSIM ──────────────────────────────────────────────────────────
    # lambda_dice ∈ [0.3, 0.7]:
    #   Yeung et al. [3] merekomendasikan lambda=0.5 sebagai titik awal
    #   balanced combination. Ni et al. [6] menggunakan 0.5/0.5. Reference
    #   project menemukan 0.7/0.3 dari grid search. Range [0.3, 0.7]
    #   mencakup semua konfigurasi yang relevan dari literatur.
    #
    # smooth ∈ [1e-7, 1.0] (log scale):
    #   Sama dengan Dice Loss — Bertels et al. [4].
    "dice_ssim": {
        "lambda_dice": ("float", 0.3, 0.7),
        "smooth":      ("float_log", 1e-7, 1.0),
    },

    # ── BCE + SSIM ───────────────────────────────────────────────────────────
    # lambda_bce ∈ [0.3, 0.7]:
    #   Ni et al. [6] menggunakan alpha=beta=0.5 (lambda_bce=0.5).
    #   Yeung et al. [3] merekomendasikan lambda=0.5 untuk balanced losses.
    #   Jadon et al. [5] menunjukkan [0.3, 0.7] sebagai range tipikal
    #   untuk combined medical segmentation losses.
    "bce_ssim": {
        "lambda_bce": ("float", 0.3, 0.7),
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _suggest(trial, loss_key: str) -> dict:
    """Suggest parameters for a trial and compute derived (constrained) params."""
    space = SEARCH_SPACES[loss_key]
    params: dict = {}

    for name, spec in space.items():
        kind, lo, hi = spec
        if kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)

    # Derived / constrained params
    if loss_key == "bce_mcc":
        params["lambda_mcc"] = round(1.0 - params["lambda_bce"], 8)
    elif loss_key == "dice_ssim":
        params["lambda_ssim"] = round(1.0 - params["lambda_dice"], 8)
    elif loss_key == "bce_ssim":
        params["lambda_ssim"] = round(1.0 - params["lambda_bce"], 8)

    return params


def _build_loss(loss_key: str, params: dict):
    """Construct a loss callable directly from params, bypassing LOSS_PARAMS."""
    from src.losses.loss_functions import _bce, _dice, _focal, _cldice, _ssim, MCCLoss

    if loss_key == "bce_mcc":
        def fn(y_true, y_pred):
            return (params["lambda_bce"] * _bce(y_true, y_pred)
                    + params["lambda_mcc"] * MCCLoss()(y_true, y_pred))
        return fn

    if loss_key == "dice":
        def fn(y_true, y_pred):
            return _dice(y_true, y_pred, smooth=params["smooth"])
        return fn

    if loss_key == "focal":
        def fn(y_true, y_pred):
            return _focal(y_true, y_pred,
                          alpha=params["alpha"],
                          gamma=params["gamma"],
                          smooth=params["smooth"])
        return fn

    if loss_key == "cldice":
        def fn(y_true, y_pred):
            cl = _cldice(y_true, y_pred, smooth=params["smooth"], iters=params["iters"])
            d  = _dice(y_true, y_pred, smooth=params["smooth"])
            return (1.0 - params["alpha"]) * d + params["alpha"] * cl
        return fn

    if loss_key == "dice_ssim":
        def fn(y_true, y_pred):
            return (params["lambda_dice"] * _dice(y_true, y_pred, smooth=params["smooth"])
                    + params["lambda_ssim"] * _ssim(y_true, y_pred))
        return fn

    if loss_key == "bce_ssim":
        def fn(y_true, y_pred):
            return (params["lambda_bce"] * _bce(y_true, y_pred)
                    + params["lambda_ssim"] * _ssim(y_true, y_pred))
        return fn

    raise ValueError(f"Unknown loss_key: {loss_key}")


# ── Tuner ─────────────────────────────────────────────────────────────────────

class LossTuner:
    """
    Optuna TPE-based hyperparameter tuner for loss functions.

    Objective : maximise F1 Score on the validation set.
    Method    : Tree-structured Parzen Estimator (TPE) — Bergstra et al. 2011,
                state-of-the-art for black-box HPT as of 2024/2025.
    Pruning   : MedianPruner removes clearly bad trials early.
    """

    def __init__(
        self,
        cfg: Union[DriveConfig, StareConfig],
        loss_key: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 30,
    ):
        self.cfg        = cfg
        self.loss_key   = loss_key
        self.x_train    = x_train
        self.y_train    = y_train
        self.x_val      = x_val
        self.y_val      = y_val
        self.n_epochs   = n_epochs
        self.out_dir    = RESULTS_DIR / cfg.name.lower() / "tuning"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.study_name = f"{cfg.name.lower()}_{loss_key}"
        self.db_path    = self.out_dir / f"{loss_key}_tuning.db"
        self.storage    = f"sqlite:///{self.db_path}"

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _objective(self, trial) -> float:
        import keras
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        from src.models import build_sa_unetv2

        params  = _suggest(trial, self.loss_key)
        loss_fn = _build_loss(self.loss_key, params)

        model = build_sa_unetv2(
            input_size    = self.cfg.input_size,
            start_neurons = self.cfg.start_neurons,
            block_size    = self.cfg.block_size,
            rate          = self.cfg.drop_rate,
        )
        model.compile(
            optimizer = Adam(learning_rate=self.cfg.learning_rate),
            loss      = loss_fn,
            metrics   = ['accuracy'],
        )
        model.fit(
            self.x_train, self.y_train,
            validation_data = (self.x_val, self.y_val),
            epochs          = self.n_epochs,
            batch_size      = self.cfg.batch_size,
            callbacks       = [EarlyStopping(monitor='val_loss',
                                             patience=_TRIAL_PATIENCE,
                                             restore_best_weights=True,
                                             verbose=0)],
            shuffle = True,
            verbose = 0,
        )

        y_pred     = model.predict(self.x_val, batch_size=self.cfg.batch_size, verbose=0)
        y_pred_bin = (y_pred.ravel() > 0.5).astype(np.uint8)
        y_true_bin = (self.y_val.ravel() > 0.5).astype(np.uint8)
        f1         = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        # Free GPU memory between trials
        del model
        gc.collect()
        keras.backend.clear_session()

        return f1

    # ── Progress callback ─────────────────────────────────────────────────────

    def _on_trial_end(self, study, trial) -> None:
        val  = f"{trial.value:.6f}" if trial.value is not None else "pruned"
        best = study.best_value if study.best_value is not None else 0.0
        print(
            f"  {trial.number + 1:>4}  F1={val}  Best={best:.6f}  {trial.params}",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tune(self, n_trials: int = 30) -> dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name  = self.study_name,
            storage     = self.storage,
            load_if_exists = True,          # resume jika study sudah ada di SQLite
            direction   = "maximize",
            sampler     = optuna.samplers.TPESampler(seed=42),
            pruner      = optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        finished    = [t for t in study.trials
                       if t.state.name in ("COMPLETE", "PRUNED")]
        already_run = len(finished)
        remaining   = max(0, n_trials - already_run)

        print(f"\n  {'═'*58}")
        print(f"  Hyperparameter Tuning — {self.loss_key.upper()} ({self.cfg.name})")
        print(f"  {'═'*58}")
        print(f"  Metode   : Optuna TPE + MedianPruner")
        print(f"  Target   : Maksimalkan F1 Score (val set)")
        print(f"  Trials   : {n_trials} total  |  Sudah: {already_run}  |  Sisa: {remaining}")
        print(f"  Epochs/trial : {self.n_epochs}")
        print(f"  DB       : {self.db_path}")
        print(f"  Output   : {self.out_dir}")

        if already_run > 0 and remaining == 0:
            print(f"\n  Semua {n_trials} trials sudah selesai. Memuat hasil dari DB...")
        elif already_run > 0:
            print(f"\n  Melanjutkan dari trial #{already_run + 1}...")
            print(f"\n  {'Trial':>5}  {'F1':>8}  {'Best':>8}  Params")
            print(f"  {'─'*56}")
        else:
            print(f"\n  {'Trial':>5}  {'F1':>8}  {'Best':>8}  Params")
            print(f"  {'─'*56}")

        t0 = time.perf_counter()
        if remaining > 0:
            study.optimize(self._objective, n_trials=remaining,
                           callbacks=[self._on_trial_end])
        elapsed = time.perf_counter() - t0

        # Reconstruct full best params (including derived/constrained values)
        best_params = dict(study.best_trial.params)
        if self.loss_key == "bce_mcc":
            best_params["lambda_mcc"] = round(1.0 - best_params["lambda_bce"], 8)
        elif self.loss_key == "dice_ssim":
            best_params["lambda_ssim"] = round(1.0 - best_params["lambda_dice"], 8)
        elif self.loss_key == "bce_ssim":
            best_params["lambda_ssim"] = round(1.0 - best_params["lambda_bce"], 8)

        result = {
            "loss_key"          : self.loss_key,
            "dataset"           : self.cfg.name,
            "best_f1"           : study.best_value,
            "best_params"       : best_params,
            "n_trials"          : len(study.trials),
            "n_epochs_per_trial": self.n_epochs,
            "elapsed_sec"       : round(elapsed, 2),
            "db_path"           : str(self.db_path),
            "all_trials": [
                {
                    "number": t.number,
                    "f1"    : t.value,
                    "params": t.params,
                    "state" : t.state.name,
                }
                for t in study.trials
            ],
        }

        json_path = self.out_dir / f"{self.loss_key}_tuning.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        self._save_plots(study)
        self._print_summary(result, json_path)

        return result

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _save_plots(self, study) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip plot.")
            return

        import optuna

        trials     = [t for t in study.trials if t.value is not None]
        if not trials:
            return

        f1_vals    = [t.value for t in trials]
        best_curve = [max(f1_vals[:i + 1]) for i in range(len(f1_vals))]
        trial_nums = [t.number for t in trials]
        param_names = list(study.best_trial.params.keys())
        n_params   = len(param_names)
        n_cols     = min(n_params, 3) if n_params else 1
        n_prows    = (n_params + n_cols - 1) // n_cols if n_params else 0
        total_rows = 2 + n_prows

        fig = plt.figure(figsize=(14, 4 * total_rows))
        fig.suptitle(
            f"Hyperparameter Tuning — {self.loss_key.upper()} ({self.cfg.name})\n"
            f"Best F1: {study.best_value:.6f}  |  Trials: {len(trials)}",
            fontsize=12, fontweight="bold",
        )
        gs = gridspec.GridSpec(total_rows, n_cols,
                               figure=fig, hspace=0.55, wspace=0.4)

        # ── Row 0: Optimization history ───────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, :])
        ax0.scatter(trial_nums, f1_vals, c="steelblue", s=35, alpha=0.7,
                    label="Trial F1", zorder=3)
        ax0.plot(trial_nums, best_curve, "r-", lw=2, label="Best so far")
        ax0.axhline(study.best_value, color="darkred", ls="--",
                    label=f"Best: {study.best_value:.6f}")
        ax0.set_xlabel("Trial"); ax0.set_ylabel("F1 Score")
        ax0.set_title("Optimization History")
        ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

        # ── Row 1: Parameter importance ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, :])
        try:
            imp   = optuna.importance.get_param_importances(study)
            names = list(imp.keys())
            vals  = list(imp.values())
            bars  = ax1.barh(names, vals, color="steelblue", edgecolor="white")
            for bar, v in zip(bars, vals):
                ax1.text(bar.get_width() + 0.005,
                         bar.get_y() + bar.get_height() / 2,
                         f"{v:.3f}", va="center", fontsize=8)
            ax1.set_xlabel("Importance (fANOVA)")
            ax1.grid(axis="x", alpha=0.3)
        except Exception:
            ax1.text(0.5, 0.5,
                     "Insufficient data for importance analysis\n(tambah jumlah trials)",
                     ha="center", va="center",
                     transform=ax1.transAxes, fontsize=10)
        ax1.set_title("Parameter Importance")

        # ── Rows 2+: Per-parameter scatter ────────────────────────────────────
        for i, param in enumerate(param_names):
            ax = fig.add_subplot(gs[2 + i // n_cols, i % n_cols])
            pvals = [t.params[param] for t in trials]
            ax.scatter(pvals, f1_vals, c="steelblue", s=25, alpha=0.7)
            bv = study.best_trial.params[param]
            ax.axvline(bv, color="red", ls="--", lw=1.5, label=f"Best={bv:.4g}")
            ax.legend(fontsize=8)
            ax.set_xlabel(param); ax.set_ylabel("F1")
            ax.set_title(f"{param} vs F1"); ax.grid(alpha=0.3)

        plot_path = self.out_dir / f"{self.loss_key}_tuning.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafik  : {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_summary(self, result: dict, json_path: Path) -> None:
        print(f"\n  {'═'*58}")
        print(f"  Tuning Selesai — {result['loss_key'].upper()} ({result['dataset']})")
        print(f"  {'═'*58}")
        print(f"  Best F1     : {result['best_f1']:.6f}")
        print(f"  Total trials: {result['n_trials']}")
        print(f"  Elapsed     : {result['elapsed_sec']:.0f}s")
        print(f"  DB (resume) : {result['db_path']}")
        print(f"  JSON        : {json_path}")
        print(f"\n  Best Params :")
        for k, v in result["best_params"].items():
            print(f"    {k:<20}: {v}")
        print(f"\n  Salin ke config.py → LOSS_PARAMS[\"{result['loss_key']}\"]:")
        print(f"    {json.dumps(result['best_params'], indent=4)}")
