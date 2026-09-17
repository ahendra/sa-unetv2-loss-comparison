from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Valid preprocessing modes — used by build_pipeline() factory
PREPROCESSING_MODES = ("rgb", "clahe", "green", "green_clahe")

# Global random seed — set the same value before every training run so that
# weight initialisation and data shuffle are identical across all loss functions.
# Change to None to disable fixed seeding (non-reproducible).
import os as _os

# Fixed seed — used for augmentation (original-image split) and Optuna TPE sampler.
# Must NOT change between runs so that all runs use identical train/val data splits.
RANDOM_SEED: int = 42

# Training seed — used for model weight initialisation, batch shuffling, and dropout.
# Override via environment variable to run multiple reproducible experiments:
#   EXPERIMENT_SEED=0   python main.py
#   EXPERIMENT_SEED=123 python main.py
# Falls back to RANDOM_SEED when not set so a plain `python main.py` is still reproducible.
_env_seed = _os.environ.get("EXPERIMENT_SEED")
TRAIN_SEED: int = int(_env_seed) if _env_seed is not None else RANDOM_SEED


BASE_DIR = Path(__file__).parent


# ── Environment detection ─────────────────────────────────────────────────────

def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_dirs() -> Tuple[Path, Path, Path]:
    """Return (datasets_dir, weights_dir, results_dir) for the current env."""
    if _is_colab():
        from google.colab import drive
        print("  Google Colab terdeteksi. Mounting Google Drive...")
        drive.mount('/content/drive')
        root = Path('/content/drive/MyDrive/Kuliah/Tesis/Program/sa_unetv2_loss_comparison')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'weights').mkdir(exist_ok=True)
        (root / 'results').mkdir(exist_ok=True)
        print(f"  Project root (Drive): {root}\n")
        return root / 'datasets', root / 'weights', root / 'results'
    return BASE_DIR / 'datasets', BASE_DIR / 'weights', BASE_DIR / 'results'


_DATASETS_DIR, WEIGHTS_DIR, RESULTS_DIR = _resolve_dirs()


# ── Dataset configs ───────────────────────────────────────────────────────────

@dataclass
class DriveConfig:
    name: str = "DRIVE"
    input_size: Tuple[int, int, int] = (592, 592, 3)
    original_h: int = 584
    original_w: int = 565

    # Original training data (augmentation source)
    train_images: str = str(_DATASETS_DIR / "DRIVE" / "train" / "images")
    train_labels: str = str(_DATASETS_DIR / "DRIVE" / "train" / "labels")

    # Augmented splits (output of RetinalAugmentationRunner)
    aug_dir: str           = str(_DATASETS_DIR / "DRIVE" / "aug")
    aug_train_images: str  = str(_DATASETS_DIR / "DRIVE" / "aug" / "train" / "images")
    aug_train_labels: str  = str(_DATASETS_DIR / "DRIVE" / "aug" / "train" / "labels")
    aug_val_images: str    = str(_DATASETS_DIR / "DRIVE" / "aug" / "validate" / "images")
    aug_val_labels: str    = str(_DATASETS_DIR / "DRIVE" / "aug" / "validate" / "labels")

    # Test set
    test_images: str = str(_DATASETS_DIR / "DRIVE" / "test" / "images")
    test_labels: str = str(_DATASETS_DIR / "DRIVE" / "test" / "labels")
    test_masks: str  = str(_DATASETS_DIR / "DRIVE" / "test" / "mask")
    use_mask_eval: bool = True

    # ── Training hyper-parameters (SA-UNetV2 original paper — DO NOT CHANGE) ──
    batch_size: int       = 8
    epochs: int           = 150
    learning_rate: float  = 1e-3
    start_neurons: int    = 16
    block_size: int       = 7
    drop_rate: float      = 0.15
    reduce_lr_patience: int  = 10
    reduce_lr_min: float     = 1e-6
    early_stop_patience: int = 20
    # Skenario 1 (SA-UNetV2 paper): "val_accuracy"
    # Skenario 2 (konsisten dengan EarlyStopping): "val_loss"
    checkpoint_monitor: str = "val_loss"

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # "rgb"         → original pipeline, no change (reproduces SA-UNetV2 paper)
    # "clahe"       → CLAHE per-channel RGB, keep 3-channel output (input channels: 3)
    # "green"       → green channel extraction only (input channels: 1)
    # "green_clahe" → green channel + CLAHE enhancement (input channels: 1)
    preprocessing_mode: str  = "clahe"
    clahe_clip_limit: float  = 2.0   # Liskowski & Krawiec (2016); Wang et al. (2020)
    clahe_tile_grid: int     = 8     # OpenCV default; standard in retinal segmentation

    def __post_init__(self):
        h, w, _ = self.input_size
        channels = 1 if self.preprocessing_mode in ("green", "green_clahe") else 3
        self.input_size = (h, w, channels)
        # Mode-specific augmentation directories — preprocessing baked in at aug-time
        _aug = _DATASETS_DIR / "DRIVE" / f"aug_{self.preprocessing_mode}"
        self.aug_dir          = str(_aug)
        self.aug_train_images     = str(_aug / "train"    / "images")
        self.aug_train_labels     = str(_aug / "train"    / "labels")
        self.aug_val_images       = str(_aug / "validate" / "images")
        self.aug_val_labels       = str(_aug / "validate" / "labels")
        self.aug_train_skeletons  = str(_aug / "train"    / "skeletons")
        self.aug_val_skeletons    = str(_aug / "validate" / "skeletons")


@dataclass
class StareConfig:
    name: str = "STARE"
    input_size: Tuple[int, int, int] = (704, 704, 3)

    # Original training data (augmentation source)
    train_images: str = str(_DATASETS_DIR / "STARE" / "train" / "images")
    train_labels: str = str(_DATASETS_DIR / "STARE" / "train" / "labels")

    # Augmented splits (output of RetinalAugmentationRunner)
    aug_dir: str           = str(_DATASETS_DIR / "STARE" / "aug")
    aug_train_images: str  = str(_DATASETS_DIR / "STARE" / "aug" / "train" / "images")
    aug_train_labels: str  = str(_DATASETS_DIR / "STARE" / "aug" / "train" / "labels")
    aug_val_images: str    = str(_DATASETS_DIR / "STARE" / "aug" / "validate" / "images")
    aug_val_labels: str    = str(_DATASETS_DIR / "STARE" / "aug" / "validate" / "labels")

    # Test set
    test_images: str      = str(_DATASETS_DIR / "STARE" / "test" / "images")
    test_labels: str      = str(_DATASETS_DIR / "STARE" / "test" / "labels")
    use_mask_eval: bool   = False

    # ── Training hyper-parameters (SA-UNetV2 original paper — DO NOT CHANGE) ──
    batch_size: int       = 2
    epochs: int           = 150
    learning_rate: float  = 1e-3
    start_neurons: int    = 16
    block_size: int       = 7
    drop_rate: float      = 0.15
    reduce_lr_patience: int  = 20
    reduce_lr_min: float     = 1e-8
    early_stop_patience: int = 30
    # Skenario 1 (SA-UNetV2 paper): "val_accuracy"
    # Skenario 2 (konsisten dengan EarlyStopping): "val_loss"
    checkpoint_monitor: str = "val_loss"

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # "rgb"         → original pipeline, no change (reproduces SA-UNetV2 paper)
    # "clahe"       → CLAHE per-channel RGB, keep 3-channel output (input channels: 3)
    # "green"       → green channel extraction only (input channels: 1)
    # "green_clahe" → green channel + CLAHE enhancement (input channels: 1)
    preprocessing_mode: str  = "clahe"
    clahe_clip_limit: float  = 2.0   # Liskowski & Krawiec (2016); Wang et al. (2020)
    clahe_tile_grid: int     = 8     # OpenCV default; standard in retinal segmentation

    def __post_init__(self):
        h, w, _ = self.input_size
        channels = 1 if self.preprocessing_mode in ("green", "green_clahe") else 3
        self.input_size = (h, w, channels)
        # Mode-specific augmentation directories — preprocessing baked in at aug-time
        _aug = _DATASETS_DIR / "STARE" / f"aug_{self.preprocessing_mode}"
        self.aug_dir          = str(_aug)
        self.aug_train_images     = str(_aug / "train"    / "images")
        self.aug_train_labels     = str(_aug / "train"    / "labels")
        self.aug_val_images       = str(_aug / "validate" / "images")
        self.aug_val_labels       = str(_aug / "validate" / "labels")
        self.aug_train_skeletons  = str(_aug / "train"    / "skeletons")
        self.aug_val_skeletons    = str(_aug / "validate" / "skeletons")


# ── Loss function registry ────────────────────────────────────────────────────

LOSS_FUNCTIONS = {
    "bce_mcc":    "BCE + MCC (Baseline)",
    "dice":       "Dice Loss",
    "focal":      "Focal Loss",
    "cldice":     "clDice Loss",
    "dice_ssim":  "Dice + SSIM",
    "bce_ssim":   "BCE + SSIM",
    "skel_recall": "Skeleton Recall Loss",
}

# ── Loss function hyperparameters ─────────────────────────────────────────────
#without tuning

# LOSS_PARAMS = {
#     "bce_mcc": {
#         "lambda_bce": 0.5,
#         "lambda_mcc": 0.5,
#     },
#     "dice": {
#         "smooth": 1e-6,
#     },
#     "focal": {
#         "alpha": 0.25,
#         "gamma": 2.0,
#         "smooth": 1e-7, #epsilon untuk mencegah log(0) 1e-7
#     },
#     "cldice": {
#         "smooth": 1,  # Shit et al. CVPR 2021: smooth=1.0 (hardcoded in original repo)
#         "iters":  25,   # Shit et al. CVPR 2021: 5...25, the iters its depends of the maximum diameter of vessel in datasets, for DRIVE and STARE the characteristic of vessel is big, so use maximum iters = 25 
#         "alpha":  0.2,  # Shit et al. CVPR 2021: L = (1-α)·Dice + α·clDice, where α ∈ [0, 0.5], for best clDice α=0.5 and for balance F1 α=0.2
#     },
#     "dice_ssim": {
#         "lambda_dice": 0.5,
#         "lambda_ssim": 0.5,
#         "smooth": 1e-6,
#     },
#     # Ref: "Retinal vascular segmentation network based on dual-scale
#     # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
#     "bce_ssim": {
#         "lambda_bce":  0.5,
#         "lambda_ssim": 0.5,
#     },
#     "skel_recall": {
#          "weight_srec": 1.0,
#          "smooth": 1e-5,
#     },
# }

#tuning DRIVE 50
# LOSS_PARAMS = {
#     "bce_mcc": {
#         "lambda_bce": 0.5559057406913699,
#         "lambda_mcc": 0.44409426
#     },
#     "dice": {
#         "smooth": 0.11567327199145977
#     },
#     "focal": {
#         "alpha": 0.542800147912569,
#         "gamma": 0.8588383811482752,
#         "smooth": 6.218004706861964e-06
#     },
#     "cldice": {
#         # tuning 2
#         # "alpha": 0.3009219503856401,
#         # "iters": 16,
#         # "smooth": 0.004183793285290999
#         # tuning 1
#         "alpha": 0.3202840996789727,
#         "iters": 30,
#         "smooth": 0.12302184430143771
#     },
#     "dice_ssim": {
#         "lambda_dice": 0.5404460046972834,
#         "smooth": 0.00904707195756839,
#         "lambda_ssim": 0.459554
#     },
#     # Ref: "Retinal vascular segmentation network based on dual-scale
#     # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
#     "bce_ssim": {
#         "lambda_bce": 0.6294373160561463,
#         "lambda_ssim": 0.37056268
#     },
#     "skel_recall": {
#         "weight_srec": 0.13066739238053282,
#         "smooth": 0.0002915443189153752
#     },
# }

#tuning STARE 50
LOSS_PARAMS = {
    "bce_mcc": {
        "lambda_bce": 0.5422423581171609,
        "lambda_mcc": 0.45775764
    },
    "dice": {
        "smooth": 4.1858227295469655e-05
    },
    "focal": {
        "alpha": 0.5052840765187266,
        "gamma": 0.7877688972507403,
        "smooth": 2.9589659956800404e-05
    },
    "cldice": {
        "alpha": 0.5062850085220214,
        "iters": 12,
        "smooth": 0.7402025335357656
    },
    "dice_ssim": {
        "lambda_dice": 0.3635944317485617,
        "smooth": 0.0004769102414304224,
        "lambda_ssim": 0.63640557
    },
    # Ref: "Retinal vascular segmentation network based on dual-scale
    # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
    "bce_ssim": {
        "lambda_bce": 0.6420156871217451,
        "lambda_ssim": 0.35798431
    },
    "skel_recall": {
        "weight_srec": 0.12486922135244156,
        "smooth": 3.3709732188335372e-06
    },
}
