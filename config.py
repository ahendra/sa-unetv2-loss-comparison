from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


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

    # Training hyper-parameters
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

    # Training hyper-parameters
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


# ── Loss function registry ────────────────────────────────────────────────────

LOSS_FUNCTIONS = {
    "bce_mcc":   "BCE + MCC (Baseline)",
    "dice":      "Dice Loss",
    "focal":     "Focal Loss",
    "cldice":    "clDice Loss",
    "dice_ssim": "Dice + SSIM",
    "bce_ssim":  "BCE + SSIM",
}

# ── Loss function hyperparameters ─────────────────────────────────────────────
#without tuning

LOSS_PARAMS = {
    "bce_mcc": {
        "lambda_bce": 0.5,
        "lambda_mcc": 0.5,
    },
    "dice": {
        "smooth": 1e-6,
    },
    # "focal": {
    #     "alpha": 0.25,
    #     "gamma": 2.0,
    #     "smooth": 1e-6,
    # },
    "focal": {
        "alpha": 0.5375999300628687,
        "gamma": 1.0960955962011756,
        "smooth": 7.4373220274892e-06,
     },
    "cldice": {
        "smooth": 1.0,  # Shit et al. CVPR 2021: smooth=1.0 (hardcoded in original repo)
        "iters":  10,
        "alpha":  0.5,  # Shit et al. CVPR 2021: L = (1-α)·Dice + α·clDice, default α=0.5
    },
    # "dice_ssim": {
    #     "lambda_dice": 0.5,
    #     "lambda_ssim": 0.5,
    #     "smooth": 1e-6,
    # },
    "dice_ssim": {
        "lambda_dice": 0.589607963945301,
        "lambda_ssim": 0.41039204,
        "smooth": 0.01231826460645177,
    },
    # Ref: "Retinal vascular segmentation network based on dual-scale
    # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
    "bce_ssim": {
        "lambda_bce":  0.5,
        "lambda_ssim": 0.5,
    },
}

#tuning DRIVE
# LOSS_PARAMS = {
#     "bce_mcc": {
#         "lambda_bce": 0.5, #0.3023440621957503,
#         "lambda_mcc": 0.5, #0.69765594,
#     },
#     "dice": {
#         "smooth": 1.5115319924378078e-05,
#     },
#     "focal": {
#         "alpha": 0.4850793731376367,
#         "gamma": 0.7591057855231964,
#         "smooth": 0.00012685264999107396,
#     },
#     "cldice": {
#         "smooth": 1.0,  # Shit et al. CVPR 2021: smooth=1.0 (hardcoded in original repo)
#         "iters":  25,
#         "alpha":  0.3227210948926833,  # Shit et al. CVPR 2021: L = (1-α)·Dice + α·clDice, default α=0.5
#     },
#     "dice_ssim": {
#         "lambda_dice": 0.589607963945301,
#         "lambda_ssim": 0.41039204,
#         "smooth": 0.01231826460645177,        
#     },
#     # Ref: "Retinal vascular segmentation network based on dual-scale
#     # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
#     "bce_ssim": {
#         "lambda_bce": 0.4240148908422406,
#         "lambda_ssim": 0.57598511
#     },
# }

#tuning STARE
# LOSS_PARAMS = {
#     "bce_mcc": {
#         "lambda_bce": 0.5, #0.362397808134481,
#         "lambda_mcc": 0.5, #0.63760219
#     },
#     "dice": {
#         "smooth": 0.5169304244323399,
#     },
#     "focal": {
#         "alpha": 0.5375999300628687,
#         "gamma": 1.0960955962011756,
#         "smooth": 7.4373220274892e-06,
#     },
#     "cldice": {
#         "smooth": 1.0,
#         "iters": 28,
#         "alpha": 0.35130001769736685,
#     },
#     "dice_ssim": {
#         "lambda_dice": 0.6542783048914624,
#         "lambda_ssim": 0.3457217,
#         "smooth": 0.13929185486422527,    
#     },
#     # Ref: "Retinal vascular segmentation network based on dual-scale
#     # morphological enhancement", Springer 2025 (DOI 10.1007/s44443-025-00191-3)
#     "bce_ssim": {
#         "lambda_bce": 0.501566111746115,
#         "lambda_ssim": 0.49843389,
#     },
# }
