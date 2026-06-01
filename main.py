"""
SA-UNetV2 Loss Function Comparison
===================================
Entry point. Sets up environment and launches the interactive CLI.

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on sys.path so absolute imports work.
# Must come before config import below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RANDOM_SEED as _CFG_SEED

# ── Determinism & reproducibility ─────────────────────────────────────────────
# These MUST be set before TensorFlow / Keras is imported.
# TF_DETERMINISTIC_OPS  : forces TF to use deterministic GPU kernels (slower
#                         but reproducible). Without this, GPU parallelism
#                         causes different floating-point rounding each run.
# TF_CUDNN_DETERMINISTIC: forces cuDNN to pick deterministic convolution
#                         algorithms instead of the fastest non-deterministic one.
# PYTHONHASHSEED        : must be set before the interpreter starts to fully
#                         suppress Python's random hash salting; setting it here
#                         (at process entry, before any imports) is the earliest
#                         reliable point short of a wrapper script.
#
# When RANDOM_SEED is None, deterministic mode is skipped entirely — TF would
# raise RuntimeError if determinism is enabled without a seed being set.
if _CFG_SEED is not None:
    _SEED = str(_CFG_SEED)
    os.environ.setdefault("TF_DETERMINISTIC_OPS",   "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC",  "1")
    os.environ.setdefault("PYTHONHASHSEED",           _SEED)

# Suppress TF/CUDA noise before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")


def _check_dependencies() -> None:
    missing = []
    for pkg in ("tensorflow", "keras_cv", "cv2", "sklearn", "imageio", "PIL", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("[ERROR] Paket berikut belum terinstall:")
        for m in missing:
            print(f"  - {m}")
        print("\nJalankan: pip install -r requirements.txt")
        sys.exit(1)


def main() -> None:
    _check_dependencies()

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    from src.ui import run_main_menu
    run_main_menu()


if __name__ == "__main__":
    main()
