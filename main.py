"""
SA-UNetV2 Loss Function Comparison
===================================
Entry point. Sets up environment and launches the interactive CLI.

Usage:
    python main.py
"""

import os
import sys

# Suppress TF/CUDA noise before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

# Ensure project root is on sys.path so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
