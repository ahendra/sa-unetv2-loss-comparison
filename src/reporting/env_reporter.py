import json
import os
import sys
from pathlib import Path


class EnvironmentReporter:
    """Collect and persist experiment environment info for Section 4.1."""

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    def collect(self) -> dict:
        import platform

        info = {
            "python_version": sys.version,
            "platform":       platform.platform(),
            "cpu_count":      os.cpu_count(),
        }

        try:
            import tensorflow as tf
            info["tensorflow_version"] = tf.__version__
            gpus = tf.config.list_physical_devices("GPU")
            gpu_list = []
            for gpu in gpus:
                entry = {"name": gpu.name}
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    entry["device_name"] = details.get("device_name", "unknown")
                except Exception:
                    pass
                gpu_list.append(entry)
            info["gpu_info"] = gpu_list if gpu_list else ["No GPU detected"]
        except ImportError:
            info["tensorflow_version"] = "N/A (tensorflow not installed)"
            info["gpu_info"] = ["N/A (tensorflow not installed)"]

        try:
            import keras
            info["keras_version"] = keras.__version__
        except ImportError:
            info["keras_version"] = "N/A (keras not installed)"

        try:
            import optuna
            info["optuna_version"] = optuna.__version__
        except ImportError:
            info["optuna_version"] = "N/A (optuna not installed)"

        try:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        except ImportError:
            info["ram_gb"] = "N/A (install psutil)"

        return info

    def report(self) -> Path:
        """Collect env info, save JSON, print table. Returns JSON path."""
        data = self.collect()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self._out_dir / "environment.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        self._print(data)
        print(f"\n  Tersimpan: {json_path}")
        return json_path

    def _print(self, data: dict) -> None:
        print(f"\n  {'═'*52}")
        print(f"  Lingkungan Eksperimen (Section 4.1)")
        print(f"  {'═'*52}")
        rows = [
            ("Python",     data["python_version"].split()[0]),
            ("TensorFlow", data["tensorflow_version"]),
            ("Keras",      data["keras_version"]),
            ("Optuna",     data["optuna_version"]),
            ("Platform",   data["platform"]),
            ("CPU cores",  str(data["cpu_count"])),
            ("RAM",        f"{data['ram_gb']} GB" if isinstance(data["ram_gb"], float)
                           else data["ram_gb"]),
        ]
        for gpu in data.get("gpu_info", []):
            name = gpu.get("device_name", gpu) if isinstance(gpu, dict) else str(gpu)
            rows.append(("GPU", name))
        for label, value in rows:
            print(f"  {label:<18}: {value}")
