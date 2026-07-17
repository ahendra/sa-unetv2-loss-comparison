import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    LOSS_FUNCTIONS, LOSS_PARAMS, RANDOM_SEED, RESULTS_DIR,
    DriveConfig, StareConfig,
)
from src.data import DriveDataLoader, StareDataLoader
from src.models import build_sa_unetv2
from src.training.trainer import set_global_seed
from src.tuning.tuner import _build_loss as _build_loss_from_params
from .palette import LOSS_COLORS


_DATASETS = [("drive", DriveConfig), ("stare", StareConfig)]
_DS_LABELS = {"drive": "DRIVE", "stare": "STARE"}
_N_WARMUP  = 3

_RC = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.0,
    "axes.linewidth":  0.6,
}


class ComputationalTimeReporter:
    """3-part computational time analysis for thesis revision:

    1. Epochs to Convergence — from existing history JSONs (no retraining).
    2. Loss Computation Micro-benchmark — full training step (forward + loss +
       backward + gradient + optimizer update) using real test data on GPU.
    3. Scatter Plot — Convergence Speed vs Segmentation Performance (F1).
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self) -> List[Path]:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        print("\n  Memuat riwayat pelatihan dan hasil evaluasi...")
        histories = self._load_all_histories()
        results   = self._load_all_results()

        print("\n  [1/3] Jumlah Epoch hingga Konvergensi...")
        p1 = self._plot_epochs_convergence(histories)
        if p1:
            paths.append(p1)

        print("\n  [3/3] Scatter: Epochs vs F1...")
        p3 = self._plot_scatter_epochs_f1(histories, results)
        if p3:
            paths.append(p3)

        print("\n  [2/3] Mikro-Benchmark Langkah Pelatihan...")
        print("  (Data pelatihan aug_clahe/, arsitektur SA-UNetV2, gradien dihitung secara eksplisit)")
        bench = self._run_all_benchmarks()
        if bench:
            env_info = self._get_env_info()
            p2 = self._plot_benchmark(bench)
            if p2:
                paths.append(p2)
            self._print_benchmark_summary(bench)
            self._save_benchmark_json(bench, env_info)
            self._save_benchmark_excel(bench, env_info)

        return paths

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_all_histories(self) -> Dict:
        histories: Dict = {}
        for ds_name, _ in _DATASETS:
            histories[ds_name] = {}
            for loss_key in LOSS_FUNCTIONS:
                p = RESULTS_DIR / ds_name / "history" / f"{loss_key}_history.json"
                if p.exists():
                    with open(p) as f:
                        histories[ds_name][loss_key] = json.load(f)
                else:
                    print(f"  [LEWATI] {ds_name}/{loss_key}_history.json tidak ditemukan")
        return histories

    def _load_all_results(self) -> Dict:
        results: Dict = {}
        for ds_name, _ in _DATASETS:
            results[ds_name] = {}
            for loss_key in LOSS_FUNCTIONS:
                p = RESULTS_DIR / ds_name / f"{loss_key}_results.json"
                if p.exists():
                    with open(p) as f:
                        results[ds_name][loss_key] = json.load(f)
        return results

    # ── Analysis 1: Epochs to Convergence ────────────────────────────────────

    def _plot_epochs_convergence(self, histories: Dict) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [PERINGATAN] matplotlib tidak terpasang, grafik dilewati.")
            return None

        loss_keys   = list(LOSS_FUNCTIONS.keys())
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        ds_names    = [d for d, _ in _DATASETS]

        epochs_data: Dict[str, List[int]] = {}
        for ds_name in ds_names:
            epochs_data[ds_name] = []
            for loss_key in loss_keys:
                h = histories[ds_name].get(loss_key)
                epochs_data[ds_name].append(len(h["loss"]) if h and "loss" in h else 0)

        all_vals = [v for ds in ds_names for v in epochs_data[ds]]
        y_max    = (max(all_vals) if all_vals else 150) * 1.25

        with matplotlib.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor("#ffffff")

            x       = np.arange(len(loss_keys))
            width   = 0.35
            colors  = ["#4E7EC5", "#E8825A"]
            offsets = [-0.5 * width, 0.5 * width]

            for i, ds_name in enumerate(ds_names):
                vals = epochs_data[ds_name]
                bars = ax.bar(x + offsets[i], vals, width,
                              label=_DS_LABELS[ds_name], color=colors[i],
                              alpha=0.85, edgecolor="white")
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + y_max * 0.01,
                                str(v), ha="center", va="bottom", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(loss_labels, rotation=20, ha="right")
            ax.set_ylabel("Epochs to Convergence")
            ax.set_title("Epochs to Convergence per Loss Function",
                         fontweight="bold")
            ax.set_ylim(0, y_max)
            ax.legend()
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            plt.tight_layout()
            path = self._out_dir / "epochs_convergence.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Tersimpan: {path.name}")
        return path

    # ── Analysis 2: Micro-benchmark ───────────────────────────────────────────

    def _load_loss_params_for_dataset(self, ds_name: str, loss_key: str) -> dict:
        """Return optimal hyperparameters for (dataset, loss_key).

        Priority:
        1. Per-dataset tuning JSON saved by LossTuner:
           results/<ds>/tuning/<loss_key>_tuning.json  →  data["best_params"]
        2. Global LOSS_PARAMS from config.py (manual copy-paste fallback).
        """
        tuning_json = RESULTS_DIR / ds_name / "tuning" / f"{loss_key}_tuning.json"
        if tuning_json.exists():
            with open(tuning_json) as f:
                data = json.load(f)
            params = data.get("best_params", {})
            if params:
                return dict(params)
        # Fallback: global LOSS_PARAMS (may not match this dataset)
        print(f"    [CADANGAN] File tuning JSON tidak ditemukan untuk {ds_name}/{loss_key} "
              f"— menggunakan parameter global LOSS_PARAMS dari config.py")
        return dict(LOSS_PARAMS.get(loss_key, {}))

    def _run_all_benchmarks(self) -> Optional[Dict]:
        try:
            __import__("tensorflow")
        except ImportError:
            print("  [PERINGATAN] TensorFlow tidak tersedia, benchmark dilewati.")
            return None

        bench: Dict = {}
        for ds_name, cfg_cls in _DATASETS:
            cfg    = cfg_cls()
            result = self._benchmark_dataset(cfg, ds_name)
            if result:
                bench[ds_name] = result
        return bench or None

    def _load_benchmark_batches(self, cfg, ds_name: str) -> Optional[List[Tuple]]:
        """Muat semua data pelatihan (aug_clahe/) dan susun menjadi daftar batch tensor.

        Semua batch di-pre-load ke GPU sebelum loop pengukuran dimulai,
        sehingga transfer data host→GPU tidak ikut terukur dalam timing.
        Setiap iterasi pengukuran menggunakan batch gambar yang berbeda
        (1 epoch penuh = semua batch dalam data pelatihan).
        """
        try:
            import tensorflow as tf

            if ds_name == "drive":
                loader       = DriveDataLoader(cfg)
                x_all, y_all = loader.load_train()
            else:
                loader       = StareDataLoader(cfg)
                x_all, y_all = loader.load_train()

            bs        = cfg.batch_size
            n_images  = len(x_all)
            n_batches = n_images // bs

            print(f"  Data {_DS_LABELS[ds_name]}: {n_images} gambar → "
                  f"{n_batches} batch (batch_size={bs}) — dari aug_clahe/ "
                  f"(prapemrosesan CLAHE telah tersimpan dalam data).")
            print(f"  Menyiapkan {n_batches} batch tensor di GPU...", end=" ", flush=True)

            device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            with tf.device(device):
                batches = [
                    (tf.constant(x_all[i * bs:(i + 1) * bs], dtype=tf.float32),
                     tf.constant(y_all[i * bs:(i + 1) * bs], dtype=tf.float32))
                    for i in range(n_batches)
                ]
            print("selesai.", flush=True)
            return batches

        except Exception as exc:
            print(f"  [PERINGATAN] Gagal memuat data benchmark ({ds_name}): {exc}")
            return None

    def _benchmark_dataset(self, cfg, ds_name: str) -> Optional[Dict]:
        try:
            import tensorflow as tf
        except ImportError:
            return None

        batches = self._load_benchmark_batches(cfg, ds_name)
        if not batches:
            return None

        n_batches = len(batches)
        n_losses  = len(LOSS_FUNCTIONS)
        ds_label  = _DS_LABELS[ds_name]
        print(f"\n  ┌─ Dataset: {ds_label}  "
              f"(input={cfg.input_size}, batch={cfg.batch_size})")
        print(f"  │  {n_losses} fungsi loss × "
              f"({_N_WARMUP} pemanasan + {n_batches} pengukuran) iterasi")
        print(f"  │  Setiap iterasi pengukuran menggunakan batch gambar yang berbeda")
        print(f"  │  ({n_batches} iterasi = 1 epoch penuh pada data pelatihan)")
        print(f"  │  Iterasi pemanasan ke-1 mencakup penyusunan graf komputasi (@tf.function),")
        print(f"  │  proses ini dapat memakan waktu lebih lama (±10–60 detik, khususnya pada clDice).")
        print(f"  └─────────────────────────────────────────────────────")

        ds_t0   = time.perf_counter()
        timings: Dict = {}

        for loss_idx, (loss_key, loss_label) in enumerate(LOSS_FUNCTIONS.items()):
            short = loss_label.replace(" (Baseline)", "")
            print(f"\n  [{loss_idx + 1}/{n_losses}] {short}", flush=True)

            # ── Bangun model ─────────────────────────────────────────────────
            print("    Membangun model SA-UNetV2...", end=" ", flush=True)
            t_build = time.perf_counter()
            set_global_seed(RANDOM_SEED)
            model     = build_sa_unetv2(
                input_size=cfg.input_size,
                start_neurons=cfg.start_neurons,
                block_size=cfg.block_size,
                rate=cfg.drop_rate,
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
            params    = self._load_loss_params_for_dataset(ds_name, loss_key)
            loss_fn   = _build_loss_from_params(loss_key, params)
            print(f"selesai ({(time.perf_counter() - t_build):.1f}s)", flush=True)

            @tf.function
            def _step(x_b, y_b):
                with tf.GradientTape() as tape:
                    y_pred     = model(x_b, training=True)
                    loss_value = loss_fn(y_b, y_pred)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss_value

            # ── Pemanasan — iterasi-1 menyusun graf komputasi (@tf.function) ─
            print(f"    Pemanasan ({_N_WARMUP} iterasi, tidak dihitung sebagai waktu komputasi):")
            for w in range(_N_WARMUP):
                label = ("penyusunan graf komputasi (@tf.function)..."
                         if w == 0 else f"iterasi ke-{w + 1}...")
                print(f"      [{w + 1}/{_N_WARMUP}] {label}", end=" ", flush=True)
                x_b, y_b = batches[w % n_batches]
                t0 = time.perf_counter()
                _step(x_b, y_b)
                print(f"({(time.perf_counter() - t0) * 1000:.0f} ms)", flush=True)

            # ── Pengukuran — 1 epoch, batch berbeda tiap iterasi ─────────────
            # print() dipanggil SETELAH elapsed dicatat — tidak masuk pengukuran
            w = len(str(n_batches))  # lebar angka untuk alignment, mis. 2 untuk 29/93
            print(f"    Pengukuran ({n_batches} langkah = 1 epoch):")
            elapsed_ms: List[float] = []
            for i in range(n_batches):
                x_b, y_b = batches[i]
                t0 = time.perf_counter()
                _step(x_b, y_b).numpy()
                elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
                print(f"      [{i + 1:{w}d}/{n_batches}] ({elapsed_ms[-1]:.0f} ms)",
                      flush=True)

            mean_ms = float(np.mean(elapsed_ms))
            std_ms  = float(np.std(elapsed_ms))
            timings[loss_key] = {"mean_ms": mean_ms, "std_ms": std_ms,
                                 "n_steps": n_batches, "steps_ms": elapsed_ms}
            print(f"    Hasil:  {mean_ms:.1f} ± {std_ms:.1f} ms  "
                  f"(min={min(elapsed_ms):.0f} ms, maks={max(elapsed_ms):.0f} ms)",
                  flush=True)

            del model, optimizer, _step
            tf.keras.backend.clear_session()

        del batches
        total_s = time.perf_counter() - ds_t0
        print(f"\n  Benchmark {ds_label} selesai dalam {total_s / 60:.1f} menit.")
        return timings

    def _plot_benchmark(self, bench: Dict) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        loss_keys   = list(LOSS_FUNCTIONS.keys())
        loss_labels = [LOSS_FUNCTIONS[k].replace(" (Baseline)", "") for k in loss_keys]
        ds_avail    = [d for d in ("drive", "stare") if d in bench]
        n_ds        = len(ds_avail)

        with matplotlib.rc_context(_RC):
            fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 5), squeeze=False)
            fig.patch.set_facecolor("#ffffff")
            fig.suptitle("Computation Time per Training Step per Loss Function",
                         fontsize=12, fontweight="bold")

            for col, ds_name in enumerate(ds_avail):
                ax      = axes[0][col]
                t       = bench[ds_name]
                means   = [t.get(k, {}).get("mean_ms", 0) for k in loss_keys]
                stds    = [t.get(k, {}).get("std_ms",  0) for k in loss_keys]
                colors  = [LOSS_COLORS.get(k, "#aaaaaa") for k in loss_keys]

                x    = np.arange(len(loss_keys))
                bars = ax.bar(x, means, width=0.65, color=colors, alpha=0.85,
                              edgecolor="white", yerr=stds, capsize=4,
                              error_kw={"elinewidth": 1.2, "ecolor": "#333333"})

                cap = (max(stds) if stds else 0)
                for bar, m in zip(bars, means):
                    if m > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + cap * 0.15 + 1,
                                f"{m:.0f}", ha="center", va="bottom", fontsize=9)

                ax.set_xticks(x)
                ax.set_xticklabels(loss_labels, rotation=20, ha="right")
                ax.set_ylabel("Time per Step (ms)")
                ax.set_title(_DS_LABELS[ds_name], fontweight="bold")
                ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            path = self._out_dir / "benchmark_training_step.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Tersimpan: {path.name}")
        return path

    def _print_benchmark_summary(self, bench: Dict) -> None:
        print("\n  ── Ringkasan Hasil Benchmark ──")
        for ds_name, timings in bench.items():
            baseline = timings.get("bce_mcc", {}).get("mean_ms", 1.0) or 1.0
            n_steps  = next(iter(timings.values()), {}).get("n_steps", 0)
            print(f"\n  {_DS_LABELS[ds_name]}  "
                  f"(baseline = bce_mcc = {baseline:.1f} ms | {n_steps} langkah = 1 epoch)")
            print(f"  {'Fungsi Loss':22s}  {'Mean (ms)':>11}  {'Std (ms)':>9}  {'vs baseline':>12}")
            for loss_key, loss_label in LOSS_FUNCTIONS.items():
                t   = timings.get(loss_key, {})
                m   = t.get("mean_ms", 0)
                s   = t.get("std_ms",  0)
                rel = m / baseline
                print(f"  {loss_label[:22]:22s}  {m:>11.1f}  {s:>9.1f}  {rel:>10.2f}×")

    def _save_benchmark_json(self, bench: Dict, env_info: dict) -> None:
        """Persist benchmark numbers + environment info to JSON."""
        output: Dict = {
            "environment": env_info,
            "n_warmup":    _N_WARMUP,
            "results":     {},
        }
        for ds_name, timings in bench.items():
            baseline = timings.get("bce_mcc", {}).get("mean_ms", 1.0) or 1.0
            n_steps  = next(iter(timings.values()), {}).get("n_steps", 0)
            output["results"][ds_name] = {"n_steps_per_epoch": n_steps}
            for loss_key in LOSS_FUNCTIONS:
                t = timings.get(loss_key, {})
                m = t.get("mean_ms", 0.0)
                s = t.get("std_ms",  0.0)
                output["results"][ds_name][loss_key] = {
                    "mean_ms":              round(m, 3),
                    "std_ms":               round(s, 3),
                    "relative_to_baseline": round(m / baseline, 4) if baseline else 0.0,
                    "steps_ms":             [round(v, 1) for v in t.get("steps_ms", [])],
                }
        path = self._out_dir / "benchmark_results.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  JSON tersimpan: {path.name}")

    def _get_env_info(self) -> dict:
        """Collect runtime environment info: GPU, memory, TF/Python versions."""
        import sys
        import platform
        import subprocess
        from datetime import datetime

        env: dict = {
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version":     sys.version.split()[0],
            "platform":           platform.system() + " " + platform.release(),
            "is_colab":           False,
            "tensorflow_version": "N/A",
            "keras_version":      "N/A",
            "n_gpus":             0,
            "gpus":               [],
            "cuda_version":       "N/A",
        }

        # Colab detection
        try:
            __import__("google.colab")
            env["is_colab"] = True
        except ImportError:
            pass

        # TF + Keras versions
        try:
            import tensorflow as tf
            env["tensorflow_version"] = tf.__version__
            try:
                import keras
                env["keras_version"] = keras.__version__
            except ImportError:
                pass

            gpus = tf.config.list_physical_devices("GPU")
            env["n_gpus"] = len(gpus)
            for gpu in gpus:
                info: dict = {"device": gpu.name, "name": "Unknown",
                              "compute_capability": "N/A"}
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    info["name"] = details.get("device_name", "Unknown")
                    cc = details.get("compute_capability")
                    if cc:
                        info["compute_capability"] = f"{cc[0]}.{cc[1]}"
                except Exception:
                    pass
                env["gpus"].append(info)
        except ImportError:
            pass

        # GPU memory + driver via nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if smi.returncode == 0:
                for i, line in enumerate(smi.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    patch = {
                        "memory_total_mb": int(parts[1]),
                        "memory_free_mb":  int(parts[2]),
                        "memory_used_mb":  int(parts[3]),
                        "driver_version":  parts[4],
                    }
                    if i < len(env["gpus"]):
                        env["gpus"][i].update(patch)
                    else:
                        patch["name"] = parts[0]
                        env["gpus"].append(patch)
        except Exception:
            pass

        # CUDA version
        try:
            nvcc = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        env["cuda_version"] = line.strip()
                        break
        except Exception:
            pass

        return env

    def _save_benchmark_excel(self, bench: Dict, env_info: dict) -> None:
        """Save benchmark results to Excel (.xlsx) for direct copy-paste to thesis."""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter
        except ImportError:
            print("  [PERINGATAN] openpyxl tidak terpasang, file Excel dilewati. "
                  "Pasang dengan: pip install openpyxl")
            return

        wb = Workbook()
        wb.remove(wb.active)

        loss_keys   = list(LOSS_FUNCTIONS.keys())
        loss_labels = [LOSS_FUNCTIONS[k] for k in loss_keys]
        ds_avail    = [d for d in ("drive", "stare") if d in bench]

        # ── Sheet 1: Benchmark Timing ────────────────────────────────────────
        ws = wb.create_sheet("Benchmark Timing")

        hdr_font  = Font(bold=True, color="FFFFFF")
        hdr_fill  = PatternFill(start_color="2176AE", end_color="2176AE", fill_type="solid")
        bold_font = Font(bold=True)
        red_font  = Font(bold=True, color="CC0000")   # slowest
        grn_font  = Font(bold=True, color="1A7A1A")   # fastest

        # Header row
        headers = ["Loss Function"]
        for ds in ds_avail:
            label = _DS_LABELS[ds]
            headers += [f"{label} Mean (ms)", f"{label} Std (ms)", f"{label} vs Baseline"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        # Data rows
        for r, (lk, ll) in enumerate(zip(loss_keys, loss_labels), 2):
            ws.cell(row=r, column=1, value=ll)
            col = 2
            for ds in ds_avail:
                t        = bench.get(ds, {}).get(lk, {})
                m        = t.get("mean_ms", 0.0)
                s        = t.get("std_ms",  0.0)
                baseline = bench.get(ds, {}).get("bce_mcc", {}).get("mean_ms", 1.0) or 1.0
                rel      = m / baseline if baseline else 0.0
                ws.cell(row=r, column=col,     value=round(m, 2))
                ws.cell(row=r, column=col + 1, value=round(s, 2))
                ws.cell(row=r, column=col + 2, value=f"{rel:.2f}×")
                col += 3

        # Colour fastest (green) and slowest (red) mean per dataset
        for ds_idx, ds in enumerate(ds_avail):
            col_mean = 2 + ds_idx * 3
            means = [bench.get(ds, {}).get(lk, {}).get("mean_ms", 0.0) for lk in loss_keys]
            if not means:
                continue
            v_max, v_min = max(means), min(means)
            for r, lk in enumerate(loss_keys, 2):
                m    = bench.get(ds, {}).get(lk, {}).get("mean_ms", 0.0)
                cell = ws.cell(row=r, column=col_mean)
                if abs(m - v_max) < 1e-6:
                    cell.font = red_font
                elif abs(m - v_min) < 1e-6:
                    cell.font = grn_font

        # Benchmark metadata rows (below data)
        meta_row = len(loss_keys) + 3
        ws.cell(row=meta_row, column=1, value="Warmup runs").font = bold_font
        ws.cell(row=meta_row, column=2, value=_N_WARMUP)
        ws.cell(row=meta_row + 1, column=1,
                value="Timed runs (langkah/epoch)").font = bold_font
        for ds_idx, ds in enumerate(ds_avail):
            n_steps = next(iter(bench.get(ds, {}).values()), {}).get("n_steps", "N/A")
            ws.cell(row=meta_row + 1, column=2 + ds_idx * 3, value=n_steps)

        # Auto column width
        for col_cells in ws.columns:
            width = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = \
                min(width + 4, 28)
        ws.row_dimensions[1].height = 32

        # ── Sheet 2: Environment ──────────────────────────────────────────────
        ws2 = wb.create_sheet("Environment")
        ws2.cell(row=1, column=1, value="Parameter").font = hdr_font
        ws2.cell(row=1, column=1).fill = hdr_fill
        ws2.cell(row=1, column=2, value="Value").font = hdr_font
        ws2.cell(row=1, column=2).fill = hdr_fill

        rows = [
            ("Timestamp",           env_info.get("timestamp",          "")),
            ("Python Version",      env_info.get("python_version",     "")),
            ("TensorFlow Version",  env_info.get("tensorflow_version", "")),
            ("Keras Version",       env_info.get("keras_version",      "")),
            ("Platform",            env_info.get("platform",           "")),
            ("Google Colab",        str(env_info.get("is_colab",       ""))),
            ("CUDA Version",        env_info.get("cuda_version",       "")),
            ("Number of GPUs",      str(env_info.get("n_gpus",         0))),
        ]
        for gpu_i, gpu in enumerate(env_info.get("gpus", [])):
            p = f"GPU {gpu_i}"
            rows += [
                (f"{p} — Name",                gpu.get("name",             "")),
                (f"{p} — Compute Capability",   gpu.get("compute_capability", "")),
                (f"{p} — Memory Total (MB)",    str(gpu.get("memory_total_mb", ""))),
                (f"{p} — Memory Free  (MB)",    str(gpu.get("memory_free_mb",  ""))),
                (f"{p} — Memory Used  (MB)",    str(gpu.get("memory_used_mb",  ""))),
                (f"{p} — Driver Version",       gpu.get("driver_version",  "")),
            ]
        rows += [("Warmup Runs", str(_N_WARMUP))]
        for ds in ("drive", "stare"):
            if ds in bench:
                n_steps = next(iter(bench[ds].values()), {}).get("n_steps", "N/A")
                rows.append((f"Timed Runs — {_DS_LABELS[ds]} (langkah/epoch)",
                              str(n_steps)))
        for r, (k, v) in enumerate(rows, 2):
            ws2.cell(row=r, column=1, value=k)
            ws2.cell(row=r, column=2, value=v)
        ws2.column_dimensions["A"].width = 32
        ws2.column_dimensions["B"].width = 44

        path = self._out_dir / "benchmark_results.xlsx"
        wb.save(str(path))
        print(f"  Excel tersimpan : {path.name}")

    # ── Analysis 3: Scatter Epochs vs F1 ─────────────────────────────────────

    def _plot_scatter_epochs_f1(self, histories: Dict, results: Dict) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        loss_keys = list(LOSS_FUNCTIONS.keys())

        with matplotlib.rc_context(_RC):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor("#ffffff")
            fig.suptitle("Convergence Speed vs Segmentation Performance",
                         fontsize=12, fontweight="bold")

            for ax_idx, (ds_name, _) in enumerate(_DATASETS):
                ax      = axes[ax_idx]
                plotted = False

                for loss_key in loss_keys:
                    h = histories[ds_name].get(loss_key)
                    r = results[ds_name].get(loss_key)
                    if not h or not r:
                        continue
                    epochs = len(h.get("loss", []))
                    f1     = r.get("f1")
                    if not epochs or f1 is None:
                        continue

                    color = LOSS_COLORS.get(loss_key, "#aaaaaa")
                    short = (LOSS_FUNCTIONS[loss_key]
                             .replace(" (Baseline)", "")
                             .replace(" Loss", "")
                             .strip())
                    ax.scatter(epochs, f1, s=90, color=color, zorder=5,
                               edgecolors="white", linewidths=0.8)
                    ax.annotate(short, (epochs, f1),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=9, color=color)
                    plotted = True

                ax.set_xlabel("Epochs to Convergence")
                ax.set_ylabel("F1 Score (%)")
                ax.set_title(_DS_LABELS[ds_name], fontweight="bold")
                ax.grid(alpha=0.3, linewidth=0.5)
                if not plotted:
                    ax.text(0.5, 0.5, "Data tidak tersedia",
                            transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=11)

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            path = self._out_dir / "scatter_epochs_vs_f1.png"
            fig.savefig(str(path), dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  Tersimpan: {path.name}")
        return path
