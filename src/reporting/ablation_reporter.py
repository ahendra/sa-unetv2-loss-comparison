import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, RESULTS_DIR, WEIGHTS_DIR
from src.preprocessing import build_pipeline




# Preprocessing conditions to test: (mode, display_label)
# Only 3-channel modes are included — green/green_clahe change the input layer
# from (H,W,3) to (H,W,1), which alters the model architecture and introduces
# a confounding variable separate from preprocessing.
_CONDITIONS: List[Tuple[str, str]] = [
    ("rgb",   "RGB Original"),
    ("clahe", "RGB + CLAHE"),
]

_ABLATION_LOSS_KEY = "bce_mcc"


class AblationReporter:
    """Run preprocessing ablation study for Section 4.2.

    Trains BCE+MCC baseline on DRIVE under two 3-channel preprocessing
    conditions: RGB Original and RGB + CLAHE. Uses the original SA-UNetV2
    training hyperparameters.

    Outputs saved to output_dir:
      - preprocessing_ablation.json          — raw metrics
      - preprocessing_ablation_table.txt     — formatted text table
      - preprocessing_ablation.png           — bar chart
      - preprocessing_sample_comparison.png  — visual before/after comparison
    """

    def __init__(self, output_dir: Path):
        self._out_dir = output_dir

    def run(self, n_epochs: int = 50) -> Dict:
        """Train + evaluate each preprocessing condition.

        Args:
            n_epochs: epochs per condition (default 50 for speed; use
                      cfg.epochs=150 for publication-quality results).

        Returns:
            dict mapping mode → {label, metrics, elapsed_sec}
        """
        results: Dict = {}

        for mode, label in _CONDITIONS:
            print(f"\n  {'─'*52}")
            print(f"  Ablation: {label}")
            print(f"  {'─'*52}")

            cfg = DriveConfig(preprocessing_mode=mode)
            pipeline = build_pipeline(mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

            from src.data import DriveDataLoader
            loader = DriveDataLoader(cfg, pipeline)

            print("  Memuat data...")
            x_train, y_train = loader.load_train()
            x_val,   y_val   = loader.load_validate()
            x_test,  y_test, masks = loader.load_test()
            print(f"  Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

            metrics, elapsed = self._train_and_eval(
                cfg, mode, x_train, y_train, x_val, y_val,
                x_test, y_test, masks, loader.restore_predictions, n_epochs,
            )
            results[mode] = {"label": label, "metrics": metrics, "elapsed_sec": elapsed}
            print(f"  F1={metrics['f1']:.4f}  Sensitivity={metrics['sensitivity']:.4f}")

        self._save(results)
        self._plot(results)
        self._plot_sample_images()
        return results

    # ── Private ──────────────────────────────────────────────────────────────

    def _train_and_eval(
        self, cfg, mode, x_train, y_train, x_val, y_val,
        x_test, y_test, masks, restore_fn, n_epochs,
    ) -> Tuple[Dict, float]:
        import keras
        from keras.callbacks import (
            Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        )
        from keras.optimizers import Adam
        from sklearn.metrics import f1_score, roc_auc_score
        import cv2

        from config import LOSS_PARAMS
        from src.losses import get_loss_function
        from src.models import build_sa_unetv2

        class _ProgressCB(Callback):
            def __init__(self, total):
                super().__init__()
                self._total = total
                self._t0 = None

            def on_train_begin(self, logs=None):  # noqa: ARG002
                self._t0 = time.perf_counter()
                print(f"  Training dimulai — max {self._total} epoch "
                      f"(EarlyStopping aktif)")
                print(f"  {'Epoch':>7}  {'Loss':>10}  {'Val Loss':>10}  "
                      f"{'Val Acc':>9}  {'Elapsed':>9}")
                print("  " + "─" * 56)
                import sys; sys.stdout.flush()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                elapsed  = time.perf_counter() - self._t0
                loss     = logs.get("loss",         float("nan"))
                val_loss = logs.get("val_loss",      float("nan"))
                val_acc  = logs.get("val_accuracy",  float("nan"))
                print(
                    f"  {epoch + 1:>4}/{self._total:<3}"
                    f"  {loss:>10.5f}"
                    f"  {val_loss:>10.5f}"
                    f"  {val_acc:>9.4f}"
                    f"  {elapsed:>7.0f}s",
                    flush=True,
                )

            def on_train_end(self, logs=None):  # noqa: ARG002
                elapsed = time.perf_counter() - self._t0
                print(f"  Training selesai dalam {elapsed:.0f}s")

        weight_path = (
            WEIGHTS_DIR / "ablation" / f"drive_{mode}_{_ABLATION_LOSS_KEY}.weights.h5"
        )
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        model = build_sa_unetv2(
            input_size=cfg.input_size,
            start_neurons=cfg.start_neurons,
            block_size=cfg.block_size,
            rate=cfg.drop_rate,
        )
        loss_fn = get_loss_function(_ABLATION_LOSS_KEY)
        ckpt_mode = "min" if "loss" in cfg.checkpoint_monitor else "max"
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss=loss_fn,
            metrics=["accuracy"],
        )

        t0 = time.perf_counter()
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=n_epochs,
            batch_size=cfg.batch_size,
            callbacks=[
                _ProgressCB(n_epochs),
                ModelCheckpoint(str(weight_path), monitor=cfg.checkpoint_monitor,
                                save_best_only=True, save_weights_only=True,
                                mode=ckpt_mode, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=cfg.reduce_lr_patience,
                                  min_lr=cfg.reduce_lr_min, verbose=0),
                EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience,
                              restore_best_weights=True, verbose=0),
            ],
            shuffle=True,
            verbose=0,
        )
        elapsed = time.perf_counter() - t0

        print("  Menjalankan inferensi pada data test...")
        y_pred_padded = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
        y_pred = restore_fn(y_pred_padded)
        print("  Menghitung metrik evaluasi (per-gambar + FOV mask)...")

        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score,
            jaccard_score, matthews_corrcoef, recall_score, roc_auc_score,
        )

        # Per-image metrics then average — same method as ModelEvaluator
        per_image: list = []
        for i, (pred, gt) in enumerate(zip(y_pred, y_test)):
            prob    = pred.squeeze().astype(np.float32)          # (H, W)
            gt_2d   = gt.squeeze()                               # (H, W)
            _, binary = cv2.threshold(prob, 0.5, 1.0, cv2.THRESH_BINARY)
            pred_bin  = binary.astype(np.uint8)

            prob_flat = prob.ravel()
            bin_flat  = pred_bin.ravel()
            gt_flat   = (gt_2d > 0.5).astype(np.uint8).ravel()

            # Apply FOV mask — same logic as ModelEvaluator
            if masks is not None and i < len(masks):
                mask_flat = masks[i].ravel() if masks[i].ndim > 1 else masks[i]
                idx = np.where(mask_flat > 0.5)[0]
                if len(idx) == 0:
                    continue
                prob_flat = prob_flat[idx]
                bin_flat  = bin_flat[idx]
                gt_flat   = gt_flat[idx]

            try:
                tn, fp, *_ = confusion_matrix(gt_flat, bin_flat).ravel()
                per_image.append({
                    "accuracy":    float(accuracy_score(gt_flat, bin_flat)),
                    "sensitivity": float(recall_score(gt_flat, bin_flat,
                                                      zero_division=0)),
                    "specificity": float(tn / (tn + fp + 1e-8)),
                    "f1":          float(f1_score(gt_flat, bin_flat,
                                                  zero_division=0)),
                    "jaccard":     float(jaccard_score(gt_flat, bin_flat,
                                                       zero_division=0)),
                    "mcc":         float(matthews_corrcoef(gt_flat, bin_flat)),
                    "auc":         float(roc_auc_score(gt_flat, prob_flat)),
                })
            except Exception as exc:
                print(f"  [WARN] Metrik gambar {i + 1} dilewati: {exc}")

        if not per_image:
            raise RuntimeError("Tidak ada metrik yang berhasil dihitung.")

        # Average across images (same as ModelEvaluator)
        metrics = {
            k: round(float(np.mean([m[k] for m in per_image])) * 100, 4)
            for k in per_image[0]
        }

        del model
        gc.collect()
        keras.backend.clear_session()

        return metrics, round(elapsed, 2)

    def _save(self, results: Dict) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # ── JSON ─────────────────────────────────────────────────────────────
        json_path = self._out_dir / "preprocessing_ablation.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # ── Text table (console + .txt file) ─────────────────────────────────
        table = self._build_table(results)
        print(table)
        txt_path = self._out_dir / "preprocessing_ablation_table.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table)

        print(f"\n  JSON  : {json_path}")
        print(f"  Tabel : {txt_path}")

    def _build_table(self, results: Dict) -> str:
        _COLS = [
            ("accuracy",    "Accuracy (%)"),
            ("sensitivity", "Sensitivity (%)"),
            ("specificity", "Specificity (%)"),
            ("f1",          "F1 (%)"),
            ("jaccard",     "Jaccard (%)"),
            ("auc",         "AUC (%)"),
            ("mcc",         "MCC (%)"),
        ]
        col_labels  = [lbl for _, lbl in _COLS]
        col_keys    = [k   for k, _  in _COLS]
        cond_labels = [v["label"] for v in results.values()]

        # Column widths
        w_cond = max(len(s) for s in cond_labels) + 2
        w_col  = max(max(len(l) for l in col_labels), 10) + 2

        sep  = "+" + ("-" * w_cond) + "+" + (("-" * w_col + "+") * len(_COLS))
        hdr  = ("|" + "Kondisi Preprocessing".center(w_cond) + "|"
                + "".join(l.center(w_col) + "|" for l in col_labels))

        lines = [
            "",
            "  ══ Hasil Preprocessing Ablation Study ══",
            "  Dataset : DRIVE  |  Model : BCE+MCC (Baseline)",
            "",
            "  " + sep,
            "  " + hdr,
            "  " + sep,
        ]

        # Collect values per metric to find best
        metric_vals = {k: [] for k in col_keys}
        for entry in results.values():
            for k in col_keys:
                metric_vals[k].append(entry["metrics"].get(k, float("nan")))

        for mode, entry in results.items():
            m = entry["metrics"]
            row = "|" + entry["label"].center(w_cond) + "|"
            for k in col_keys:
                val = m.get(k, float("nan"))
                cell = f"{val:.2f}"
                # Mark best value with *
                valid = [v for v in metric_vals[k] if v == v]
                if valid and abs(val - max(valid)) < 0.001:
                    cell += "*"
                row += cell.center(w_col) + "|"
            lines.append("  " + row)

        lines += [
            "  " + sep,
            "  * = nilai terbaik per metrik",
            "",
        ]
        return "\n".join(lines)

    def _plot(self, results: Dict) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib tidak terinstall, skip plot.")
            return

        metric_keys   = ["accuracy", "sensitivity", "specificity", "f1", "jaccard", "auc"]
        metric_labels = ["Accuracy", "Sensitivity", "Specificity", "F1", "Jaccard", "AUC"]
        n_cond = len(results)
        x      = np.arange(len(metric_keys))
        width  = 0.7 / max(n_cond, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, n_cond))

        # Dynamic y-axis range — zoom in so small differences are visible
        all_vals = [entry["metrics"].get(m, 0) for entry in results.values() for m in metric_keys]
        v_min = min(all_vals) if all_vals else 0.0
        v_max = max(all_vals) if all_vals else 100.0
        span  = max(v_max - v_min, 1.0)
        y_min = max(0.0,   v_min - max(2.0, span * 0.30))
        y_max = min(100.0, v_max + max(1.0, span * 0.15))

        fig, ax = plt.subplots(figsize=(12, 7))
        for i, (mode, entry) in enumerate(results.items()):
            vals   = [entry["metrics"].get(m, 0) for m in metric_keys]
            offset = (i - n_cond / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=entry["label"],
                            color=colors[i], alpha=0.85, edgecolor="white")
            label_offset = (y_max - y_min) * 0.008
            for bar in bars:
                h = bar.get_height()
                if h > y_min:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + label_offset,
                        f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title("Preprocessing Ablation Study — DRIVE (BCE+MCC, baseline)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(y_min, y_max)

        path = self._out_dir / "preprocessing_ablation.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafik tersimpan: {path}")

    def _plot_sample_images(self, n_samples: int = 3, crop_size: int = 200) -> None:
        """Save side-by-side visual comparison of RGB Original vs RGB + CLAHE.

        Generates a figure with n_samples rows × 4 columns:
          Col 0: Full image — RGB Original  (red box = crop area)
          Col 1: Full image — RGB + CLAHE   (red box = crop area)
          Col 2: Zoomed crop — RGB Original
          Col 3: Zoomed crop — RGB + CLAHE

        Saved as preprocessing_sample_comparison.png alongside other outputs.
        """
        import os

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from PIL import Image
        except ImportError:
            print("  [WARN] matplotlib/PIL tidak terinstall, skip sample images.")
            return

        cfg = DriveConfig()
        pipeline_rgb   = build_pipeline("rgb",   cfg.clahe_clip_limit, cfg.clahe_tile_grid)
        pipeline_clahe = build_pipeline("clahe", cfg.clahe_clip_limit, cfg.clahe_tile_grid)

        test_dir = cfg.test_images
        if not os.path.isdir(test_dir):
            print(f"  [WARN] Test dir tidak ditemukan: {test_dir}, skip sample images.")
            return

        files = sorted(
            f for f in os.listdir(test_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(test_dir, f))
        )[:n_samples]

        if not files:
            print("  [WARN] Tidak ada gambar test ditemukan, skip sample images.")
            return

        print(f"\n  Membuat perbandingan visual preprocessing ({n_samples} sampel)...")

        fig, axes = plt.subplots(
            n_samples, 4,
            figsize=(16, n_samples * 4.2),
            gridspec_kw={"wspace": 0.05, "hspace": 0.15},
        )
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            "RGB Original",
            "RGB + CLAHE",
            "Detail: RGB Original",
            "Detail: RGB + CLAHE",
        ]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=8)

        r = crop_size // 2

        for i, fname in enumerate(files):
            img_arr = np.array(
                Image.open(os.path.join(test_dir, fname)).convert("RGB")
            )
            rgb_out   = pipeline_rgb.apply(img_arr.copy())
            clahe_out = pipeline_clahe.apply(img_arr.copy())

            # Crop center — always inside the retinal circle
            h, w = img_arr.shape[:2]
            cy = max(r, min(h - r, h // 2))
            cx = max(r, min(w - r, w // 2))
            rgb_crop   = rgb_out  [cy - r: cy + r, cx - r: cx + r]
            clahe_crop = clahe_out[cy - r: cy + r, cx - r: cx + r]

            # Full images with red crop-area indicator
            for j, img in enumerate([rgb_out, clahe_out]):
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                rect = patches.Rectangle(
                    (cx - r, cy - r), crop_size, crop_size,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
                axes[i, j].add_patch(rect)

            # Zoomed crops
            axes[i, 2].imshow(rgb_crop)
            axes[i, 2].axis("off")
            axes[i, 3].imshow(clahe_crop)
            axes[i, 3].axis("off")

            # Row label (image filename)
            axes[i, 0].set_ylabel(
                os.path.splitext(fname)[0],
                fontsize=9, rotation=0, labelpad=58, va="center",
            )

        fig.suptitle(
            "Perbandingan Preprocessing: RGB Original vs RGB + CLAHE  —  DRIVE test set\n"
            "Kotak merah = area yang diperbesar (200 × 200 px, pusat retina)",
            fontsize=12, y=1.01,
        )

        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / "preprocessing_sample_comparison.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Perbandingan sampel tersimpan: {path}")
