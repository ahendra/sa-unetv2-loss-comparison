import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from config import DriveConfig, LOSS_FUNCTIONS, RESULTS_DIR, WEIGHTS_DIR
from src.preprocessing import build_pipeline


# Preprocessing conditions to test: (mode, display_label)
_CONDITIONS: List[Tuple[str, str]] = [
    ("rgb",        "RGB Original"),
    ("green",      "Green Channel"),
    ("green_clahe","Green + CLAHE"),
]

_ABLATION_LOSS_KEY = "bce_mcc"


class AblationReporter:
    """Run preprocessing ablation study for Section 4.2.

    Trains BCE+MCC baseline on DRIVE under three preprocessing conditions:
    RGB, Green Channel, and Green + CLAHE. Uses the original SA-UNetV2
    training hyperparameters. Results are saved as JSON + bar chart PNG.

    Single Responsibility: preprocessing ablation only (DRIVE + BCE+MCC).
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
        return results

    # ── Private ──────────────────────────────────────────────────────────────

    def _train_and_eval(
        self, cfg, mode, x_train, y_train, x_val, y_val,
        x_test, y_test, masks, restore_fn, n_epochs,
    ) -> Tuple[Dict, float]:
        import keras
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from keras.optimizers import Adam
        from sklearn.metrics import f1_score, roc_auc_score
        import cv2

        from config import LOSS_PARAMS
        from src.losses import get_loss_function
        from src.models import build_sa_unetv2

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

        y_pred_padded = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
        y_pred = restore_fn(y_pred_padded)

        # Aggregate metrics over all test images
        all_prob, all_bin, all_gt = [], [], []
        for pred, gt in zip(y_pred, y_test):
            prob = pred.squeeze().astype(np.float32)
            gt_2d = gt.squeeze()
            _, binary = cv2.threshold(prob, 0.5, 1.0, cv2.THRESH_BINARY)

            if masks is not None:
                pass  # simplified: use full image for ablation

            all_prob.extend(prob.ravel().tolist())
            all_bin.extend(binary.ravel().astype(np.uint8).tolist())
            all_gt.extend((gt_2d > 0.5).astype(np.uint8).ravel().tolist())

        all_prob = np.array(all_prob)
        all_bin  = np.array(all_bin, dtype=np.uint8)
        all_gt   = np.array(all_gt,  dtype=np.uint8)

        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score,
            jaccard_score, matthews_corrcoef, roc_auc_score,
        )
        tn, fp, fn, tp = confusion_matrix(all_gt, all_bin).ravel()
        metrics = {
            "accuracy":    round(float(accuracy_score(all_gt, all_bin)) * 100, 4),
            "sensitivity": round(float(tp / (tp + fn + 1e-8)) * 100, 4),
            "specificity": round(float(tn / (tn + fp + 1e-8)) * 100, 4),
            "f1":          round(float(f1_score(all_gt, all_bin, zero_division=0)) * 100, 4),
            "jaccard":     round(float(jaccard_score(all_gt, all_bin, zero_division=0)) * 100, 4),
            "mcc":         round(float(matthews_corrcoef(all_gt, all_bin)) * 100, 4),
            "auc":         round(float(roc_auc_score(all_gt, all_prob)) * 100, 4),
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
        conditions    = [v["label"] for v in results.values()]
        x = np.arange(len(metric_keys))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (mode, entry) in enumerate(results.items()):
            vals = [entry["metrics"].get(m, 0) for m in metric_keys]
            ax.bar(x + i * width, vals, width, label=entry["label"], alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score (%)")
        ax.set_title("Preprocessing Ablation Study — DRIVE (BCE+MCC, baseline)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

        path = self._out_dir / "preprocessing_ablation.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafik tersimpan: {path}")
