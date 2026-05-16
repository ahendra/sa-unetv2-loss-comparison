import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from skimage.measure import label, euler_number
from skimage.morphology import skeletonize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

from config import DriveConfig, StareConfig, RESULTS_DIR


# ── Topological helpers ───────────────────────────────────────────────────────

def _betti_numbers(binary: np.ndarray):
    """Return (β0, β1) for a 2D binary image using 8-connectivity throughout.

    β0 = number of connected foreground components  (via skimage.label)
    β1 = number of independent loops/holes
       = β0 − χ  where χ = Euler characteristic (skimage.euler_number)

    Both β0 and χ use connectivity=2 (8-connected) for consistency.
    Reference: standard in topology-preserving segmentation literature
    (Hu et al. 2021, Clough et al. 2020).
    """
    binary_bool = binary.astype(bool)
    # β0: 8-connected components (connectivity=2 for 2D)
    _, b0 = label(binary_bool, connectivity=2, return_num=True)
    # Euler characteristic χ = β0 − β1  →  β1 = β0 − χ
    euler = euler_number(binary_bool, connectivity=2)
    b1 = max(b0 - euler, 0)
    return int(b0), int(b1)


def _cldice(y_true_bin: np.ndarray, y_pred_bin: np.ndarray,
            smooth: float = 1e-6) -> float:
    """clDice using morphological skeletonize (evaluation version)."""
    skel_pred = skeletonize(y_pred_bin.astype(bool))
    skel_true = skeletonize(y_true_bin.astype(bool))

    tprec = (np.sum(skel_pred & y_true_bin.astype(bool)) + smooth) / (
        np.sum(skel_pred) + smooth)
    tsens = (np.sum(skel_true & y_pred_bin.astype(bool)) + smooth) / (
        np.sum(skel_true) + smooth)

    return float(2.0 * tprec * tsens / (tprec + tsens + smooth))


# ── Per-image metric computation ──────────────────────────────────────────────

def _compute_metrics(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    y_pred_prob: np.ndarray,
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()

    b0_true, b1_true = _betti_numbers(y_true_2d)
    b0_pred, b1_pred = _betti_numbers(y_pred_2d)

    return {
        "accuracy":    float(accuracy_score(y_true_flat, y_pred_flat)),
        "sensitivity": float(recall_score(y_true_flat, y_pred_flat,
                                          zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "auc":         float(roc_auc_score(y_true_flat, y_pred_prob)),
        "mcc":         float(matthews_corrcoef(y_true_flat, y_pred_flat)),
        "f1":          float(f1_score(y_true_flat, y_pred_flat,
                                      zero_division=0)),
        "jaccard":     float(jaccard_score(y_true_flat, y_pred_flat,
                                           zero_division=0)),
        "cldice":      _cldice(y_true_2d, y_pred_2d),
        "betti0_error": abs(b0_pred - b0_true),
        "betti1_error": abs(b1_pred - b1_true),
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Evaluates a model on test data and persists metrics and predictions."""

    def __init__(self, cfg: Union[DriveConfig, StareConfig]):
        self.cfg = cfg

    def evaluate(
        self,
        model,
        loss_name: str,
        x_test_padded: np.ndarray,
        y_test: Union[np.ndarray, List[np.ndarray]],
        masks: Optional[np.ndarray] = None,
        restore_fn=None,
    ) -> Dict[str, float]:
        print(f"\n  Running inference on {len(x_test_padded)} test images...")
        t0 = time.perf_counter()
        y_pred_padded = model.predict(
            x_test_padded, batch_size=self.cfg.batch_size, verbose=0)
        elapsed = time.perf_counter() - t0
        print(f"  Inference done in {elapsed:.2f}s  "
              f"({elapsed/len(x_test_padded):.3f}s/image)")

        if restore_fn is not None:
            y_pred = restore_fn(y_pred_padded)
        else:
            y_pred = y_pred_padded

        pred_list = (list(y_pred) if isinstance(y_pred, np.ndarray)
                     else y_pred)
        gt_list   = (list(y_test) if isinstance(y_test, np.ndarray)
                     else y_test)

        # Prepare prediction save directory
        pred_dir = (RESULTS_DIR / self.cfg.name.lower()
                    / "predictions" / loss_name)
        pred_dir.mkdir(parents=True, exist_ok=True)

        per_image: List[Dict] = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            prob   = pred.squeeze().astype(np.float32)   # (H, W)
            gt_2d  = gt.squeeze()                        # (H, W)

            _, binary = cv2.threshold(prob, 0.5, 1.0, cv2.THRESH_BINARY)
            pred_bin_2d = binary.astype(np.uint8)        # (H, W)

            pred_prob_flat = prob.ravel()
            pred_bin_flat  = pred_bin_2d.ravel()
            gt_flat        = (gt_2d > 0.5).astype(np.uint8).ravel()

            # Apply FOV mask if available (DRIVE)
            prob_masked    = pred_prob_flat
            bin_masked     = pred_bin_flat
            gt_masked      = gt_flat
            if masks is not None and i < len(masks):
                mask_flat = (masks[i].ravel() if masks[i].ndim > 1
                             else masks[i])
                idx = np.where(mask_flat > 0.5)[0]
                if len(idx) == 0:
                    continue
                prob_masked = pred_prob_flat[idx]
                bin_masked  = pred_bin_flat[idx]
                gt_masked   = gt_flat[idx]

                # 2D mask for topology metrics (keep full image, mask out background)
                mask_2d    = (masks[i] if masks[i].ndim == 2
                              else masks[i].squeeze()) > 0.5
                pred_bin_2d_masked = (pred_bin_2d & mask_2d).astype(np.uint8)
                gt_2d_masked       = ((gt_2d > 0.5) & mask_2d).astype(np.uint8)
            else:
                pred_bin_2d_masked = pred_bin_2d
                gt_2d_masked       = (gt_2d > 0.5).astype(np.uint8)

            try:
                m = _compute_metrics(gt_masked, bin_masked, prob_masked,
                                     gt_2d_masked, pred_bin_2d_masked)
                per_image.append(m)
            except Exception as e:
                print(f"  [WARN] Could not compute metrics for image {i}: {e}")
                continue

            # Save binary prediction and probability map
            cv2.imwrite(str(pred_dir / f"pred_{i+1:03d}.png"),
                        (pred_bin_2d * 255).astype(np.uint8))
            cv2.imwrite(str(pred_dir / f"prob_{i+1:03d}.png"),
                        (prob * 255).astype(np.uint8))

        if not per_image:
            raise RuntimeError(
                "No valid metrics computed. Check dataset paths and labels.")

        avg = {
            key: float(np.mean([m[key] for m in per_image]))
            for key in per_image[0]
        }
        avg["inference_time_sec"] = round(elapsed, 4)
        avg["num_images"] = len(per_image)

        self._save_results(loss_name, avg)
        self._print_results(loss_name, avg)
        print(f"  Prediction images → {pred_dir}")
        return avg

    # ── Persistence ──────────────────────────────────────────────────────────

    def load_results(self, loss_name: str) -> Optional[Dict]:
        path = self._result_path(loss_name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def results_exist(self, loss_name: str) -> bool:
        return self._result_path(loss_name).exists()

    def _save_results(self, loss_name: str, metrics: Dict) -> None:
        path = self._result_path(loss_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _skip = {"inference_time_sec", "num_images"}
        pct = {
            k: (round(v * 100, 2) if "betti" not in k and k not in _skip else v)
            for k, v in metrics.items()
        }
        with open(path, 'w') as f:
            json.dump(pct, f, indent=2)
        print(f"  Results saved to: {path}")

    def _print_results(self, loss_name: str, metrics: Dict) -> None:
        print(f"\n  ── Evaluation Results: {loss_name} ({self.cfg.name}) ──")
        labels = {
            "accuracy":     "Accuracy",
            "sensitivity":  "Sensitivity (Recall)",
            "specificity":  "Specificity",
            "auc":          "AUC-ROC",
            "mcc":          "MCC",
            "f1":           "F1 Score",
            "jaccard":      "Jaccard Index",
            "cldice":       "clDice",
            "betti0_error": "Betti-0 Error (β0)",
            "betti1_error": "Betti-1 Error (β1)",
        }
        _skip = {"inference_time_sec", "num_images"}
        for key, lbl in labels.items():
            if key in metrics:
                if "betti" in key:
                    print(f"    {lbl:<28}: {metrics[key]:.2f}")
                else:
                    val = metrics[key] * 100 if metrics[key] <= 1.0 else metrics[key]
                    print(f"    {lbl:<28}: {val:.2f}%")

    def _result_path(self, loss_name: str) -> Path:
        return (RESULTS_DIR / self.cfg.name.lower()
                / f"{loss_name}_results.json")
