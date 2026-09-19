"""
patch_mcc_per_image.py
----------------------
Patch semua *_results.json dengan menambahkan nilai MCC ke setiap entri
per_image_metrics, dihitung dari prediction images yang sudah tersimpan.

Tidak memerlukan re-training atau re-inference.
Jalankan dari root project:
    %run patch_mcc_per_image.py

Setelah selesai, jalankan ulang hanya section 4.10 multiseed reporter agar
pairwise_wilcoxon.json dan significance_matrix.png di-regenerate.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import matthews_corrcoef

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from config import LOSS_FUNCTIONS, RESULTS_DIR, _DATASETS_DIR

SEEDS = [42, 123, 456, 789, 2026]

DATASET_CFG = {
    "drive": {
        "labels_dir": _DATASETS_DIR / "DRIVE" / "test" / "labels",
        "masks_dir":  _DATASETS_DIR / "DRIVE" / "test" / "mask",
        "use_mask":   True,
    },
    "stare": {
        "labels_dir": _DATASETS_DIR / "STARE" / "test" / "labels",
        "masks_dir":  None,
        "use_mask":   False,
    },
}


def _load_sorted_pngs(folder: Path) -> list[np.ndarray]:
    # Coba beberapa ekstensi — mask DRIVE asli berformat .gif
    for pattern in ("*.png", "*.gif", "*.tif", "*.tiff", "*.jpg", "*.bmp"):
        files = sorted(folder.glob(pattern))
        if files:
            return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    raise FileNotFoundError(f"Tidak ada file gambar (png/gif/tif/jpg) di: {folder}")


def _compute_mcc_per_image(
    pred_dir: Path,
    labels: list[np.ndarray],
    masks: list[np.ndarray] | None,
) -> dict[str, float]:
    """
    Kembalikan {image_id: mcc_scaled} di mana mcc_scaled = mcc * 100
    (konsisten dengan cara penyimpanan dataset-level di _save_results).
    """
    pred_files = sorted(pred_dir.glob("pred_*.png"))
    if not pred_files:
        raise FileNotFoundError(f"Tidak ada file pred_*.png di: {pred_dir}")

    results: dict[str, float] = {}
    for i, pred_file in enumerate(pred_files):
        img_id = f"img_{i + 1:03d}"

        pred_bin = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        gt       = labels[i]

        pred_flat = (pred_bin.ravel() > 127).astype(np.uint8)
        gt_flat   = (gt.ravel()       > 127).astype(np.uint8)

        # Apply FOV mask jika tersedia (DRIVE)
        if masks is not None and i < len(masks):
            mask_flat = (masks[i].ravel() > 127)
            idx       = np.where(mask_flat)[0]
            if len(idx) == 0:
                continue
            pred_flat = pred_flat[idx]
            gt_flat   = gt_flat[idx]

        mcc = float(matthews_corrcoef(gt_flat, pred_flat))
        results[img_id] = round(mcc * 100, 4)  # skala ×100 sesuai _wilcoxon_overlap

    return results


def patch_json(json_path: Path, mcc_by_id: dict[str, float]) -> int:
    """Tambahkan mcc ke per_image_metrics. Kembalikan jumlah entri yang di-patch."""
    with open(json_path) as f:
        data = json.load(f)

    pim = data.get("per_image_metrics", [])
    if not pim:
        print(f"  [SKIP] per_image_metrics kosong: {json_path.name}")
        return 0

    patched = 0
    for entry in pim:
        img_id = entry.get("image_id")
        if img_id and img_id in mcc_by_id:
            entry["mcc"] = mcc_by_id[img_id]
            patched += 1

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return patched


def main():
    total_json_patched = 0
    total_images_patched = 0
    errors = []

    for dataset_name, cfg in DATASET_CFG.items():
        results_dir  = RESULTS_DIR / dataset_name
        pred_base    = results_dir / "predictions"
        labels_dir   = cfg["labels_dir"]
        masks_dir    = cfg["masks_dir"]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Muat ground truth labels (urutan sorted)
        if not labels_dir.exists():
            print(f"  [ERROR] Labels dir tidak ditemukan: {labels_dir}")
            errors.append(str(labels_dir))
            continue
        labels = _load_sorted_pngs(labels_dir)
        print(f"  Labels loaded: {len(labels)} gambar dari {labels_dir}")

        # Muat FOV masks jika ada
        masks = None
        if cfg["use_mask"] and masks_dir and masks_dir.exists():
            masks = _load_sorted_pngs(masks_dir)
            print(f"  Masks loaded:  {len(masks)} gambar dari {masks_dir}")
        elif cfg["use_mask"]:
            print(f"  [WARN] Masks dir tidak ditemukan: {masks_dir} — tanpa masking")

        for loss_key in LOSS_FUNCTIONS:
            for seed in SEEDS:
                seed_tag  = f"seed{seed}"
                pred_dir  = pred_base / f"{loss_key}_{seed_tag}"
                json_path = results_dir / f"{loss_key}_{seed_tag}_results.json"

                if not pred_dir.exists():
                    print(f"  [SKIP] pred dir tidak ada: {pred_dir.name}")
                    continue
                if not json_path.exists():
                    print(f"  [SKIP] JSON tidak ada: {json_path.name}")
                    continue

                try:
                    mcc_by_id = _compute_mcc_per_image(pred_dir, labels, masks)
                    n = patch_json(json_path, mcc_by_id)
                    print(f"  Patched {n:2d} gambar  ← {json_path.name}")
                    total_json_patched += 1
                    total_images_patched += n
                except Exception as e:
                    msg = f"  [ERROR] {json_path.name}: {e}"
                    print(msg)
                    errors.append(msg)

    print(f"\n{'='*60}")
    print(f"Selesai: {total_json_patched} file JSON di-patch, "
          f"{total_images_patched} entri per-gambar di-update.")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nTidak ada error.")
    print("\nLangkah selanjutnya:")
    print("  Jalankan ulang section 4.10 multiseed reporter untuk regenerate")
    print("  per_image_averaged.json, pairwise_wilcoxon.json, dan significance_matrix.png")


if __name__ == "__main__":
    main()
