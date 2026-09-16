import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from config import DriveConfig, StareConfig, LOSS_FUNCTIONS, RESULTS_DIR
from src.data import RetinalAugmentationRunner, DriveDataLoader, StareDataLoader
from src.preprocessing import build_pipeline
from src.evaluation import ModelEvaluator
from src.losses import get_loss_function
from src.training import ModelTrainer
from src.tuning import LossTuner
from src.reporting import (
    AblationReporter, ClaheTuningReporter, CombinedHistoryReporter,
    ComparisonBarReporter, ComparisonReporter, ComparisonReporterEN,
    ComputationalTimeReporter, EnvironmentReporter, HistoryReporter,
    MultiSeedReporter, TuningPlotReporter, VisualizationReporter,
)


def _load_multiseed_summary(ds: str) -> Optional[dict]:
    """Load multiseed_summary.json for a single dataset (drive/stare), or None."""
    p = (RESULTS_DIR / ds / "reports"
         / "section_4_10_multiseed" / ds / "multiseed_summary.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_multiseed_summaries() -> dict:
    """Return {ds: summary_dict} for both drive and stare (skip missing)."""
    out = {}
    for ds in ("drive", "stare"):
        s = _load_multiseed_summary(ds)
        if s:
            out[ds] = s
    return out


def _seeds_from_summary(summary: Optional[dict]) -> list:
    """Extract seed list from a multiseed summary dict, empty list if None."""
    if not summary:
        return []
    return summary.get("seeds", [])


# ── Formatting helpers ────────────────────────────────────────────────────────

def _separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _header(title: str) -> None:
    _separator("═")
    print(f"  {title}")
    _separator("═")


def _clear():
    try:
        from IPython.display import clear_output
        clear_output(wait=False)
        import sys
        sys.stdout.flush()
    except ImportError:
        os.system('cls' if os.name == 'nt' else 'clear')


def _prompt(options: list[str], back_label: str = "Back") -> int:
    """Display a numbered menu and return the 1-based choice (0 = not matched)."""
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    if back_label:
        print(f"  [0] {back_label}")
    import sys
    sys.stdout.flush()
    while True:
        raw = input("\n  Pilihan Anda: ").strip()
        if raw.isdigit():
            choice = int(raw)
            if 0 <= choice <= len(options):
                return choice
        print("  Input tidak valid. Masukkan angka yang tersedia.")


# ── Augmentation ──────────────────────────────────────────────────────────────

def _run_augmentation_drive() -> None:
    _header("Augmentasi Dataset — DRIVE")
    print("  Pilih mode preprocessing:\n")
    mode_choice = _prompt(
        [
            "RGB Original   → aug_rgb/   (untuk training utama & eksperimen loss)",
            "RGB + CLAHE    → aug_clahe/ (untuk ablation study preprocessing)",
        ],
        back_label="Kembali",
    )
    if mode_choice == 0:
        return

    mode = "rgb" if mode_choice == 1 else "clahe"
    cfg  = DriveConfig(preprocessing_mode=mode)

    _header(f"Augmentasi Dataset — DRIVE  [{mode.upper()}]")
    print(f"  Sumber gambar : {cfg.train_images}")
    print(f"  Sumber label  : {cfg.train_labels}")
    print(f"  Output        : {cfg.aug_dir}")
    if mode == "clahe":
        print("  Preprocessing : RGB + CLAHE diterapkan ke setiap gambar asli SEBELUM augmentasi")
    else:
        print("  Preprocessing : None (raw RGB)")
    print()
    print("  Spesifikasi augmentasi (SA-UNetV2 paper):")
    print("    randomRotation ×3, randomColor ×3, randomGaussian ×3")
    print("    h-flip ×1, v-flip ×1, hv-flip ×1  →  12 augmentasi/gambar")
    print("    + copy gambar original → 13 file/gambar asli")
    print("    Split di level gambar ASLI (seed=42): 18 train | 2 validate")
    print("    → 18 × 13 = 234 train  |  2 × 13 = 26 validate\n")

    if not Path(cfg.train_images).is_dir():
        print(f"  [ERROR] Direktori tidak ditemukan: {cfg.train_images}")
        input("  Tekan Enter untuk kembali...")
        return

    aug_train = Path(cfg.aug_train_images)
    if aug_train.exists() and any(aug_train.iterdir()):
        ans = input(f"  Folder {Path(cfg.aug_dir).name}/ sudah ada. Hapus dan buat ulang? (y/n): ").strip().lower()
        if ans != 'y':
            input("  Dilewati. Tekan Enter...")
            return
        import shutil
        shutil.rmtree(cfg.aug_dir, ignore_errors=True)

    confirm = input("  Mulai augmentasi? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    def drive_label_fn(img_name: str) -> str:
        stem = img_name.split('_')[0]
        for ext in ('.gif', '.png'):
            candidate = os.path.join(cfg.train_labels, f"{stem}_manual1{ext}")
            if os.path.exists(candidate):
                return f"{stem}_manual1{ext}"
        return f"{stem}_manual1.gif"

    pipeline = build_pipeline(mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid) if mode != "rgb" else None
    runner = RetinalAugmentationRunner()
    runner.run(
        src_img_dir         = cfg.train_images,
        src_lbl_dir         = cfg.train_labels,
        aug_base_dir        = cfg.aug_dir,
        label_suffix_fn     = drive_label_fn,
        preprocessing_pipeline = pipeline,
    )
    input("\n  Selesai. Tekan Enter untuk kembali...")


def _run_augmentation_stare() -> None:
    _header("Augmentasi Dataset — STARE")
    print("  Pilih mode preprocessing:\n")
    mode_choice = _prompt(
        [
            "RGB Original   → aug_rgb/   (untuk training utama & eksperimen loss)",
            "RGB + CLAHE    → aug_clahe/ (untuk ablation study preprocessing)",
        ],
        back_label="Kembali",
    )
    if mode_choice == 0:
        return

    mode = "rgb" if mode_choice == 1 else "clahe"
    cfg  = StareConfig(preprocessing_mode=mode)

    _header(f"Augmentasi Dataset — STARE  [{mode.upper()}]")
    print(f"  Sumber gambar : {cfg.train_images}")
    print(f"  Sumber label  : {cfg.train_labels}")
    print(f"  Output        : {cfg.aug_dir}")
    if mode == "clahe":
        print("  Preprocessing : RGB + CLAHE diterapkan ke setiap gambar asli SEBELUM augmentasi")
    else:
        print("  Preprocessing : None (raw RGB)")
    print()
    print("  Spesifikasi augmentasi (SA-UNetV2 paper — STARE):")
    print("    randomRotation ×3, randomColor ×3, randomGaussian ×3")
    print("    h-flip ×1, v-flip ×1, hv-flip ×1  →  12 augmentasi/gambar")
    print("    + copy gambar original → 13 file/gambar asli")
    print("    Split di level gambar ASLI (seed=42): 14 train | 2 validate")
    print("    → 14 × 13 = 182 train  |  2 × 13 = 26 validate\n")

    if not Path(cfg.train_images).is_dir():
        print(f"  [ERROR] Direktori tidak ditemukan: {cfg.train_images}")
        input("  Tekan Enter untuk kembali...")
        return

    aug_train = Path(cfg.aug_train_images)
    if aug_train.exists() and any(aug_train.iterdir()):
        ans = input(f"  Folder {Path(cfg.aug_dir).name}/ sudah ada. Hapus dan buat ulang? (y/n): ").strip().lower()
        if ans != 'y':
            input("  Dilewati. Tekan Enter...")
            return
        import shutil
        shutil.rmtree(cfg.aug_dir, ignore_errors=True)

    confirm = input("  Mulai augmentasi? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    def stare_label_fn(img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        return f"{base}.ah.ppm"

    pipeline = build_pipeline(mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid) if mode != "rgb" else None
    runner = RetinalAugmentationRunner()
    runner.run(
        src_img_dir         = cfg.train_images,
        src_lbl_dir         = cfg.train_labels,
        aug_base_dir        = cfg.aug_dir,
        label_suffix_fn     = stare_label_fn,
        preprocessing_pipeline = pipeline,
    )
    input("\n  Selesai. Tekan Enter untuk kembali...")


# ── Training ──────────────────────────────────────────────────────────────────

def _ensure_skel_dir(cfg) -> bool:
    """Pre-compute tubed skeleton maps for all augmented labels (idempotent).

    Only processes labels for which a skeleton PNG does not yet exist.
    Returns True on success, False if label directories are missing.
    """
    from src.losses.skeleton_utils import build_skeleton_dir
    pairs = [
        (cfg.aug_train_labels, cfg.aug_train_skeletons),
        (cfg.aug_val_labels,   cfg.aug_val_skeletons),
    ]
    for label_dir, skel_dir in pairs:
        lbl_path  = Path(label_dir)
        skel_path = Path(skel_dir)
        if not lbl_path.is_dir():
            print(f"\n  [ERROR] Label dir tidak ditemukan: {label_dir}")
            return False
        n_labels = sum(1 for f in lbl_path.iterdir()
                       if f.suffix.lower() == '.png' and not f.name.startswith('.'))
        n_skels  = (sum(1 for f in skel_path.iterdir()
                        if f.suffix.lower() == '.png' and not f.name.startswith('.'))
                    if skel_path.is_dir() else 0)
        if n_skels < n_labels:
            missing = n_labels - n_skels
            print(f"\n  Menghitung {missing} skeleton baru di: {skel_dir}")
            count = build_skeleton_dir(label_dir, skel_dir)
            print(f"  {count} skeleton tersimpan.")
    return True


def _stack_skeleton(cfg, y_train: np.ndarray, y_val: np.ndarray):
    """Return (y_train_stacked, y_val_stacked) with skeleton appended as last channel."""
    if isinstance(cfg, DriveConfig):
        loader = DriveDataLoader(cfg)
    else:
        loader = StareDataLoader(cfg)
    print("  Memuat skeleton train...")
    skel_train = loader.load_train_skeleton()
    print("  Memuat skeleton validate...")
    skel_val   = loader.load_validate_skeleton()
    return (np.concatenate([y_train, skel_train], axis=-1),
            np.concatenate([y_val,   skel_val],   axis=-1))


def _ensure_aug_dir(cfg) -> bool:
    """Cek apakah aug dir untuk mode aktif sudah ada; tawarkan generate jika belum.

    Returns True jika dir sudah ada atau berhasil di-generate, False jika dibatalkan.
    """
    aug_train = Path(cfg.aug_train_images)
    if aug_train.exists() and any(aug_train.iterdir()):
        return True

    mode = cfg.preprocessing_mode
    print(f"\n  [WARN] Folder augmentasi untuk mode '{mode}' belum ditemukan:")
    print(f"         {cfg.aug_dir}")
    print(f"\n  Folder ini diperlukan sebelum training dapat dimulai.")
    ans = input("  Generate augmentasi sekarang? (y/n): ").strip().lower()
    if ans != 'y':
        input("  Dibatalkan. Tekan Enter untuk kembali...")
        return False

    if not Path(cfg.train_images).is_dir():
        print(f"\n  [ERROR] Direktori sumber gambar tidak ditemukan: {cfg.train_images}")
        input("  Tekan Enter untuk kembali...")
        return False

    pipeline = (build_pipeline(mode, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
                if mode != "rgb" else None)

    if isinstance(cfg, DriveConfig):
        def label_fn(img_name: str) -> str:
            stem = img_name.split('_')[0]
            for ext in ('.gif', '.png'):
                candidate = os.path.join(cfg.train_labels, f"{stem}_manual1{ext}")
                if os.path.exists(candidate):
                    return f"{stem}_manual1{ext}"
            return f"{stem}_manual1.gif"
    else:
        def label_fn(img_name: str) -> str:
            base = os.path.splitext(img_name)[0]
            return f"{base}.ah.ppm"

    RetinalAugmentationRunner().run(
        src_img_dir         = cfg.train_images,
        src_lbl_dir         = cfg.train_labels,
        aug_base_dir        = cfg.aug_dir,
        label_suffix_fn     = label_fn,
        preprocessing_pipeline = pipeline,
    )
    return True


def _multi_seed_train_menu(trainer: ModelTrainer, cfg) -> None:
    """Train satu fungsi loss berulang kali dengan seed berbeda → tampilkan mean ± SD."""
    _header(f"Multi-Seed Experiment — {cfg.name}")
    print("  Training diulang dengan beberapa seed berbeda.")
    print("  Setiap run disimpan sebagai bobot terpisah (misal: bce_mcc_seed42.weights.h5).")
    print("  Gunakan hasil ini untuk melaporkan mean ± SD ke reviewer.\n")

    # Pilih fungsi loss
    loss_items = list(LOSS_FUNCTIONS.items())
    print("  Pilih fungsi loss:\n")
    choice = _prompt([v for _, v in loss_items], back_label="Kembali")
    if choice == 0:
        return
    loss_key, loss_label = loss_items[choice - 1]

    # Input seeds
    print(f"\n  Loss dipilih: {loss_label}")
    print("  Contoh input seeds: 42 0 123 456 789")
    raw = input("  Masukkan seeds (pisahkan spasi): ").strip()
    seeds = []
    for s in raw.split():
        try:
            seeds.append(int(s))
        except ValueError:
            pass
    if not seeds:
        print("  [ERROR] Tidak ada seed valid. Kembali.")
        input("  Tekan Enter...")
        return

    print(f"\n  Akan melatih {len(seeds)} run dengan seeds: {seeds}")
    print(f"  Weight files: {loss_key}_seed<N>.weights.h5")
    confirm = input("  Mulai? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    # Muat data sekali
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    if loss_key == "skel_recall":
        if not _ensure_skel_dir(cfg):
            input("  Tekan Enter...")
            return
        y_train, y_val = _stack_skeleton(cfg, y_train, y_val)

    loss_fn = get_loss_function(loss_key)

    # Training loop — gunakan dict agar pairing seed→nilai selalu benar
    # (append ke list bisa mismatch jika ada seed yang di-skip tanpa data)
    seed_vl_map:  dict = {}   # seed → best val_loss
    seed_va_map:  dict = {}   # seed → best val_acc pada epoch terbaik
    seed_ela_map: dict = {}   # seed → elapsed_sec

    for seed in seeds:
        tag = f"seed{seed}"
        print(f"\n{'=' * 60}")
        print(f"  Run seed={seed}  →  tag={tag}")
        print('=' * 60)

        weights_ok = trainer.weights_exist(loss_key, seed_tag=tag)
        history_ok = trainer.history_exists(loss_key, seed_tag=tag)

        if weights_ok and not history_ok:
            # Weights ada tapi history tidak → training sebelumnya terputus
            print(f"  [PERINGATAN] Weights ada tapi history tidak ditemukan.")
            print(f"  Training sebelumnya kemungkinan terputus (koneksi putus / Colab disconnect).")
            print(f"  Bobot yang tersimpan hanya mencerminkan epoch yang sempat berjalan.")
            ans = input(f"  Latih ulang dari awal? (y/n) [y]: ").strip().lower()
            if ans == 'n':
                print("  Dilewati — bobot mungkin tidak optimal (training tidak selesai).")
                continue
        elif weights_ok and history_ok:
            ans = input(f"  Bobot '{tag}' sudah ada (training selesai). Latih ulang? (y/n): ").strip().lower()
            if ans != 'y':
                print("  Dilewati — memuat history yang ada.")
                hist = trainer.load_history(loss_key, seed_tag=tag)
                if hist:
                    vl = hist.get("val_loss", [])
                    va = hist.get("val_accuracy", [])
                    if vl:
                        best_ep = int(np.argmin(vl))
                        seed_vl_map[seed] = float(vl[best_ep])
                        if va and best_ep < len(va):
                            seed_va_map[seed] = float(va[best_ep])
                    seed_ela_map[seed] = float(hist.get("elapsed_sec", 0.0))
                continue

        trainer.train(loss_key, loss_fn, x_train, y_train, x_val, y_val,
                      seed=seed, seed_tag=tag)

        hist = trainer.load_history(loss_key, seed_tag=tag)
        if hist:
            vl = hist.get("val_loss", [])
            va = hist.get("val_accuracy", [])
            if vl:
                best_ep = int(np.argmin(vl))
                seed_vl_map[seed] = float(vl[best_ep])
                if va and best_ep < len(va):
                    seed_va_map[seed] = float(va[best_ep])
            seed_ela_map[seed] = float(hist.get("elapsed_sec", 0.0))

    # ── Ringkasan mean ± SD ────────────────────────────────────────────────────
    vl_vals  = list(seed_vl_map.values())
    va_vals  = list(seed_va_map.values())
    ela_vals = list(seed_ela_map.values())

    print(f"\n{'═' * 60}")
    print(f"  Ringkasan Multi-Seed: {loss_label} ({cfg.name})")
    print(f"{'═' * 60}")
    print(f"  Seeds     : {seeds}")
    print(f"  Runs      : {len(vl_vals)}/{len(seeds)}")

    if vl_vals:
        mean_vl = float(np.mean(vl_vals))
        std_vl  = float(np.std(vl_vals, ddof=1)) if len(vl_vals) > 1 else 0.0
        print(f"\n  Best Val Loss  : {mean_vl:.5f} ± {std_vl:.5f}")
        for seed in seeds:
            vl = seed_vl_map.get(seed)
            status = f"{vl:.5f}" if vl is not None else "— (tidak selesai)"
            print(f"    seed={seed:<6}: {status}")

    if va_vals:
        mean_va = float(np.mean(va_vals))
        std_va  = float(np.std(va_vals, ddof=1)) if len(va_vals) > 1 else 0.0
        print(f"\n  Best Val Acc   : {mean_va:.4f} ± {std_va:.4f}")

    if ela_vals:
        total = sum(ela_vals)
        print(f"\n  Total waktu    : {total:.0f}s  ({total/60:.1f} menit)")

    print(f"\n  Bobot tersimpan di:")
    for seed in seeds:
        tag = f"seed{seed}"
        wp  = trainer._weight_path(loss_key, tag)
        status = "✓" if wp.exists() else "✗"
        print(f"    [{status}] {wp.name}")

    print(f"\n  Untuk mean ± SD metrik test (F1, AUC, dll):")
    print(f"  → Jalankan Evaluasi per seed via menu Evaluasi.")
    print(f"  → Pilih loss '{loss_label}', lalu masukkan seed tag saat diminta.")

    input("\n  Tekan Enter untuk kembali...")


def _multi_seed_train_all(trainer: ModelTrainer, cfg) -> None:
    """Train semua fungsi loss dengan beberapa seed — satu invokasi tanpa interupsi manual."""
    _header(f"Multi-Seed Experiment — Semua Loss — {cfg.name}")
    print("  Training semua fungsi loss secara berurutan dengan seeds yang sama.")
    print("  Bobot disimpan per seed (misal: bce_mcc_seed42.weights.h5).")
    print("  Jika weights sudah ada untuk seed tertentu, run tersebut dilewati otomatis.\n")

    # ── Input seeds ────────────────────────────────────────────────────────────
    print("  Contoh: 42 123 456 789 2026")
    raw = input("  Masukkan seeds (pisahkan spasi): ").strip()
    seeds = []
    for tok in raw.split():
        try:
            seeds.append(int(tok))
        except ValueError:
            pass
    if not seeds:
        print("  [ERROR] Tidak ada seed valid. Kembali.")
        input("  Tekan Enter...")
        return

    n_loss  = len(LOSS_FUNCTIONS)
    n_seeds = len(seeds)
    print(f"\n  Seeds      : {seeds}")
    print(f"  Loss       : {n_loss} fungsi (semua)")
    print(f"  Total runs : {n_seeds * n_loss}")
    confirm = input("  Mulai? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    # ── Muat data sekali ───────────────────────────────────────────────────────
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    # Skeleton pre-computation dilakukan sekali sebelum skel_recall
    skel_ready              = False
    y_train_skel: Optional[np.ndarray] = None
    y_val_skel:   Optional[np.ndarray] = None

    # Kumpulkan statistik untuk ringkasan akhir
    summary: dict = {}

    # ── Loop semua loss ────────────────────────────────────────────────────────
    for loss_idx, (loss_key, loss_label) in enumerate(LOSS_FUNCTIONS.items()):
        print(f"\n{'═' * 60}")
        print(f"  [{loss_idx + 1}/{n_loss}] {loss_label}")
        print(f"{'═' * 60}")

        _y_train, _y_val = y_train, y_val
        if loss_key == "skel_recall":
            if not skel_ready:
                if not _ensure_skel_dir(cfg):
                    print("  [ERROR] Skeleton generation gagal. Loss ini dilewati.")
                    summary[loss_key] = {"label": loss_label, "n_done": 0,
                                         "val_loss": [], "elapsed": []}
                    continue
                y_train_skel, y_val_skel = _stack_skeleton(cfg, y_train, y_val)
                skel_ready = True
            _y_train, _y_val = y_train_skel, y_val_skel

        loss_fn = get_loss_function(loss_key)
        val_losses_best: list = []
        elapsed_list:    list = []

        # ── Loop semua seed ────────────────────────────────────────────────────
        for seed in seeds:
            tag = f"seed{seed}"
            print(f"\n  {'─' * 40}")
            print(f"  seed={seed}  →  {loss_key}_{tag}")

            weights_ok = trainer.weights_exist(loss_key, seed_tag=tag)
            history_ok = trainer.history_exists(loss_key, seed_tag=tag)

            if weights_ok and history_ok:
                # Training selesai normal — aman untuk dilewati
                print(f"  Selesai (weights + history ada) — dilewati otomatis.")
                hist = trainer.load_history(loss_key, seed_tag=tag)
                if hist:
                    vl = hist.get("val_loss", [])
                    if vl:
                        val_losses_best.append(min(vl))
                    elapsed_list.append(hist.get("elapsed_sec", 0.0))
                continue

            if weights_ok and not history_ok:
                # Weights ada tapi history tidak → training terputus sebelum selesai
                # Bobot hanya mencerminkan epoch yang sempat berjalan (tidak lengkap)
                print(f"  [PERINGATAN] Weights ada tapi history tidak ditemukan.")
                print(f"  Training sebelumnya kemungkinan terputus sebelum selesai.")
                print(f"  Melatih ulang dari awal — weights lama akan ditimpa.")

            trainer.train(loss_key, loss_fn, x_train, _y_train, x_val, _y_val,
                          seed=seed, seed_tag=tag)

            hist = trainer.load_history(loss_key, seed_tag=tag)
            if hist:
                vl = hist.get("val_loss", [])
                if vl:
                    val_losses_best.append(min(vl))
                elapsed_list.append(hist.get("elapsed_sec", 0.0))

        summary[loss_key] = {
            "label":    loss_label,
            "n_done":   len(val_losses_best),
            "val_loss": val_losses_best,
            "elapsed":  elapsed_list,
        }

    # ── Ringkasan akhir ────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  RINGKASAN MULTI-SEED SEMUA LOSS — {cfg.name}")
    print(f"{'═' * 70}")
    print(f"  {'Fungsi Loss':26s}  {'Runs':>8}  {'Best Val Loss (mean±SD)':>24}  {'Total Waktu':>12}")
    print(f"  {'─' * 26}  {'─' * 8}  {'─' * 24}  {'─' * 12}")

    for loss_key, info in summary.items():
        vls  = info["val_loss"]
        elps = info["elapsed"]

        vl_str = (
            f"{float(np.mean(vls)):.5f} ± "
            f"{float(np.std(vls, ddof=1)) if len(vls) > 1 else 0.0:.5f}"
            if vls else "—"
        )
        time_str = (
            f"{sum(elps) / 60:.1f} min"
            if elps else "—"
        )
        print(f"  {info['label'][:26]:26s}  "
              f"{info['n_done']:>4}/{n_seeds:>3}  "
              f"{vl_str:>24}  "
              f"{time_str:>12}")

    print(f"\n  Untuk mean ± SD metrik test (F1, AUC, dll):")
    print(f"  → Jalankan Evaluasi via menu Evaluasi → masukkan seed tag (seed42, seed123, ...).")
    input("\n  Tekan Enter untuk kembali...")


def _loss_selection_menu(trainer: ModelTrainer, cfg) -> None:
    _header(f"Training Model — {cfg.name}")
    print("  Pilih fungsi loss:\n")

    loss_items = list(LOSS_FUNCTIONS.items())
    options = (
        [f"{v}" for _, v in loss_items]
        + ["Semua Fungsi Loss (Train All)"]
        + ["Multi-Seed Experiment — Satu Loss (Mean ± SD)"]
        + ["Multi-Seed Experiment — Semua Loss (Mean ± SD)"]
    )

    choice = _prompt(options, back_label="Kembali ke menu dataset")
    if choice == 0:
        return
    if choice == len(loss_items) + 1:
        _train_all(trainer, cfg)
    elif choice == len(loss_items) + 2:
        _multi_seed_train_menu(trainer, cfg)
    elif choice == len(loss_items) + 3:
        _multi_seed_train_all(trainer, cfg)
    else:
        key, label = loss_items[choice - 1]
        _train_single(trainer, cfg, key, label)


def _train_single(trainer: ModelTrainer, cfg, loss_key: str, loss_label: str) -> None:
    print(f"\n  Loss: {loss_label}")
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    if loss_key == "skel_recall":
        if not _ensure_skel_dir(cfg):
            input("  Tekan Enter untuk kembali...")
            return
        y_train, y_val = _stack_skeleton(cfg, y_train, y_val)

    if trainer.weights_exist(loss_key):
        ans = input(f"\n  Bobot untuk '{loss_key}' sudah ada. Latih ulang? (y/n): ").strip().lower()
        if ans != 'y':
            input("  Dilewati. Tekan Enter...")
            return

    loss_fn = get_loss_function(loss_key)
    trainer.train(loss_key, loss_fn, x_train, y_train, x_val, y_val)
    input("\n  Training selesai. Tekan Enter untuk kembali...")


def _train_all(trainer: ModelTrainer, cfg) -> None:
    print("\n  Training semua fungsi loss secara berurutan...\n")
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    # Pre-compute skeletons once if skel_recall is in the list
    skel_ready = False
    y_train_skel_stacked: Optional[np.ndarray] = None
    y_val_skel_stacked:   Optional[np.ndarray] = None

    for loss_key, loss_label in LOSS_FUNCTIONS.items():
        print(f"\n{'=' * 60}")
        print(f"  [{list(LOSS_FUNCTIONS.keys()).index(loss_key)+1}/{len(LOSS_FUNCTIONS)}] {loss_label}")
        print('=' * 60)

        if trainer.weights_exist(loss_key):
            ans = input(f"  Bobot '{loss_key}' sudah ada. Lewati? (y/n) [y]: ").strip().lower()
            if ans != 'n':
                print("  Dilewati.")
                continue

        _y_train, _y_val = y_train, y_val
        if loss_key == "skel_recall":
            if not skel_ready:
                if not _ensure_skel_dir(cfg):
                    print("  [ERROR] Skeleton generation gagal. Dilewati.")
                    continue
                y_train_skel_stacked, y_val_skel_stacked = _stack_skeleton(cfg, y_train, y_val)
                skel_ready = True
            _y_train, _y_val = y_train_skel_stacked, y_val_skel_stacked

        loss_fn = get_loss_function(loss_key)
        trainer.train(loss_key, loss_fn, x_train, _y_train, x_val, _y_val)

    input("\n  Semua training selesai. Tekan Enter untuk kembali...")


def _load_training_data(cfg):
    if not _ensure_aug_dir(cfg):
        return None, None, None, None
    try:
        aug_folder = f"aug_{cfg.preprocessing_mode}"
        if isinstance(cfg, DriveConfig):
            loader = DriveDataLoader(cfg)
            print(f"\n  Memuat data training DRIVE ({aug_folder}/train)...")
            x_train, y_train = loader.load_train()
            print(f"  Train: {x_train.shape}")
            print(f"  Memuat data validasi DRIVE ({aug_folder}/validate)...")
            x_val, y_val = loader.load_validate()
            print(f"  Validasi: {x_val.shape}")
        else:
            loader = StareDataLoader(cfg)
            print(f"\n  Memuat data training STARE ({aug_folder}/train)...")
            x_train, y_train = loader.load_train()
            print(f"  Train: {x_train.shape}")
            print(f"  Memuat data validasi STARE ({aug_folder}/validate)...")
            x_val, y_val = loader.load_validate()
            print(f"  Validasi: {x_val.shape}")
        return x_train, y_train, x_val, y_val
    except Exception as e:
        print(f"\n  [ERROR] Gagal memuat data: {e}")
        print("  Pastikan path dataset sudah benar di config.py")
        input("  Tekan Enter untuk kembali...")
        return None, None, None, None


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluation_menu(trainer: ModelTrainer, evaluator: ModelEvaluator, cfg) -> None:
    _header(f"Evaluasi Model — {cfg.name}")

    # Tanya seed tag (opsional — kosong = model default tanpa tag)
    print("  Seed tag (opsional): kosongkan untuk model default,")
    print("  atau masukkan tag seed (contoh: seed42, seed0, seed123)")
    seed_tag = input("  Seed tag: ").strip()
    if seed_tag:
        print(f"  → Mengevaluasi bobot dengan tag: [{seed_tag}]")
    print()

    loss_items = list(LOSS_FUNCTIONS.items())
    options = [
        f"{v}"
        + (" [✓ bobot]" if trainer.weights_exist(k, seed_tag=seed_tag) else "")
        + (" [✓ hasil]" if evaluator.results_exist(k) else "")
        for k, v in loss_items
    ] + ["Evaluasi Semua Model"]

    choice = _prompt(options, back_label="Kembali ke menu dataset")
    if choice == 0:
        return
    if choice == len(options):
        _evaluate_all(trainer, evaluator, cfg, seed_tag=seed_tag)
    else:
        key, label = loss_items[choice - 1]
        _evaluate_single(trainer, evaluator, cfg, key, seed_tag=seed_tag)


def _evaluate_single(trainer: ModelTrainer, evaluator: ModelEvaluator, cfg,
                     loss_key: str, seed_tag: str = "") -> None:
    if not trainer.weights_exist(loss_key, seed_tag=seed_tag):
        tag_info = f" [seed_tag={seed_tag}]" if seed_tag else ""
        print(f"\n  [ERROR] Bobot untuk '{loss_key}'{tag_info} tidak ditemukan. Lakukan training terlebih dahulu.")
        input("  Tekan Enter untuk kembali...")
        return

    try:
        tag_info = f" [seed_tag={seed_tag}]" if seed_tag else ""
        print(f"\n  Memuat model dengan bobot '{loss_key}'{tag_info}...")
        model = trainer.load_weights(loss_key, seed_tag=seed_tag)

        x_test, y_test, masks, restore_fn = _load_test_data(cfg)
        if x_test is None:
            return

        evaluator.evaluate(model, loss_key, x_test, y_test, masks, restore_fn,
                           seed_tag=seed_tag)
    except Exception as e:
        print(f"\n  [ERROR] Evaluasi gagal: {e}")

    input("\n  Tekan Enter untuk kembali...")


def _evaluate_all(trainer: ModelTrainer, evaluator: ModelEvaluator, cfg,
                  seed_tag: str = "") -> None:
    x_test, y_test, masks, restore_fn = _load_test_data(cfg)
    if x_test is None:
        return

    for loss_key in LOSS_FUNCTIONS:
        print(f"\n{'=' * 60}")
        print(f"  Evaluasi: {LOSS_FUNCTIONS[loss_key]}")
        if not trainer.weights_exist(loss_key, seed_tag=seed_tag):
            tag_info = f" [{seed_tag}]" if seed_tag else ""
            print(f"  [SKIP] Tidak ada bobot untuk '{loss_key}'{tag_info}.")
            continue
        try:
            model = trainer.load_weights(loss_key, seed_tag=seed_tag)
            evaluator.evaluate(model, loss_key, x_test, y_test, masks, restore_fn)
        except Exception as e:
            print(f"  [ERROR] {e}")

    input("\n  Selesai. Tekan Enter untuk kembali...")


def _load_test_data(cfg):
    try:
        if isinstance(cfg, DriveConfig):
            loader = DriveDataLoader(cfg)
            print("\n  Memuat data test DRIVE...")
            x_test, y_test, masks = loader.load_test()
            restore_fn = loader.restore_predictions
            print(f"  Test (padded): {x_test.shape}")
        else:
            loader = StareDataLoader(cfg)
            print("\n  Memuat data test STARE...")
            x_test, y_test = loader.load_test()
            masks = None
            restore_fn = loader.restore_predictions
            print(f"  Test: {x_test.shape}")
        return x_test, y_test, masks, restore_fn
    except Exception as e:
        print(f"\n  [ERROR] Gagal memuat data test: {e}")
        print("  Pastikan path dataset sudah benar di config.py")
        input("  Tekan Enter untuk kembali...")
        return None, None, None, None


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def _ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  Masukkan angka antara {lo}–{hi}.")


def _run_tuning_single(cfg, loss_key: str, n_trials: int, n_epochs: int,
                       auto_resume: bool = False) -> None:
    try:
        import optuna
    except ImportError:
        print("\n  [ERROR] optuna tidak terinstall.")
        print("          pip install optuna")
        input("  Tekan Enter untuk kembali...")
        return

    # Cek apakah ada sesi tuning sebelumnya di SQLite
    from config import RESULTS_DIR
    db_path = RESULTS_DIR / cfg.name.lower() / "tuning" / f"{loss_key}_tuning.db"
    if db_path.exists():
        study_name = f"{cfg.name.lower()}_{loss_key}"
        try:
            existing = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
            )
            finished = [t for t in existing.trials
                        if t.state.name in ("COMPLETE", "PRUNED")]
            n_done = len(finished)
            best   = existing.best_value if existing.best_trial else None
            print(f"\n  Ditemukan sesi tuning sebelumnya untuk '{loss_key}':")
            print(f"    Trials selesai : {n_done}")
            if best is not None:
                print(f"    Best F1 saat ini: {best:.6f}")

            if auto_resume:
                if n_done >= n_trials:
                    print(f"  [AUTO] Semua {n_trials} trials sudah selesai. Dilewati.")
                    return
                print(f"  [AUTO] Resume otomatis — melanjutkan dari trial #{n_done + 1}...")
            else:
                print()
                print("  [1] Lanjutkan (resume)")
                print("  [2] Mulai baru (hapus sesi lama)")
                while True:
                    raw = input("\n  Pilihan Anda [1/2]: ").strip()
                    if raw in ("1", "2"):
                        break
                    print("  Masukkan 1 atau 2.")
                if raw == "2":
                    db_path.unlink()
                    print("  Sesi lama dihapus. Memulai tuning baru...")
        except Exception:
            pass  # DB ada tapi tidak bisa dibaca — biarkan tuner tangani

    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    if loss_key == "skel_recall":
        if not _ensure_skel_dir(cfg):
            input("  Tekan Enter untuk kembali...")
            return
        y_train, y_val = _stack_skeleton(cfg, y_train, y_val)

    tuner = LossTuner(cfg, loss_key, x_train, y_train, x_val, y_val,
                      n_epochs=n_epochs)
    tuner.tune(n_trials=n_trials)
    if not auto_resume:
        input("\n  Tekan Enter untuk kembali...")


def _tuning_menu(cfg) -> None:
    _header(f"Hyperparameter Tuning — {cfg.name}")
    print("  Metode  : Optuna TPE (Tree-structured Parzen Estimator)")
    print("  Target  : Maksimalkan F1/Dice Score pada validation set\n")

    loss_items = list(LOSS_FUNCTIONS.items())
    options = [v for _, v in loss_items] + ["Tuning Semua Fungsi Loss"]

    choice = _prompt(options, back_label="Kembali")
    if choice == 0:
        return

    print()
    n_trials = _ask_int("Jumlah trials", default=30, lo=5, hi=200)
    n_epochs = _ask_int("Epochs per trial", default=30, lo=5, hi=150)

    if choice == len(options):
        for loss_key, loss_label in loss_items:
            print(f"\n{'=' * 60}")
            print(f"  {loss_label}")
            print('=' * 60)
            _run_tuning_single(cfg, loss_key, n_trials, n_epochs, auto_resume=True)
        input("\n  Semua tuning selesai. Tekan Enter untuk kembali...")
    else:
        loss_key, _ = loss_items[choice - 1]
        _run_tuning_single(cfg, loss_key, n_trials, n_epochs)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _reporting_menu(cfg, trainer: ModelTrainer) -> None:
    _header(f"Reporting & Analysis — {cfg.name}")
    print("  Output tersimpan di: results/<dataset>/reports/\n")

    base = RESULTS_DIR / cfg.name.lower() / "reports"

    choice = _prompt(
        [
            "4.1  Lingkungan Eksperimen",
            "4.2  Preprocessing Ablation Study",
            "4.3  CLAHE Parameter Tuning Study",
            "4.4  Training History Curves",
            "4.5  Segmentation Visualization Grid (Bahasa Indonesia)",
            "4.5b Segmentation Visualization Grid (English — Journal Format)",
            "4.6  Loss Function Comparison (Radar + Bar + Ranking)",
            "4.6b Radar Chart Comparison — DRIVE + STARE (English / Journal)",
            "4.6c Bar Chart Comparison — DRIVE + STARE (Journal, Helvetica 8pt)",
            "4.7  Training History Gabungan (DRIVE + STARE)",
            "4.8  Tuning Hyperparameter Loss — Regenerate Charts",
            "4.9  Computational Time Analysis (Epochs + Benchmark + Scatter)",
            "4.10 Multi-Seed Analysis + Statistical Significance (Wilcoxon)",
            "Generate Semua Report",
        ],
        back_label="Kembali",
    )
    if choice == 0:
        return
    elif choice == 1:
        _run_env_report(base / "section_4_1_environment")
    elif choice == 2:
        _run_ablation_report(base / "section_4_2_ablation")
    elif choice == 3:
        _run_clahe_tuning_report(base / "section_4_3_clahe_tuning")
    elif choice == 4:
        _run_history_report(cfg, trainer, base / "section_4_4_history")
    elif choice == 5:
        _run_visualization_report(cfg, trainer, base / "section_4_5_visualization")
    elif choice == 6:
        _run_visualization_report_en(cfg, trainer, base / "section_4_5b_visualization_en")
    elif choice == 7:
        _run_comparison_report(cfg, base / "section_4_6_comparison")
    elif choice == 8:
        _run_comparison_report_en(base / "section_4_6b_comparison_en")
    elif choice == 9:
        _run_comparison_bar_report(base / "section_4_6c_comparison_bar")
    elif choice == 10:
        _run_combined_history_report(base / "section_4_7_combined_history")
    elif choice == 11:
        _run_tuning_plot_report(base / "section_4_8_tuning_plots")
    elif choice == 12:
        _run_computational_time_report(base / "section_4_9_computational_time")
    elif choice == 13:
        _run_multiseed_report(cfg, base / "section_4_10_multiseed")
    elif choice == 14:
        _run_all_reports(cfg, trainer, base)


def _run_env_report(out_dir) -> None:
    reporter = EnvironmentReporter(out_dir)
    reporter.report()
    input("\n  Tekan Enter untuk kembali...")


def _run_ablation_report(out_dir) -> None:
    print("\n  Preprocessing Ablation Study")
    print("  Melatih BCE+MCC pada DRIVE dengan 2 kondisi preprocessing.")
    n_epochs = _ask_int("Epochs per kondisi (gunakan 150 untuk hasil publikasi)", 50, 5, 150)
    reporter = AblationReporter(out_dir)
    reporter.run(n_epochs=n_epochs)
    input("\n  Tekan Enter untuk kembali...")


def _run_clahe_tuning_report(out_dir) -> None:
    print("\n  CLAHE Parameter Tuning Study")
    print("  Pencarian sequential: Step A (5 clipLimit × tile=8) → Step B (2 tileGridSize).")
    print("  Total: 7 kombinasi × 2 dataset (DRIVE+STARE) = 14 training run.")
    print("  Setiap run: augment dari nol → train BCE+MCC → evaluate.\n")
    n_epochs = _ask_int("Epochs per run (gunakan 150 untuk hasil publikasi)", 50, 5, 150)
    reporter = ClaheTuningReporter(out_dir)
    reporter.run(n_epochs=n_epochs)
    input("\n  Tekan Enter untuk kembali...")


def _run_history_report(cfg, trainer: ModelTrainer, out_dir) -> None:
    ms_summary = _load_multiseed_summary(cfg.name.lower())
    seeds      = _seeds_from_summary(ms_summary)
    reporter   = HistoryReporter(cfg, trainer, out_dir, seeds=seeds or None)
    paths = reporter.generate_all()
    if paths:
        print(f"\n  {len(paths)} kurva training tersimpan.")
    input("\n  Tekan Enter untuk kembali...")


def _run_visualization_report(cfg, trainer: ModelTrainer, out_dir) -> None:
    x_test, y_test, masks, restore_fn = _load_test_data(cfg)
    if x_test is None:
        return
    n_max = len(x_test)
    n     = min(_ask_int(f"Jumlah gambar sampel (max {n_max})", 3, 1, n_max), n_max)
    reporter = VisualizationReporter(cfg, out_dir, n_samples=n)
    paths = reporter.generate(x_test, y_test, list(range(n)))
    if paths:
        print(f"\n  {len(paths)} file visualisasi tersimpan (1 file per sampel).")
    input("\n  Tekan Enter untuk kembali...")


def _run_visualization_report_en(cfg, _trainer: ModelTrainer, out_dir) -> None:
    x_test, y_test, masks, restore_fn = _load_test_data(cfg)
    if x_test is None:
        return
    n_max = len(x_test)
    n     = min(_ask_int(f"Jumlah gambar sampel (max {n_max})", 3, 1, n_max), n_max)
    reporter = VisualizationReporter(cfg, out_dir, n_samples=n, lang="en")
    paths = reporter.generate(x_test, y_test, list(range(n)))
    if paths:
        print(f"\n  {len(paths)} file visualisasi (EN) tersimpan.")
    input("\n  Tekan Enter untuk kembali...")


def _run_comparison_report(cfg, out_dir) -> None:
    ms_summary = _load_multiseed_summary(cfg.name.lower())
    reporter   = ComparisonReporter(out_dir, multiseed_summary=ms_summary)
    paths = reporter.generate_all(dataset=cfg.name.lower())
    if paths:
        print(f"\n  {len(paths)} file comparison tersimpan.")
    input("\n  Tekan Enter untuk kembali...")


def _run_comparison_report_en(out_dir) -> None:
    ms = _load_multiseed_summaries()
    reporter = ComparisonReporterEN(out_dir, multiseed_summaries=ms or None)
    path = reporter.generate()
    if path:
        print(f"\n  File tersimpan: {path}")
    input("\n  Tekan Enter untuk kembali...")


def _run_comparison_bar_report(out_dir) -> None:
    ms = _load_multiseed_summaries()
    reporter = ComparisonBarReporter(out_dir, multiseed_summaries=ms or None)
    paths = reporter.generate()
    if paths:
        print(f"\n  {len(paths)} file bar chart tersimpan.")
    input("\n  Tekan Enter untuk kembali...")


def _run_combined_history_report(out_dir) -> None:
    ms_drive = _load_multiseed_summary("drive")
    seeds    = _seeds_from_summary(ms_drive)
    reporter = CombinedHistoryReporter(out_dir, seeds=seeds or None)
    path = reporter.generate()
    if path:
        print(f"\n  File tersimpan: {path}")
    input("\n  Tekan Enter untuk kembali...")


def _run_tuning_plot_report(out_dir) -> None:
    print("\n  Regenerate Hyperparameter Tuning Charts")
    print("  Memuat DB Optuna yang sudah ada dan membuat ulang 3 chart per fungsi loss.")
    print("  (Tidak melakukan tuning ulang — hanya regenerasi file PNG)\n")
    reporter = TuningPlotReporter(out_dir)
    paths = reporter.regenerate_all()
    if paths:
        print(f"\n  {len(paths)} file chart berhasil diregenerasi.")
    else:
        print("\n  [WARN] Tidak ada chart yang diregenerasi. Pastikan DB tuning sudah ada.")
    input("\n  Tekan Enter untuk kembali...")


def _run_computational_time_report(out_dir) -> None:
    print("\n  Computational Time Analysis")
    print("  [1] Epochs to Convergence — dari history JSON yang sudah ada")
    print("  [2] Micro-benchmark Training Step — SA-UNetV2 + GradientTape, data training nyata")
    print("  [3] Scatter Plot — Epochs vs F1 Score")
    print("\n  Analisis (1) dan (3) tidak memerlukan GPU.")
    print("  Analisis (2) memerlukan GPU; pastikan sesi Colab terhubung ke GPU.\n")
    ms         = _load_multiseed_summaries()
    ms_drive   = ms.get("drive")
    seeds      = _seeds_from_summary(ms_drive)
    reporter   = ComputationalTimeReporter(
        out_dir,
        multiseed_summaries=ms or None,
        seeds=seeds or None,
    )
    paths = reporter.generate()
    if paths:
        print(f"\n  {len(paths)} file tersimpan di: {out_dir}")
    input("\n  Tekan Enter untuk kembali...")


def _run_multiseed_report(cfg, out_dir) -> None:
    from src.reporting.multiseed_reporter import _REQUIRED_SEEDS
    print("\n  Multi-Seed Analysis & Statistical Significance")
    print("  Memuat hasil evaluasi per seed, menghitung mean ± SD,")
    print("  dan menjalankan uji Wilcoxon signed-rank (per-image, 5-seed averaged).\n")
    print(f"  Required seeds: {_REQUIRED_SEEDS}")
    print("  Prerequisite: Evaluasi sudah dijalankan untuk SEMUA seed dan loss function.")
    print(f"  Contoh file: bce_mcc_seed42_results.json\n")

    # Confirm seeds (default to required seeds)
    raw = input(
        f"  Seeds [{' '.join(str(s) for s in _REQUIRED_SEEDS)}]: "
    ).strip()
    if raw:
        seeds = []
        for s in raw.split():
            try:
                seeds.append(int(s))
            except ValueError:
                pass
        if not seeds:
            print("  [ERROR] Tidak ada seed valid.")
            input("  Tekan Enter...")
            return
    else:
        seeds = _REQUIRED_SEEDS
    print(f"  Seeds yang digunakan: {seeds}")

    # Save to dataset-specific subdir so DRIVE/STARE don't overwrite each other
    ds_out_dir = out_dir / cfg.name.lower()
    try:
        reporter = MultiSeedReporter(cfg, ds_out_dir, seeds)
        paths = reporter.generate()
    except FileNotFoundError as e:
        print(str(e))
        input("\n  Tekan Enter untuk kembali...")
        return
    except ValueError as e:
        print(f"\n  [ERROR] {e}")
        input("\n  Tekan Enter untuk kembali...")
        return

    if paths:
        print(f"\n  {len(paths)} file tersimpan di: {out_dir}")
    input("\n  Tekan Enter untuk kembali...")


def _run_all_reports(cfg, trainer: ModelTrainer, base) -> None:
    print("\n  Generating semua reports...\n")

    ms         = _load_multiseed_summaries()
    ms_cfg     = _load_multiseed_summary(cfg.name.lower())
    seeds      = _seeds_from_summary(ms_cfg)

    _run_env_report(base / "section_4_1_environment")
    HistoryReporter(
        cfg, trainer, base / "section_4_4_history",
        seeds=seeds or None,
    ).generate_all()

    x_test, y_test, masks, _ = _load_test_data(cfg)
    if x_test is not None:
        vis_paths = VisualizationReporter(
            cfg, base / "section_4_5_visualization"
        ).generate(x_test, y_test, list(range(min(3, len(x_test)))))
        if vis_paths:
            print(f"  {len(vis_paths)} file visualisasi tersimpan.")

    ComparisonReporter(
        base / "section_4_6_comparison",
        multiseed_summary=ms_cfg,
    ).generate_all(cfg.name.lower())
    ComparisonReporterEN(
        base / "section_4_6b_comparison_en",
        multiseed_summaries=ms or None,
    ).generate()
    ComparisonBarReporter(
        base / "section_4_6c_comparison_bar",
        multiseed_summaries=ms or None,
    ).generate()
    CombinedHistoryReporter(
        base / "section_4_7_combined_history",
        seeds=seeds or None,
    ).generate()
    print(f"\n  Semua reports tersimpan di: {base}")
    input("\n  Tekan Enter untuk kembali...")


# ── Dataset Sub-menu ──────────────────────────────────────────────────────────

def _dataset_menu(dataset_name: str) -> None:
    cfg: Union[DriveConfig, StareConfig]
    if dataset_name == "DRIVE":
        cfg = DriveConfig()
    else:
        cfg = StareConfig()

    trainer = ModelTrainer(cfg)
    evaluator = ModelEvaluator(cfg)

    while True:
        _clear()
        _header(f"Dataset: {dataset_name}")
        choice = _prompt(
            [
                "Augmentasi Dataset",
                "Training Model",
                "Evaluasi Model",
                "Hyperparameter Tuning",
                "Reporting & Analysis",
            ],
            back_label="Kembali ke Main Menu",
        )

        if choice == 0:
            break
        elif choice == 1:
            if dataset_name == "DRIVE":
                _run_augmentation_drive()
            else:
                _run_augmentation_stare()
        elif choice == 2:
            _loss_selection_menu(trainer, cfg)
        elif choice == 3:
            _evaluation_menu(trainer, evaluator, cfg)
        elif choice == 4:
            _tuning_menu(cfg)
        elif choice == 5:
            _reporting_menu(cfg, trainer)


# ── All Results View ──────────────────────────────────────────────────────────

_METRIC_KEYS = [
    "accuracy", "sensitivity", "specificity", "auc",
    "mcc", "f1", "jaccard", "cldice", "betti0_error", "betti1_error",
]
_METRIC_HEADERS = [
    "Loss Function", "Accuracy (%)", "Sensitivity (%)", "Specificity (%)",
    "AUC (%)", "MCC (%)", "F1 (%)", "Jaccard (%)", "clDice (%)", "β0 Err", "β1 Err",
]
_LOWER_IS_BETTER = {"betti0_error", "betti1_error"}


def _save_results_excel(all_data: dict) -> Optional[Path]:
    """Save evaluation results to Excel with best-value cells bolded."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        print("\n  [WARN] openpyxl tidak terinstall, skip export Excel.")
        print("         Install dengan: pip install openpyxl")
        return None

    bold = Font(bold=True)
    header_font = Font(bold=True)
    wb = Workbook()
    wb.remove(wb.active)

    for dataset_name, data in all_data.items():
        ws = wb.create_sheet(title=dataset_name)
        str_rows = data["str_rows"]
        num_rows = data["num_rows"]

        # Header row
        for c, h in enumerate(_METRIC_HEADERS, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = header_font

        # Data rows — loss label as string, metrics as numeric
        for r, (str_row, num_row) in enumerate(zip(str_rows, num_rows), 2):
            ws.cell(row=r, column=1, value=str_row[0])
            for c, val in enumerate(num_row, 2):
                ws.cell(row=r, column=c, value=val)

        # Bold best value per metric column
        for col_offset, key in enumerate(_METRIC_KEYS):
            col_idx = col_offset + 2  # skip "Loss Function" column
            vals = [num_rows[r][col_offset] for r in range(len(num_rows))]
            valid = [(i, v) for i, v in enumerate(vals)
                     if v is not None and v == v]  # exclude None and NaN
            if not valid:
                continue
            best = (min if key in _LOWER_IS_BETTER else max)(v for _, v in valid)
            for i, v in valid:
                if abs(v - best) < 1e-9:
                    ws.cell(row=i + 2, column=col_idx).font = bold

        # Auto column width
        for col in ws.columns:
            width = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = width + 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"evaluation_results_{timestamp}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))
    return out_path


def _view_all_results() -> None:
    _clear()
    _header("Semua Hasil Evaluasi")

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    all_data: dict = {}

    for dataset in ("drive", "stare"):
        result_dir = RESULTS_DIR / dataset
        if not result_dir.is_dir():
            continue

        print(f"\n  ── Dataset: {dataset.upper()} ──\n")
        str_rows = []
        num_rows = []

        for loss_key, loss_label in LOSS_FUNCTIONS.items():
            path = result_dir / f"{loss_key}_results.json"
            if not path.exists():
                str_rows.append([loss_label] + ["—"] * 10)
                num_rows.append([None] * 10)
                continue

            with open(path) as f:
                m = json.load(f)

            nums = [m.get(k) for k in _METRIC_KEYS]
            num_rows.append(nums)
            str_rows.append([
                loss_label,
                f"{m.get('accuracy',     float('nan')):.2f}",
                f"{m.get('sensitivity',  float('nan')):.2f}",
                f"{m.get('specificity',  float('nan')):.2f}",
                f"{m.get('auc',          float('nan')):.2f}",
                f"{m.get('mcc',          float('nan')):.2f}",
                f"{m.get('f1',           float('nan')):.2f}",
                f"{m.get('jaccard',      float('nan')):.2f}",
                f"{m.get('cldice',       float('nan')):.2f}",
                f"{m.get('betti0_error', float('nan')):.2f}",
                f"{m.get('betti1_error', float('nan')):.2f}",
            ])

        all_data[dataset.upper()] = {"str_rows": str_rows, "num_rows": num_rows}

        if tabulate:
            print(tabulate(str_rows, headers=_METRIC_HEADERS, tablefmt="rounded_outline"))
        else:
            col_w = [max(len(str(r[i])) for r in [_METRIC_HEADERS] + str_rows)
                     for i in range(len(_METRIC_HEADERS))]
            fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
            print(fmt.format(*_METRIC_HEADERS))
            print("  " + "  ".join("-" * w for w in col_w))
            for row in str_rows:
                print(fmt.format(*row))

    if all_data:
        excel_path = _save_results_excel(all_data)
        if excel_path:
            print(f"\n  Tabel Excel tersimpan di: {excel_path}")

    input("\n\n  Tekan Enter untuk kembali ke Main Menu...")


# ── Main Menu ─────────────────────────────────────────────────────────────────

def run_main_menu() -> None:
    while True:
        _clear()
        _header("SA-UNetV2 Loss Function Comparison")
        print("  Penelitian komparasi fungsi loss untuk segmentasi pembuluh darah retina\n")

        choice = _prompt(
            [
                "Select Dataset (DRIVE)",
                "Select Dataset (STARE)",
                "View All Evaluation Results",
            ],
            back_label="Exit",
        )

        if choice == 0:
            print("\n  Keluar dari program. Sampai jumpa!")
            sys.exit(0)
        elif choice == 1:
            _dataset_menu("DRIVE")
        elif choice == 2:
            _dataset_menu("STARE")
        elif choice == 3:
            _view_all_results()
