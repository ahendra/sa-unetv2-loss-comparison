import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from config import DriveConfig, StareConfig, LOSS_FUNCTIONS, RESULTS_DIR
from src.data import RetinalAugmentationRunner, DriveDataLoader, StareDataLoader
from src.evaluation import ModelEvaluator
from src.losses import get_loss_function
from src.training import ModelTrainer
from src.tuning import LossTuner


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
    except ImportError:
        os.system('cls' if os.name == 'nt' else 'clear')


def _prompt(options: list[str], back_label: str = "Back") -> int:
    """Display a numbered menu and return the 1-based choice (0 = not matched)."""
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    if back_label:
        print(f"  [0] {back_label}")
    while True:
        raw = input("\n  Pilihan Anda: ").strip()
        if raw.isdigit():
            choice = int(raw)
            if 0 <= choice <= len(options):
                return choice
        print("  Input tidak valid. Masukkan angka yang tersedia.")


# ── Augmentation ──────────────────────────────────────────────────────────────

def _run_augmentation_drive(cfg: DriveConfig) -> None:
    _header("Augmentasi Dataset — DRIVE")
    print(f"  Sumber gambar : {cfg.train_images}")
    print(f"  Sumber label  : {cfg.train_labels}")
    print(f"  Output        : {cfg.aug_dir}")
    print()
    print("  Spesifikasi augmentasi (SA-UNetV2 paper):")
    print("    randomRotation ×3, randomColor ×3, randomGaussian ×3")
    print("    h-flip ×1, v-flip ×1, hv-flip ×1  →  12 augmentasi/gambar")
    print("    + copy gambar original → total 260 gambar")
    print("    Split: 234 train / 26 validate (proporsional per tipe)\n")

    if not Path(cfg.train_images).is_dir():
        print(f"  [ERROR] Direktori tidak ditemukan: {cfg.train_images}")
        input("  Tekan Enter untuk kembali...")
        return

    aug_train = Path(cfg.aug_train_images)
    if aug_train.exists() and any(aug_train.iterdir()):
        ans = input("  Folder aug/train sudah ada. Hapus dan buat ulang? (y/n): ").strip().lower()
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

    runner = RetinalAugmentationRunner()
    runner.run(
        src_img_dir    = cfg.train_images,
        src_lbl_dir    = cfg.train_labels,
        aug_base_dir   = cfg.aug_dir,
        label_suffix_fn= drive_label_fn,
    )
    input("\n  Selesai. Tekan Enter untuk kembali...")


def _run_augmentation_stare(cfg: StareConfig) -> None:
    _header("Augmentasi Dataset — STARE")
    print(f"  Sumber gambar : {cfg.train_images}")
    print(f"  Sumber label  : {cfg.train_labels}")
    print(f"  Output        : {cfg.aug_dir}")
    print()
    print("  Spesifikasi augmentasi (SA-UNetV2 paper — STARE):")
    print("    randomRotation ×3, randomColor ×3, randomGaussian ×3")
    print("    h-flip ×1, v-flip ×1, hv-flip ×1  →  12 augmentasi/gambar")
    print("    + copy gambar original → total 208 gambar (16 asli × 13)")
    print("    Split: 187 train / 21 validate (random, random_state=42)\n")

    if not Path(cfg.train_images).is_dir():
        print(f"  [ERROR] Direktori tidak ditemukan: {cfg.train_images}")
        input("  Tekan Enter untuk kembali...")
        return

    aug_train = Path(cfg.aug_train_images)
    if aug_train.exists() and any(aug_train.iterdir()):
        ans = input("  Folder aug/train sudah ada. Hapus dan buat ulang? (y/n): ").strip().lower()
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

    runner = RetinalAugmentationRunner()
    runner.run(
        src_img_dir    = cfg.train_images,
        src_lbl_dir    = cfg.train_labels,
        aug_base_dir   = cfg.aug_dir,
        label_suffix_fn= stare_label_fn,
    )
    input("\n  Selesai. Tekan Enter untuk kembali...")


# ── Training ──────────────────────────────────────────────────────────────────

def _loss_selection_menu(trainer: ModelTrainer, cfg) -> None:
    _header(f"Training Model — {cfg.name}")
    print("  Pilih fungsi loss:\n")

    loss_items = list(LOSS_FUNCTIONS.items())
    options = [f"{v}" for _, v in loss_items] + ["Semua Fungsi Loss (Train All)"]

    choice = _prompt(options, back_label="Kembali ke menu dataset")
    if choice == 0:
        return
    if choice == len(options):
        _train_all(trainer, cfg)
    else:
        key, label = loss_items[choice - 1]
        _train_single(trainer, cfg, key, label)


def _train_single(trainer: ModelTrainer, cfg, loss_key: str, loss_label: str) -> None:
    print(f"\n  Loss: {loss_label}")
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

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

    for loss_key, loss_label in LOSS_FUNCTIONS.items():
        print(f"\n{'=' * 60}")
        print(f"  [{list(LOSS_FUNCTIONS.keys()).index(loss_key)+1}/{len(LOSS_FUNCTIONS)}] {loss_label}")
        print('=' * 60)

        if trainer.weights_exist(loss_key):
            ans = input(f"  Bobot '{loss_key}' sudah ada. Lewati? (y/n) [y]: ").strip().lower()
            if ans != 'n':
                print("  Dilewati.")
                continue

        loss_fn = get_loss_function(loss_key)
        trainer.train(loss_key, loss_fn, x_train, y_train, x_val, y_val)

    input("\n  Semua training selesai. Tekan Enter untuk kembali...")


def _load_training_data(cfg):
    try:
        if isinstance(cfg, DriveConfig):
            loader = DriveDataLoader(cfg)
            print("\n  Memuat data training DRIVE (aug)...")
            x_train, y_train = loader.load_train()
            print(f"  Train: {x_train.shape}")
            print("  Memuat data validasi DRIVE...")
            x_val, y_val = loader.load_validate()
            print(f"  Validasi: {x_val.shape}")
        else:
            loader = StareDataLoader(cfg)
            print("\n  Memuat data training STARE (aug/train)...")
            x_train, y_train = loader.load_train()
            print(f"  Train: {x_train.shape}")
            print("  Memuat data validasi STARE (aug/validate)...")
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
    print("  Pilih loss/model yang akan dievaluasi:\n")

    loss_items = list(LOSS_FUNCTIONS.items())
    options = [
        f"{v}"
        + (" [✓ bobot]" if trainer.weights_exist(k) else "")
        + (" [✓ hasil]" if evaluator.results_exist(k) else "")
        for k, v in loss_items
    ] + ["Evaluasi Semua Model"]

    choice = _prompt(options, back_label="Kembali ke menu dataset")
    if choice == 0:
        return
    if choice == len(options):
        _evaluate_all(trainer, evaluator, cfg)
    else:
        key, label = loss_items[choice - 1]
        _evaluate_single(trainer, evaluator, cfg, key)


def _evaluate_single(trainer: ModelTrainer, evaluator: ModelEvaluator, cfg, loss_key: str) -> None:
    if not trainer.weights_exist(loss_key):
        print(f"\n  [ERROR] Bobot untuk '{loss_key}' tidak ditemukan. Lakukan training terlebih dahulu.")
        input("  Tekan Enter untuk kembali...")
        return

    try:
        print(f"\n  Memuat model dengan bobot '{loss_key}'...")
        model = trainer.load_weights(loss_key)

        x_test, y_test, masks, restore_fn = _load_test_data(cfg)
        if x_test is None:
            return

        evaluator.evaluate(model, loss_key, x_test, y_test, masks, restore_fn)
    except Exception as e:
        print(f"\n  [ERROR] Evaluasi gagal: {e}")

    input("\n  Tekan Enter untuk kembali...")


def _evaluate_all(trainer: ModelTrainer, evaluator: ModelEvaluator, cfg) -> None:
    x_test, y_test, masks, restore_fn = _load_test_data(cfg)
    if x_test is None:
        return

    for loss_key in LOSS_FUNCTIONS:
        print(f"\n{'=' * 60}")
        print(f"  Evaluasi: {LOSS_FUNCTIONS[loss_key]}")
        if not trainer.weights_exist(loss_key):
            print(f"  [SKIP] Tidak ada bobot untuk '{loss_key}'.")
            continue
        try:
            model = trainer.load_weights(loss_key)
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


def _run_tuning_single(cfg, loss_key: str, n_trials: int, n_epochs: int) -> None:
    x_train, y_train, x_val, y_val = _load_training_data(cfg)
    if x_train is None:
        return

    try:
        import optuna  # noqa: F401
    except ImportError:
        print("\n  [ERROR] optuna tidak terinstall.")
        print("          pip install optuna")
        input("  Tekan Enter untuk kembali...")
        return

    tuner = LossTuner(cfg, loss_key, x_train, y_train, x_val, y_val,
                      n_epochs=n_epochs)
    tuner.tune(n_trials=n_trials)
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
            _run_tuning_single(cfg, loss_key, n_trials, n_epochs)
    else:
        loss_key, _ = loss_items[choice - 1]
        _run_tuning_single(cfg, loss_key, n_trials, n_epochs)


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
            ],
            back_label="Kembali ke Main Menu",
        )

        if choice == 0:
            break
        elif choice == 1:
            if dataset_name == "DRIVE":
                _run_augmentation_drive(cfg)
            else:
                _run_augmentation_stare(cfg)
        elif choice == 2:
            _loss_selection_menu(trainer, cfg)
        elif choice == 3:
            _evaluation_menu(trainer, evaluator, cfg)
        elif choice == 4:
            _tuning_menu(cfg)


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
