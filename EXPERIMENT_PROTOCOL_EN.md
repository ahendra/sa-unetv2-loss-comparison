# Experiment Execution Protocol — SA-UNetV2 Loss Comparison

This document describes the correct and mandatory execution order to fully reproduce all experimental results for the retinal vessel segmentation loss function comparison study using the SA-UNetV2 architecture.

---

## General Information

| Property | Value |
|---|---|
| Datasets | DRIVE (20 images), STARE (16 images) |
| Loss Functions | 7 variants (bce\_mcc, dice, focal, cldice, dice\_ssim, bce\_ssim, skel\_recall) |
| Multi-Seed Seeds | 42, 123, 456, 789, 2026 |
| Total Phases | 11 (strict order required) |
| Training Epochs | 150 (max, with early stopping) |
| Optimizer | Adam (lr=1e-3) |

---

## ⚠ Critical Rules — Read Before Starting

1. **`RANDOM_SEED = 42`** in `config.py` controls the train/val image split (augmentation) AND the Optuna TPE sampler. **Never change this value** — doing so invalidates reproducibility across all experiments.

2. After **Phase 1** (CLAHE Tuning) and **Phase 5** (Loss HP Tuning), `config.py` **must be updated manually** before the next phase can proceed correctly.

3. For multi-seed experiments, use the `EXPERIMENT_SEED` environment variable — never change `RANDOM_SEED`. The multi-seed menu handles this automatically.

4. **Report Section 4.10 (Multi-Seed) must be generated FIRST** among all reports — it produces `multiseed_summary.json` which is consumed automatically by all other reporters.

5. If the dataset split fix has been applied, delete old `aug_clahe/` and `aug_rgb/` directories before re-augmenting.

---

## Execution Order Summary

```
Phase 0  → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
              ↓
        (manual: update config.py CLAHE params)
                               ↓
                        (manual: update config.py LOSS_PARAMS)
                                              ↓
                               Phase 6 → Phase 7 → Phase 8
                                              ↓
                                    Phase 9 → Phase 10 → Phase 11
```

| Phase | Name | Type | Thesis Section |
|---|---|---|---|
| 0 | Prerequisites & Setup | Manual | — |
| 1 | CLAHE Parameter Tuning | Automated | § 4.3 |
| 2 | Update config.py (CLAHE) | **Manual** | — |
| 3 | Main Augmentation | Automated | — |
| 4 | Preprocessing Ablation Study | Automated | § 4.2 |
| 5 | Loss Hyperparameter Tuning | Automated | § 4.8 |
| 6 | Update config.py (LOSS\_PARAMS) | **Manual** | — |
| 7 | Main Training | Automated | § 4.4 |
| 8 | Main Evaluation | Automated | § 4.6 |
| 9 | Multi-Seed Training | Automated | § 4.10 |
| 10 | Multi-Seed Evaluation | Automated | § 4.10 |
| 11 | Generate All Reports | Automated | § 4.1–4.10 |

---

## Phase 0 — Prerequisites & Setup

### Required dataset directory structure

```
datasets/
├── DRIVE/
│   ├── train/images/   ← 20 .tif files (21_training.tif … 40_training.tif)
│   ├── train/labels/   ← 20 *_manual1.gif files
│   ├── test/images/    ← 20 .tif files
│   ├── test/labels/    ← 20 *_manual1.gif files
│   └── test/mask/      ← 20 *_test_mask.gif files (FOV masks)
└── STARE/
    ├── train/images/   ← 16 .ppm files
    ├── train/labels/   ← 16 *.ah.ppm files
    ├── test/images/    ← remaining .ppm files
    └── test/labels/    ← *.ah.ppm labels
```

### Delete old augmented directories (if split fix has been applied)

```powershell
Remove-Item -Recurse -Force datasets\DRIVE\aug_clahe
Remove-Item -Recurse -Force datasets\DRIVE\aug_rgb
Remove-Item -Recurse -Force datasets\STARE\aug_clahe
Remove-Item -Recurse -Force datasets\STARE\aug_rgb
```

> Skip this step if running from scratch (no existing aug directories). The CLAHE tuning reporter creates its own isolated directories (`aug_clip1p5_tile8/` etc.) and is not affected by deleting `aug_clahe/`.

### Launch the program

```bash
cd sa_unetv2_loss_comparison
python main.py
```

---

## Phase 1 — CLAHE Parameter Tuning (§ 4.3)

**Purpose:** Determine the optimal `clipLimit` and `tileGridSize` values for CLAHE preprocessing used throughout all subsequent experiments.

**Why first?** This phase determines the CLAHE parameters used in the main augmentation (Phase 3). Each combination gets its own isolated augmentation directory (`aug_clip1p5_tile8/` etc.), so it does not interfere with the main `aug_clahe/` directory.

### Search specification

| | Step A | Step B |
|---|---|---|
| Varied parameter | clipLimit ∈ {1.0, 1.5, 2.0, 3.0, 4.0} | tileGridSize ∈ {4, 16} |
| Fixed parameter | tileGridSize = 8 | clipLimit = best from Step A |
| Number of runs | 5 combinations × 2 datasets | 2 combinations × 2 datasets |

**Total: 7 combinations × 2 datasets = 14 training runs, each with 150 epochs.**

**Composite score** used to select the best combination:
```
Score = 0.35 × F1 + 0.25 × AUC + 0.20 × Sensitivity − 0.20 × β0_error
(all metrics are min-max normalized per dataset before weighting)
```

### Menu

```
Any dataset → [5] Reporting & Analysis
  → [4.3] CLAHE Parameter Tuning Study
  → Epochs per run: 150
```

The reporter runs both DRIVE and STARE automatically in one invocation. If interrupted, restart — runs with existing weights and result JSON are skipped automatically.

### Output

```
results/drive/reports/section_4_3_clahe_tuning/
├── clahe_tuning_results.json    ← best_clip, best_tile, and all combination scores
├── clahe_tuning_drive.png       ← bar chart DRIVE (7 combinations × 10 metrics)
└── clahe_tuning_stare.png       ← bar chart STARE

weights/drive/
└── drive_clahe_tune_clip*_tile*.weights.h5   ← per combination

results/drive/
└── clahe_tune_*_results.json                 ← per combination metrics
```

### Action after completion

Open `clahe_tuning_results.json`, read `best_clip` and `best_tile`. Proceed to **Phase 2**.

---

## Phase 2 — Update config.py (CLAHE Parameters)

**Type: Manual**

Edit `config.py` in both `DriveConfig` and `StareConfig`:

```python
# Replace with values from clahe_tuning_results.json → best_clip and best_tile
clahe_clip_limit: float = 1.5   # ← best_clip from Phase 1
clahe_tile_grid: int    = 16    # ← best_tile from Phase 1
```

> **Note:** The current defaults (`clip=1.5, tile=16`) are the values from the included tuning run. Update only if re-running Phase 1 produces different results.

---

## Phase 3 — Main Augmentation

**Purpose:** Generate the canonical `aug_clahe/` directories used by all training, loss tuning, and multi-seed phases.

### Augmentation specification

| | DRIVE | STARE |
|---|---|---|
| Original train images | 18 images | 14 images |
| Original val images | 2 images | 2 images |
| Augmented files per image | 12 variants + 1 original copy = 13 files | same |
| Total train files | 18 × 13 = 234 | 14 × 13 = 182 |
| Total val files | 2 × 13 = 26 | 2 × 13 = 26 |

The split is performed at the level of original images using `RANDOM_SEED=42`, before augmentation. CLAHE preprocessing is baked into each saved file at generation time — it is not applied at load time.

### Menu — DRIVE

```
[1] DRIVE → [1] Augmentasi Dataset
  → [1] RGB + CLAHE → aug_clahe/
  → Confirm: y
```

### Menu — STARE

```
[2] STARE → [1] Augmentasi Dataset
  → [1] RGB + CLAHE → aug_clahe/
  → Confirm: y
```

---

## Phase 4 — Preprocessing Ablation Study (§ 4.2)

**Purpose:** Compare RGB Original vs RGB + CLAHE (with optimal parameters) to demonstrate the effectiveness of CLAHE preprocessing.

**Why after Phase 3?** The ablation uses the same `aug_clahe/` directory created in Phase 3, ensuring the CLAHE condition uses the optimal parameters from Phase 1. It also auto-generates `aug_rgb/` if missing.

> Ablation weights are saved to `weights/ablation/` and do not overwrite main experiment weights.

### Menu

```
[1] DRIVE → [5] Reporting & Analysis
  → [4.2] Preprocessing Ablation Study
  → Epochs per condition: 150
```

### Output

```
results/DRIVE/reports/section_4_2_ablation/
├── preprocessing_ablation.json              ← per-condition metrics
├── preprocessing_ablation_table.txt         ← formatted text table (LaTeX-ready)
├── preprocessing_ablation.png               ← bar chart, 10 metrics × 2 conditions, DPI 300
└── preprocessing_sample_comparison.png      ← RGB vs CLAHE visual, 3 test images
```

---

## Phase 5 — Loss Function Hyperparameter Tuning (§ 4.8)

**Purpose:** Find optimal hyperparameters for each loss function using Optuna TPE.

**Depends on Phase 3.** Tuning loads data from `aug_clahe/`. If not yet present, the menu will offer to generate it automatically — ensure `config.py` CLAHE params have already been updated (Phase 2) before that happens.

### Specification

| Parameter | Value |
|---|---|
| Method | Optuna TPE + MedianPruner |
| Objective | Maximize F1 Score on validation set |
| Trials per loss | 30 |
| Epochs per trial | 50 |
| Resume | SQLite DB — safe to interrupt and continue |

### Search space per loss function

| Loss | Parameter | Range |
|---|---|---|
| `bce_mcc` | lambda\_bce | [0.3, 0.7]; lambda\_mcc = 1 − lambda\_bce |
| `dice` | smooth | [1e-7, 1.0] log-scale |
| `focal` | alpha | [0.1, 0.9] |
| `focal` | gamma | [0.5, 5.0] |
| `focal` | smooth | [1e-7, 1e-3] log-scale |
| `cldice` | alpha | [0.3, 0.7] |
| `cldice` | iters | [5, 30] |
| `cldice` | smooth | [1e-7, 1.0] log-scale |
| `dice_ssim` | lambda\_dice | [0.3, 0.7]; lambda\_ssim = 1 − lambda\_dice |
| `dice_ssim` | smooth | [1e-7, 1.0] log-scale |
| `bce_ssim` | lambda\_bce | [0.3, 0.7]; lambda\_ssim = 1 − lambda\_bce |
| `skel_recall` | weight\_srec | [0.1, 10.0] log-scale |
| `skel_recall` | smooth | [1e-7, 1e-3] log-scale |

### Menu — Tune all losses (recommended)

```
[1] DRIVE → [4] Hyperparameter Tuning
  → [Tuning Semua Fungsi Loss]
  → Number of trials: 30
  → Epochs per trial: 50
```

Repeat for STARE. If interrupted, restart and choose **resume** to continue from the last completed trial.

### Output (per loss, per dataset)

```
results/drive/tuning/
├── {loss_key}_tuning.json           ← best_params + full trial history
├── {loss_key}_tuning.db             ← SQLite DB (for resume)
├── {loss_key}_tuning_history.png    ← optimization history curve
├── {loss_key}_tuning_importance.png ← fANOVA parameter importance
└── {loss_key}_tuning_scatter.png    ← per-parameter vs F1 scatter
```

### Action after completion

Open each `{loss_key}_tuning.json` and read `best_params`. The terminal also prints a ready-to-paste code block. Proceed to **Phase 6**.

---

## Phase 6 — Update config.py (LOSS\_PARAMS)

**Type: Manual**

Edit `config.py`, copy `best_params` from each tuning JSON into `LOSS_PARAMS`:

```python
LOSS_PARAMS = {
    "bce_mcc": {
        "lambda_bce": 0.5962,   # ← from bce_mcc_tuning.json → best_params
        "lambda_mcc": 0.4038,
    },
    "dice": {
        "smooth": 0.001558,
    },
    "focal": {
        "alpha":  0.4927,
        "gamma":  0.5769,
        "smooth": 2.44e-05,
    },
    "cldice": {
        "alpha":  0.3041,
        "iters":  19,
        "smooth": 2.55e-05,
    },
    "dice_ssim": {
        "lambda_dice": 0.6277,
        "smooth":      1.23e-07,
        "lambda_ssim": 0.3723,
    },
    "bce_ssim": {
        "lambda_bce":  0.5473,
        "lambda_ssim": 0.4527,
    },
    "skel_recall": {
        "weight_srec": 1.0,
        "smooth":      1e-5,
    },
}
```

> **Important:** `LOSS_PARAMS` applies globally to both DRIVE and STARE training. If DRIVE and STARE tuning produced different best parameters, decide which set to use (or run experiments separately). Currently, the DRIVE-tuned values are used for both datasets.

---

## Phase 7 — Main Training (§ 4.4)

**Purpose:** Train SA-UNetV2 for all 7 loss functions on both datasets using the default seed.

| Parameter | DRIVE | STARE |
|---|---|---|
| Batch size | 8 | 2 |
| Max epochs | 150 | 150 |
| EarlyStopping patience | 20 epochs | 30 epochs |
| Checkpoint monitor | val\_loss | val\_loss |
| ReduceLR patience | 10 epochs | 20 epochs |

### Menu — DRIVE

```
[1] DRIVE → [2] Training Model
  → [Semua Fungsi Loss (Train All)]
```

When asked about existing weights: `y` to skip (resume), `n` to retrain.

> `skel_recall` will automatically pre-compute skeleton maps before training on the first run — this may take several minutes.

### Menu — STARE

```
[2] STARE → [2] Training Model
  → [Semua Fungsi Loss (Train All)]
```

### Output (per loss, per dataset)

```
weights/drive/
└── drive_{loss_key}.weights.h5          ← best checkpoint (monitored: val_loss)

results/drive/history/
└── {loss_key}_history.json              ← epoch-by-epoch loss, accuracy + elapsed_sec
```

---

## Phase 8 — Main Evaluation (§ 4.6)

**Purpose:** Evaluate all trained models on the held-out test set using 10 metrics.

> DRIVE uses FOV masks (`use_mask_eval=True`) — metrics are computed only within the field of view. STARE has no masks (`use_mask_eval=False`) — full-image evaluation.

### Menu — DRIVE

```
[1] DRIVE → [3] Evaluasi Model
  → Seed tag: [leave blank, press Enter]
  → [Evaluasi Semua Model]
```

### Menu — STARE

```
[2] STARE → [3] Evaluasi Model
  → Seed tag: [leave blank, press Enter]
  → [Evaluasi Semua Model]
```

### Output (10 metrics: accuracy, F1, AUC, MCC, Jaccard, clDice, sensitivity, specificity, β0, β1)

```
results/drive/
└── {loss_key}_results.json              ← all evaluation metrics
```

---

## Phase 9 — Multi-Seed Training (§ 4.10)

**Purpose:** Retrain each model with 5 different seeds to obtain statistically valid mean ± SD.

Seeds control weight initialization, batch shuffling, and dropout — not the data split. All seeds use the same `aug_clahe/` directory from Phase 3.

**Total: 5 seeds × 7 losses × 2 datasets = 70 training runs.**

### Menu — Per loss function, DRIVE (repeat 7 times)

```
[1] DRIVE → [2] Training Model
  → [Multi-Seed Experiment (Mean ± SD)]
  → Select loss function
  → Enter seeds: 42 123 456 789 2026
  → Confirm: y
```

Repeat for each of the 7 loss functions and for STARE.

### Output

```
weights/drive/
└── drive_{loss_key}_seed{N}.weights.h5   ← per seed (N = 42, 123, 456, 789, 2026)

results/drive/history/
└── {loss_key}_seed{N}_history.json       ← per-seed training history
```

---

## Phase 10 — Multi-Seed Evaluation (§ 4.10)

**Purpose:** Evaluate each per-seed model to collect the metric distribution.

**Total: 5 seeds × 7 losses × 2 datasets = 70 evaluation runs.**

### Menu — Per seed, DRIVE (repeat 5 times with different seed tags)

```
[1] DRIVE → [3] Evaluasi Model
  → Seed tag: seed42          ← repeat with: seed42, seed123, seed456, seed789, seed2026
  → [Evaluasi Semua Model]
```

Repeat for each seed tag and for STARE.

### Output

```
results/drive/
├── {loss_key}_seed42_results.json
├── {loss_key}_seed123_results.json
├── {loss_key}_seed456_results.json
├── {loss_key}_seed789_results.json
└── {loss_key}_seed2026_results.json
```

---

## Phase 11 — Report Generation

**Purpose:** Generate all charts and analyses for every thesis section.

### ⚠ Mandatory order: Section 4.10 FIRST

Section 4.10 produces `multiseed_summary.json` at a canonical path that is automatically discovered by all other reporters when the "Generate All Reports" menu option is selected.

### Step 1 — Multi-Seed Report DRIVE (§ 4.10)

```
[1] DRIVE → [5] Reporting & Analysis
  → [4.10 Multi-Seed Analysis + Statistical Significance]
  → Seeds: [press Enter for default: 42 123 456 789 2026]
```

Output at `results/DRIVE/reports/section_4_10_multiseed/drive/`:
- `multiseed_summary.json` ← consumed by all other reporters
- Per-metric bar charts with mean ± SD
- Wilcoxon signed-rank heatmap (statistical significance test)
- Per-seed training curve overlays

### Step 2 — Multi-Seed Report STARE (§ 4.10)

```
[2] STARE → [5] Reporting & Analysis
  → [4.10 Multi-Seed Analysis + Statistical Significance]
  → Seeds: [press Enter]
```

### Step 3 — Generate all reports DRIVE

```
[1] DRIVE → [5] Reporting & Analysis
  → [Generate Semua Report]
```

Automatically loads `multiseed_summary.json` and passes it to every reporter. Generates sections 4.1, 4.4, 4.5, 4.5b, 4.6, 4.6b, 4.6c, 4.7, and 4.9 in one run.

### Step 4 — Generate all reports STARE

```
[2] STARE → [5] Reporting & Analysis
  → [Generate Semua Report]
```

### Step 5 — Regenerate tuning charts (§ 4.8)

```
Either dataset → [5] Reporting & Analysis
  → [4.8 Tuning Hyperparameter Loss — Regenerate Charts]
```

Reads from the existing SQLite DB — does not rerun tuning.

### Report Section Map

| Section | Content |
|---|---|
| § 4.1 | Environment report (hardware, library versions) |
| § 4.2 | Preprocessing ablation (RGB vs CLAHE) |
| § 4.3 | CLAHE parameter tuning charts |
| § 4.4 | Training history curves (per loss function) |
| § 4.5 | Segmentation visualization grid |
| § 4.5b | Segmentation visualization (English, journal format) |
| § 4.6 | Loss comparison: radar chart + bar chart + ranking |
| § 4.6b | Radar chart comparison DRIVE + STARE (English) |
| § 4.6c | Bar chart comparison DRIVE + STARE (journal format) |
| § 4.7 | Combined training history DRIVE + STARE (2×6 grid) |
| § 4.8 | Hyperparameter tuning charts per loss function |
| § 4.9 | Computational time analysis (epochs, benchmark, scatter) |
| § 4.10 | Multi-seed analysis + Wilcoxon significance test |

---

## Appendix — Output File Map

```
sa_unetv2_loss_comparison/
├── config.py                              ← RANDOM_SEED, CLAHE params, LOSS_PARAMS
├── weights/
│   ├── drive/
│   │   ├── drive_{loss_key}.weights.h5           ← Phase 7 (default seed)
│   │   └── drive_{loss_key}_seed{N}.weights.h5   ← Phase 9 (multi-seed)
│   ├── stare/
│   └── ablation/
│       ├── drive_rgb_bce_mcc.weights.h5           ← Phase 4
│       └── drive_clahe_bce_mcc.weights.h5
├── results/
│   ├── drive/
│   │   ├── {loss_key}_results.json                ← Phase 8 (main evaluation)
│   │   ├── {loss_key}_seed{N}_results.json        ← Phase 10 (multi-seed evaluation)
│   │   ├── history/
│   │   │   ├── {loss_key}_history.json            ← Phase 7
│   │   │   └── {loss_key}_seed{N}_history.json    ← Phase 9
│   │   ├── tuning/
│   │   │   ├── {loss_key}_tuning.json             ← Phase 5 (best params)
│   │   │   └── {loss_key}_tuning.db               ← Optuna SQLite (for resume)
│   │   └── reports/
│   │       ├── section_4_1_environment/
│   │       ├── section_4_2_ablation/
│   │       ├── section_4_3_clahe_tuning/
│   │       ├── section_4_4_history/
│   │       ├── section_4_5_visualization/
│   │       ├── section_4_6_comparison/
│   │       ├── section_4_7_combined_history/
│   │       ├── section_4_8_tuning_plots/
│   │       ├── section_4_9_computational_time/
│   │       └── section_4_10_multiseed/drive/
│   │           └── multiseed_summary.json         ← REQUIRED before other reports
│   └── stare/                                     ← identical structure
└── datasets/
    ├── DRIVE/
    │   ├── aug_clahe/                             ← Phase 3 (main augmentation)
    │   ├── aug_rgb/                               ← Phase 4 (auto-generated by ablation)
    │   ├── aug_clip1p5_tile8/                     ← Phase 1 (CLAHE tuning, isolated)
    │   └── aug_clip1p0_tile8/ …
    └── STARE/                                     ← identical aug structure
```

---

*This document was generated based on source code investigation of the SA-UNetV2 Loss Comparison repository.*
