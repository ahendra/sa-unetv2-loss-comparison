# Protokol Eksekusi Eksperimen — SA-UNetV2 Loss Comparison

Dokumen ini mendeskripsikan urutan eksekusi yang benar dan wajib untuk mereproduksi seluruh hasil eksperimen penelitian komparasi fungsi loss pada segmentasi pembuluh darah retina menggunakan arsitektur SA-UNetV2.

---

## Informasi Umum

| Properti | Nilai |
|---|---|
| Dataset | DRIVE (20 gambar), STARE (16 gambar) |
| Fungsi Loss | 7 varian (bce\_mcc, dice, focal, cldice, dice\_ssim, bce\_ssim, skel\_recall) |
| Seeds Multi-Seed | 42, 123, 456, 789, 2026 |
| Total Phase | 11 (urutan wajib) |
| Epochs Training | 150 (max, dengan early stopping) |
| Optimizer | Adam (lr=1e-3) |

---

## ⚠ Aturan Kritis — Baca Sebelum Memulai

1. **`RANDOM_SEED = 42`** di `config.py` mengontrol pembagian gambar train/val (augmentasi) DAN sampler Optuna TPE. **Jangan pernah ubah nilai ini** — mengubahnya membatalkan reproduksibilitas seluruh eksperimen.

2. Setelah **Phase 1** (CLAHE Tuning) dan **Phase 5** (Loss HP Tuning), file `config.py` **harus diperbarui secara manual** sebelum phase berikutnya dapat berjalan dengan benar.

3. Untuk eksperimen multi-seed, gunakan environment variable `EXPERIMENT_SEED` — jangan ubah `RANDOM_SEED`. Menu multi-seed di program menangani ini secara otomatis.

4. **Report Section 4.10 (Multi-Seed) harus di-generate PERTAMA** di antara semua report — ia menghasilkan `multiseed_summary.json` yang dikonsumsi otomatis oleh semua reporter lain.

5. Jika perbaikan split dataset telah diterapkan, hapus folder `aug_clahe/` dan `aug_rgb/` lama sebelum augmentasi ulang.

---

## Ringkasan Urutan Eksekusi

```
Phase 0  → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
              ↓
        (manual: update config.py CLAHE)
                               ↓
                        (manual: update config.py LOSS_PARAMS)
                                              ↓
                               Phase 6 → Phase 7 → Phase 8
                                              ↓
                                    Phase 9 → Phase 10 → Phase 11
```

| Phase | Nama | Tipe | Section Tesis |
|---|---|---|---|
| 0 | Prerequisites & Setup | Manual | — |
| 1 | CLAHE Parameter Tuning | Otomatis | § 4.3 |
| 2 | Update config.py (CLAHE) | **Manual** | — |
| 3 | Augmentasi Utama | Otomatis | — |
| 4 | Preprocessing Ablation Study | Otomatis | § 4.2 |
| 5 | Loss Hyperparameter Tuning | Otomatis | § 4.8 |
| 6 | Update config.py (LOSS\_PARAMS) | **Manual** | — |
| 7 | Training Utama | Otomatis | § 4.4 |
| 8 | Evaluasi Utama | Otomatis | § 4.6 |
| 9 | Multi-Seed Training | Otomatis | § 4.10 |
| 10 | Multi-Seed Evaluasi | Otomatis | § 4.10 |
| 11 | Generate Semua Report | Otomatis | § 4.1–4.10 |

---

## Phase 0 — Prerequisites & Setup

### Struktur dataset yang diperlukan

```
datasets/
├── DRIVE/
│   ├── train/images/   ← 20 file .tif (21_training.tif … 40_training.tif)
│   ├── train/labels/   ← 20 file *_manual1.gif
│   ├── test/images/    ← 20 file .tif
│   ├── test/labels/    ← 20 file *_manual1.gif
│   └── test/mask/      ← 20 file *_test_mask.gif (FOV masks)
└── STARE/
    ├── train/images/   ← 16 file .ppm
    ├── train/labels/   ← 16 file *.ah.ppm
    ├── test/images/    ← file .ppm sisa
    └── test/labels/    ← file *.ah.ppm sisa
```

### Hapus augmentasi lama (jika split fix diterapkan)

```powershell
Remove-Item -Recurse -Force datasets\DRIVE\aug_clahe
Remove-Item -Recurse -Force datasets\DRIVE\aug_rgb
Remove-Item -Recurse -Force datasets\STARE\aug_clahe
Remove-Item -Recurse -Force datasets\STARE\aug_rgb
```

> Lewati langkah ini jika menjalankan dari awal (belum ada folder aug sebelumnya). Reporter CLAHE tuning membuat directory isolasi sendiri (`aug_clip1p5_tile8/` dst.) dan tidak terpengaruh oleh penghapusan `aug_clahe/`.

### Jalankan program

```bash
cd sa_unetv2_loss_comparison
python main.py
```

---

## Phase 1 — CLAHE Parameter Tuning (§ 4.3)

**Tujuan:** Menentukan nilai optimal `clipLimit` dan `tileGridSize` untuk preprocessing CLAHE yang digunakan di seluruh eksperimen selanjutnya.

**Mengapa di-run pertama?** Phase ini menentukan parameter CLAHE yang akan digunakan di augmentasi utama (Phase 3). Setiap kombinasi mendapat directory augmentasi terisolasi sendiri (`aug_clip1p5_tile8/` dst.) sehingga tidak mengganggu `aug_clahe/` utama.

### Spesifikasi pencarian

| | Step A | Step B |
|---|---|---|
| Yang divariasikan | clipLimit ∈ {1.0, 1.5, 2.0, 3.0, 4.0} | tileGridSize ∈ {4, 16} |
| Yang ditetapkan | tileGridSize = 8 | clipLimit = best dari Step A |
| Jumlah run | 5 kombinasi × 2 dataset | 2 kombinasi × 2 dataset |

**Total: 7 kombinasi × 2 dataset = 14 training run, masing-masing 150 epoch.**

**Composite score** untuk memilih kombinasi terbaik:
```
Score = 0.35 × F1 + 0.25 × AUC + 0.20 × Sensitivity - 0.20 × β0_error
(semua dinormalisasi min-max per dataset)
```

### Menu

```
Dataset mana saja → [5] Reporting & Analysis
  → [4.3] CLAHE Parameter Tuning Study
  → Epochs per run: 150
```

Reporter berjalan otomatis untuk DRIVE dan STARE dalam satu invokasi. Jika diinterupsi, jalankan ulang — run yang sudah memiliki weights dan result JSON akan dilewati otomatis.

### Output

```
results/drive/reports/section_4_3_clahe_tuning/
├── clahe_tuning_results.json    ← best_clip, best_tile, dan semua skor
├── clahe_tuning_drive.png       ← bar chart DRIVE (7 kombinasi × 10 metrik)
└── clahe_tuning_stare.png       ← bar chart STARE

weights/drive/
└── drive_clahe_tune_clip*_tile*.weights.h5   ← per kombinasi

results/drive/
└── clahe_tune_*_results.json                 ← per kombinasi
```

### Tindakan setelah selesai

Buka `clahe_tuning_results.json`, baca `best_clip` dan `best_tile`. Lanjutkan ke **Phase 2**.

---

## Phase 2 — Update config.py (CLAHE Parameters)

**Tipe: Manual**

Edit `config.py` pada `DriveConfig` dan `StareConfig`:

```python
# Ganti dengan nilai dari clahe_tuning_results.json → best_clip dan best_tile
clahe_clip_limit: float = 1.5   # ← best_clip dari Phase 1
clahe_tile_grid: int    = 16    # ← best_tile dari Phase 1
```

> **Catatan:** Default saat ini (`clip=1.5, tile=16`) adalah hasil dari tuning run yang sudah disertakan. Perbarui hanya jika re-run Phase 1 menghasilkan nilai berbeda.

---

## Phase 3 — Augmentasi Utama

**Tujuan:** Membuat directory `aug_clahe/` canonical yang digunakan oleh semua training, tuning loss, dan multi-seed.

### Spesifikasi augmentasi

| | DRIVE | STARE |
|---|---|---|
| Gambar asli train | 18 gambar | 14 gambar |
| Gambar asli val | 2 gambar | 2 gambar |
| Augmentasi per gambar | 12 varian + 1 copy original = 13 file | sama |
| Total file train | 18 × 13 = 234 | 14 × 13 = 182 |
| Total file val | 2 × 13 = 26 | 2 × 13 = 26 |

Split dilakukan di level gambar asli menggunakan `RANDOM_SEED=42`, sebelum augmentasi. Preprocessing CLAHE di-bake ke setiap file saat augmentasi — bukan diterapkan saat load.

### Menu — DRIVE

```
[1] DRIVE → [1] Augmentasi Dataset
  → [1] RGB + CLAHE → aug_clahe/
  → Konfirmasi: y
```

### Menu — STARE

```
[2] STARE → [1] Augmentasi Dataset
  → [1] RGB + CLAHE → aug_clahe/
  → Konfirmasi: y
```

---

## Phase 4 — Preprocessing Ablation Study (§ 4.2)

**Tujuan:** Membandingkan RGB Original vs RGB + CLAHE (dengan parameter optimal) untuk membuktikan efektivitas CLAHE.

**Mengapa setelah Phase 3?** Ablation menggunakan `aug_clahe/` yang dibuat di Phase 3, memastikan kondisi CLAHE menggunakan parameter optimal dari Phase 1. Ablation juga akan auto-generate `aug_rgb/` jika belum ada.

> Weights ablation disimpan di `weights/ablation/` — tidak menimpa weights eksperimen utama.

### Menu

```
[1] DRIVE → [5] Reporting & Analysis
  → [4.2] Preprocessing Ablation Study
  → Epochs per kondisi: 150
```

### Output

```
results/DRIVE/reports/section_4_2_ablation/
├── preprocessing_ablation.json              ← metrik per kondisi
├── preprocessing_ablation_table.txt         ← tabel teks (bisa dicopy ke LaTeX)
├── preprocessing_ablation.png               ← bar chart 10 metrik × 2 kondisi
└── preprocessing_sample_comparison.png      ← visual RGB vs CLAHE (3 gambar test)
```

---

## Phase 5 — Loss Function Hyperparameter Tuning (§ 4.8)

**Tujuan:** Menemukan hyperparameter optimal untuk setiap fungsi loss menggunakan Optuna TPE.

**Bergantung pada Phase 3.** Tuning memuat data dari `aug_clahe/`. Jika belum ada, menu akan menawarkan untuk generate otomatis — pastikan `config.py` sudah diperbarui (Phase 2) sebelum itu terjadi.

### Spesifikasi

| Parameter | Nilai |
|---|---|
| Metode | Optuna TPE + MedianPruner |
| Objective | Maksimalkan F1 Score (val set) |
| Trials per loss | 30 |
| Epochs per trial | 50 |
| Resume | SQLite DB — aman diinterupsi dan dilanjutkan |

### Search space per fungsi loss

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

### Menu — Tune semua loss (rekomendasi)

```
[1] DRIVE → [4] Hyperparameter Tuning
  → [Tuning Semua Fungsi Loss]
  → Jumlah trials: 30
  → Epochs per trial: 50
```

Ulangi untuk STARE. Jika diinterupsi, jalankan ulang dan pilih **resume** untuk melanjutkan dari trial terakhir.

### Output (per loss, per dataset)

```
results/drive/tuning/
├── {loss_key}_tuning.json       ← best_params + semua trial history
├── {loss_key}_tuning.db         ← SQLite DB (untuk resume)
├── {loss_key}_tuning_history.png    ← kurva optimasi
├── {loss_key}_tuning_importance.png ← fANOVA parameter importance
└── {loss_key}_tuning_scatter.png    ← scatter param vs F1
```

### Tindakan setelah selesai

Buka tiap `{loss_key}_tuning.json`, baca bagian `best_params`. Terminal juga mencetak blok kode siap-copy. Lanjutkan ke **Phase 6**.

---

## Phase 6 — Update config.py (LOSS\_PARAMS)

**Tipe: Manual**

Edit `config.py`, salin `best_params` dari setiap JSON hasil tuning ke `LOSS_PARAMS`:

```python
LOSS_PARAMS = {
    "bce_mcc": {
        "lambda_bce": 0.5962,   # ← dari bce_mcc_tuning.json → best_params
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

> **Penting:** `LOSS_PARAMS` berlaku global untuk DRIVE dan STARE. Jika tuning DRIVE dan STARE menghasilkan parameter berbeda, tentukan set mana yang digunakan (atau jalankan eksperimen terpisah). Saat ini nilai tuning DRIVE digunakan untuk keduanya.

---

## Phase 7 — Main Training (§ 4.4)

**Tujuan:** Melatih model SA-UNetV2 untuk semua 7 fungsi loss pada kedua dataset, menggunakan seed default.

| Parameter | DRIVE | STARE |
|---|---|---|
| Batch size | 8 | 2 |
| Max epochs | 150 | 150 |
| EarlyStopping patience | 20 epoch | 30 epoch |
| Monitor checkpoint | val\_loss | val\_loss |
| ReduceLR patience | 10 epoch | 20 epoch |

### Menu — DRIVE

```
[1] DRIVE → [2] Training Model
  → [Semua Fungsi Loss (Train All)]
```

Saat ditanya weight sudah ada: `y` untuk lewati (resume), `n` untuk latih ulang.

> `skel_recall` akan otomatis komputasi skeleton maps sebelum training pada run pertama — ini dapat memakan beberapa menit.

### Menu — STARE

```
[2] STARE → [2] Training Model
  → [Semua Fungsi Loss (Train All)]
```

### Output (per loss, per dataset)

```
weights/drive/
└── drive_{loss_key}.weights.h5          ← checkpoint terbaik (val_loss)

results/drive/history/
└── {loss_key}_history.json              ← epoch-by-epoch loss, accuracy + elapsed_sec
```

---

## Phase 8 — Main Evaluation (§ 4.6)

**Tujuan:** Evaluasi semua model terlatih pada test set menggunakan 10 metrik.

> DRIVE menggunakan FOV mask (`use_mask_eval=True`) — metrik dihitung hanya di dalam field of view. STARE tidak menggunakan mask (`use_mask_eval=False`).

### Menu — DRIVE

```
[1] DRIVE → [3] Evaluasi Model
  → Seed tag: [kosongkan, tekan Enter]
  → [Evaluasi Semua Model]
```

### Menu — STARE

```
[2] STARE → [3] Evaluasi Model
  → Seed tag: [kosongkan, tekan Enter]
  → [Evaluasi Semua Model]
```

### Output (10 metrik: accuracy, F1, AUC, MCC, Jaccard, clDice, sensitivity, specificity, β0, β1)

```
results/drive/
└── {loss_key}_results.json              ← semua metrik evaluasi
```

---

## Phase 9 — Multi-Seed Training (§ 4.10)

**Tujuan:** Melatih ulang setiap model dengan 5 seed berbeda untuk mendapatkan mean ± SD yang valid secara statistik.

Seed mengontrol inisialisasi bobot, shuffle batch, dan dropout — bukan split data. Semua seed menggunakan `aug_clahe/` yang sama dari Phase 3.

**Total: 5 seeds × 7 loss × 2 dataset = 70 training run.**

### Menu — Per fungsi loss, DRIVE (ulangi 7 kali)

```
[1] DRIVE → [2] Training Model
  → [Multi-Seed Experiment (Mean ± SD)]
  → Pilih fungsi loss
  → Masukkan seeds: 42 123 456 789 2026
  → Konfirmasi: y
```

Ulangi untuk setiap 7 fungsi loss dan untuk STARE.

### Output

```
weights/drive/
└── drive_{loss_key}_seed{N}.weights.h5   ← per seed (N = 42, 123, 456, 789, 2026)

results/drive/history/
└── {loss_key}_seed{N}_history.json       ← history per seed
```

---

## Phase 10 — Multi-Seed Evaluation (§ 4.10)

**Tujuan:** Evaluasi setiap model per-seed untuk mengumpulkan distribusi metrik.

**Total: 5 seeds × 7 loss × 2 dataset = 70 evaluasi run.**

### Menu — Per seed, DRIVE (ulangi 5 kali dengan seed tag berbeda)

```
[1] DRIVE → [3] Evaluasi Model
  → Seed tag: seed42          ← ganti: seed42, seed123, seed456, seed789, seed2026
  → [Evaluasi Semua Model]
```

Ulangi untuk setiap seed tag dan untuk STARE.

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

**Tujuan:** Menghasilkan semua chart dan analisis untuk seluruh section tesis.

### ⚠ Urutan wajib: Section 4.10 DULU

Section 4.10 menghasilkan `multiseed_summary.json` di path canonical yang ditemukan otomatis oleh semua reporter lain saat menu "Generate Semua Report" dijalankan.

### Langkah 1 — Multi-Seed Report DRIVE (§ 4.10)

```
[1] DRIVE → [5] Reporting & Analysis
  → [4.10 Multi-Seed Analysis + Statistical Significance]
  → Seeds: [tekan Enter untuk default: 42 123 456 789 2026]
```

Output di `results/DRIVE/reports/section_4_10_multiseed/drive/`:
- `multiseed_summary.json` ← dikonsumsi oleh semua reporter lain
- Bar chart mean ± SD per metrik
- Wilcoxon signed-rank heatmap (uji signifikansi statistik)
- Kurva training overlay per seed

### Langkah 2 — Multi-Seed Report STARE (§ 4.10)

```
[2] STARE → [5] Reporting & Analysis
  → [4.10 Multi-Seed Analysis + Statistical Significance]
  → Seeds: [tekan Enter]
```

### Langkah 3 — Generate semua report DRIVE

```
[1] DRIVE → [5] Reporting & Analysis
  → [Generate Semua Report]
```

Otomatis memuat `multiseed_summary.json` dan meneruskannya ke semua reporter.

### Langkah 4 — Generate semua report STARE

```
[2] STARE → [5] Reporting & Analysis
  → [Generate Semua Report]
```

### Langkah 5 — Regenerate tuning charts (§ 4.8)

```
Dataset mana saja → [5] Reporting & Analysis
  → [4.8 Tuning Hyperparameter Loss — Regenerate Charts]
```

Membaca ulang dari SQLite DB yang ada — tidak melakukan tuning ulang.

### Peta section report

| Section | Isi |
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

## Appendix — Peta File Output

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
│   │   ├── {loss_key}_results.json                ← Phase 8 (main eval)
│   │   ├── {loss_key}_seed{N}_results.json        ← Phase 10 (multi-seed eval)
│   │   ├── history/
│   │   │   ├── {loss_key}_history.json            ← Phase 7
│   │   │   └── {loss_key}_seed{N}_history.json    ← Phase 9
│   │   ├── tuning/
│   │   │   ├── {loss_key}_tuning.json             ← Phase 5 (best params)
│   │   │   └── {loss_key}_tuning.db               ← Optuna SQLite (resume)
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
│   │           └── multiseed_summary.json         ← WAJIB ada sebelum report lain
│   └── stare/                                     ← struktur identik
└── datasets/
    ├── DRIVE/
    │   ├── aug_clahe/                             ← Phase 3 (augmentasi utama)
    │   ├── aug_rgb/                               ← Phase 4 (auto-generate ablation)
    │   ├── aug_clip1p5_tile8/                     ← Phase 1 (CLAHE tuning, terisolasi)
    │   └── aug_clip1p0_tile8/ …
    └── STARE/                                     ← struktur aug identik
```

---

*Dokumen ini dibuat berdasarkan investigasi kode sumber repository SA-UNetV2 Loss Comparison.*
