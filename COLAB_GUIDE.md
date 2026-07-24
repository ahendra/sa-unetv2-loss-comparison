# Panduan Menjalankan Project di Google Colab

**SA-UNetV2 Loss Function Comparison**
Segmentasi Pembuluh Darah Retina

---

## Daftar Isi

1. [Persiapan Google Drive](#1-persiapan-google-drive)
2. [Struktur Dataset yang Diperlukan](#2-struktur-dataset-yang-diperlukan)
3. [Menjalankan Notebook](#3-menjalankan-notebook)
4. [Konfigurasi config.py](#4-konfigurasi-configpy)
5. [Alur Kerja Lengkap (Urutan Menu)](#5-alur-kerja-lengkap-urutan-menu)
6. [Penjelasan Setiap Menu](#6-penjelasan-setiap-menu)
7. [Lokasi Output File](#7-lokasi-output-file)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Persiapan Google Drive

Project ini secara otomatis me-mount Google Drive dan menyimpan semua data, bobot model, dan hasil di dalamnya. **Tidak perlu mengubah path secara manual** — `config.py` mendeteksi Colab dan mengatur path sendiri.

### Struktur folder yang harus dibuat di Google Drive:

```
MyDrive/
└── Kuliah/
    └── Tesis/
        └── Program/
            └── sa_unetv2_loss_comparison/
                ├── datasets/          ← upload dataset di sini
                │   ├── DRIVE/
                │   └── STARE/
                ├── weights/           ← dibuat otomatis saat training
                └── results/           ← dibuat otomatis saat evaluasi/report
```

> **Penting:** Path `MyDrive/Kuliah/Tesis/Program/sa_unetv2_loss_comparison/` sudah di-hardcode di `config.py`. Jika struktur folder di Google Drive Anda berbeda, lihat [bagian konfigurasi](#42-mengubah-path-google-drive-jika-diperlukan).

---

## 2. Struktur Dataset yang Diperlukan

Upload dataset ke Google Drive **sebelum** menjalankan program.

### DRIVE Dataset

```
datasets/DRIVE/
├── train/
│   ├── images/          ← file gambar training (*.tif)
│   └── labels/          ← file label training (*_manual1.gif atau *_manual1.png)
└── test/
    ├── images/          ← file gambar test (*.tif)
    ├── labels/          ← file label test (*_manual1.gif)
    └── mask/            ← file mask FOV test (*_test_mask.gif)
```

### STARE Dataset

```
datasets/STARE/
├── train/
│   ├── images/          ← file gambar training (*.ppm)
│   └── labels/          ← file label training (*.ah.ppm)
└── test/
    ├── images/          ← file gambar test (*.ppm)
    └── labels/          ← file label test (*.ah.ppm)
```

> Dataset DRIVE dan STARE dapat diunduh dari sumber resminya masing-masing.
> Folder `aug_clahe/` (data augmentasi) **tidak perlu diupload** — akan dibuat otomatis melalui menu Augmentasi.

---

## 3. Menjalankan Notebook

Buka file `sa-unetv2-loss-comparison.ipynb` di Google Colab, lalu jalankan sel secara berurutan:

### Sel 1 — Clone repository

```python
!git clone https://github.com/ahendra/sa-unetv2-loss-comparison.git
```

Mengunduh kode project dari GitHub ke `/content/sa-unetv2-loss-comparison/`.

### Sel 2 — Masuk ke direktori project

```python
%cd sa-unetv2-loss-comparison
```

### Sel 3 — Install dependensi

```python
!pip install -r requirements.txt
```

Menginstall semua library yang diperlukan:

| Library       | Versi   | Fungsi                    |
| ------------- | ------- | ------------------------- |
| tensorflow    | 2.19.0  | Framework deep learning   |
| keras-cv      | ≥0.9.0  | DropBlock2D layer         |
| opencv-python | ≥4.9.0  | Preprocessing gambar      |
| optuna        | ≥3.6.0  | Hyperparameter tuning     |
| scikit-image  | ≥0.22.0 | Evaluasi topologi (Betti) |
| openpyxl      | ≥3.1.0  | Export hasil ke Excel     |
| matplotlib    | ≥3.8.0  | Visualisasi dan grafik    |

> Instalasi membutuhkan waktu 2–5 menit. Sel ini hanya perlu dijalankan sekali per sesi Colab.

### Sel 4 — Update ke versi terbaru (opsional)

```python
!git pull
```

Mengambil pembaruan terbaru dari repository jika ada.

### Sel 5 — Jalankan program

```python
%run main.py
```

Program akan meminta izin mount Google Drive, lalu menampilkan menu interaktif.

---

## 4. Konfigurasi config.py

File `config.py` berisi semua konfigurasi eksperimen. Sebagian besar tidak perlu diubah karena sudah mengikuti spesifikasi paper SA-UNetV2 asli.

### 4.1 Parameter yang TIDAK boleh diubah

Parameter berikut sudah ditetapkan sesuai paper SA-UNetV2 dan harus konsisten di semua eksperimen:

```python
# config.py — DriveConfig
batch_size      = 8       # DRIVE
epochs          = 150     # maksimum epoch
learning_rate   = 1e-3
start_neurons   = 16      # arsitektur SA-UNetV2
block_size      = 7       # DropBlock2D
drop_rate       = 0.15

# config.py — StareConfig
batch_size      = 2       # STARE (gambar lebih besar)
epochs          = 150
```

### 4.2 Mengubah Path Google Drive (jika diperlukan)

Jika struktur folder Google Drive Anda berbeda dari default, ubah baris berikut di `config.py`:

```python
# Baris yang perlu diubah (sekitar baris 33):
root = Path('/content/drive/MyDrive/Kuliah/Tesis/Program/sa_unetv2_loss_comparison')

# Contoh jika folder Anda di lokasi lain:
root = Path('/content/drive/MyDrive/TesisSaya/sa_unetv2_loss_comparison')
```

### 4.3 Mode Preprocessing

```python
# Default (yang digunakan untuk eksperimen utama):
preprocessing_mode = "clahe"   # RGB + CLAHE enhancement

# Pilihan lain (untuk ablation study):
preprocessing_mode = "rgb"     # RGB original (tanpa preprocessing tambahan)
```

> Jangan ubah `preprocessing_mode` di `config.py` secara manual. Mode dipilih secara interaktif melalui menu Augmentasi di program.

### 4.4 Random Seed

```python
RANDOM_SEED = none   # sesuai paper SA-UNetV2
```

### 4.5 Hyperparameter Fungsi Loss (`LOSS_PARAMS`)

```python
# Di bagian bawah config.py — parameter hasil tuning Optuna diisi disini, di sesuaikan dengan dataset yang akan diuji (DRIVE/STARE)
LOSS_PARAMS = {
    "bce_mcc":   {"lambda_bce": ..., "lambda_mcc": ...},
    "dice":      {"smooth": ...},
    "focal":     {"alpha": ..., "gamma": ..., "smooth": ...},
    "cldice":    {"alpha": ..., "iters": ..., "smooth": ...},
    "dice_ssim": {"lambda_dice": ..., "smooth": ..., "lambda_ssim": ...},
    "bce_ssim":  {"lambda_bce": ..., "lambda_ssim": ...},
}
```

---

## 5. Alur Kerja Lengkap (Urutan Menu)

Jalankan menu-menu berikut **secara berurutan** untuk setiap dataset (DRIVE dan STARE):

```
Main Menu
├── [1] Select Dataset (DRIVE)
│   ├── LANGKAH 1: [1] Augmentasi Dataset  ─── Wajib dilakukan pertama kali
│   ├── LANGKAH 2: [4] Hyperparameter Tuning ── Opsional tapi direkomendasikan
│   ├── LANGKAH 3: [2] Training Model  ──────── Wajib sebelum evaluasi
│   ├── LANGKAH 4: [3] Evaluasi Model  ──────── Wajib sebelum reporting
│   └── LANGKAH 5: [5] Reporting & Analysis ─── Generate grafik dan tabel
│
└── [2] Select Dataset (STARE)
    └── (ulangi langkah 1–5 untuk STARE)
```

### Estimasi Waktu per Langkah (GPU A100)

| Langkah                      | DRIVE                 | STARE                  |
| ---------------------------- | --------------------- | ---------------------- |
| Augmentasi                   | ~2 menit              | ~2 menit               |
| Tuning (30 trial × 30 epoch) | ~60–90 menit per loss | ~90–120 menit per loss |
| Training (150 epoch maks)    | ~8–15 menit per loss  | ~12–20 menit per loss  |
| Evaluasi                     | ~1 menit per loss     | ~1 menit per loss      |
| Reporting semua              | ~5–30 menit           | ~5–30 menit            |

---

## 6. Penjelasan Setiap Menu

### Menu Dataset → [1] Augmentasi Dataset

Membuat data augmentasi dari gambar training asli. Harus dilakukan **sebelum training**.

- Pilih mode: **RGB + CLAHE** (untuk eksperimen utama) atau RGB Original (untuk ablation)
- Output: folder `aug_clahe/` dengan 234 gambar train + 26 validasi (DRIVE), atau 187+21 (STARE)
- Spesifikasi: rotation ×3, color jitter ×3, gaussian noise ×3, h-flip, v-flip, hv-flip = 12 augmentasi per gambar asli

> Jika folder augmentasi sudah ada, program akan menanya apakah ingin dihapus dan dibuat ulang.

### Menu Dataset → [4] Hyperparameter Tuning

Mencari hyperparameter optimal untuk setiap fungsi loss menggunakan Optuna TPE.

- Pilih fungsi loss atau "Tuning Semua" (direkomendasikan)
- Masukkan jumlah trials (default 30, rekomendasi 50 untuk hasil publikasi)
- Masukkan epochs per trial (default 30)
- Hasil disimpan ke: `results/<dataset>/tuning/<loss>_tuning.json` dan `<loss>_tuning.db`
- **Resume otomatis**: jika sesi Colab terputus, tuning akan dilanjutkan dari trial terakhir

> Proses ini memakan waktu paling lama. Pastikan Colab terhubung ke runtime GPU dan aktifkan "High-RAM" jika tersedia.

### Menu Dataset → [2] Training Model

Melatih model SA-UNetV2 dengan setiap fungsi loss.

- Pilih fungsi loss tertentu atau "Train All" (melatih semua 6 secara berurutan)
- Jika bobot sudah ada, program akan menanya apakah ingin melatih ulang
- Bobot tersimpan di: `weights/<dataset>/<loss_key>_best.weights.h5`
- History training tersimpan di: `results/<dataset>/<loss_key>_history.json`

### Menu Dataset → [3] Evaluasi Model

Mengevaluasi model terlatih pada data test.

- Memerlukan bobot yang sudah ada dari proses training
- Menghitung 10 metrik: Accuracy, Sensitivity, Specificity, AUC, MCC, F1, Jaccard, clDice, Betti-0 error, Betti-1 error
- Hasil tersimpan di: `results/<dataset>/<loss_key>_results.json`

### Menu Dataset → [5] Reporting & Analysis

Generate semua grafik dan laporan untuk BAB IV tesis.

| Sub-menu                       | Output                                                       | Catatan              |
| ------------------------------ | ------------------------------------------------------------ | -------------------- |
| 4.1 Lingkungan Eksperimen      | `environment.json`, `environment_table.png`                  |                      |
| 4.2 Preprocessing Ablation     | `preprocessing_ablation.png`                                 | Perlu training ulang |
| 4.3 CLAHE Parameter Tuning     | `clahe_tuning_*.png`                                         | Perlu training ulang |
| 4.4 Training History Curves    | `<loss>_history.png`                                         | Butuh history JSON   |
| 4.5 Segmentation Visualization | `segmentation_grid_*.png`                                    | Butuh bobot model    |
| 4.6 Loss Function Comparison   | `radar_*.png`, `comparison_*.png`                            | Butuh results JSON   |
| 4.6c Bar Chart                 | `bar_chart_drive.png`, `bar_chart_stare.png`                 |                      |
| 4.7 Combined History           | `combined_history.png`                                       |                      |
| 4.8 Tuning Charts              | `<loss>_tuning_*.png`                                        | Butuh tuning DB      |
| 4.9 Computational Time         | `benchmark_*.png`, `scatter_*.png`, `benchmark_results.json` | Butuh GPU            |

---

## 7. Lokasi Output File

Semua output tersimpan di Google Drive secara otomatis:

```
sa_unetv2_loss_comparison/
├── weights/
│   ├── drive/
│   │   ├── bce_mcc_best.weights.h5
│   │   ├── dice_best.weights.h5
│   │   └── ...
│   └── stare/
│       └── ...
└── results/
    ├── drive/
    │   ├── bce_mcc_results.json       ← metrik evaluasi
    │   ├── bce_mcc_history.json       ← kurva training per epoch
    │   ├── tuning/
    │   │   ├── bce_mcc_tuning.json    ← parameter terbaik Optuna
    │   │   └── bce_mcc_tuning.db      ← database Optuna (untuk resume)
    │   └── reports/
    │       ├── section_4_1_environment/
    │       ├── section_4_4_history/
    │       ├── section_4_6_comparison/
    │       └── section_4_9_computational_time/
    │           ├── benchmark_drive.png
    │           ├── benchmark_stare.png
    │           ├── scatter_drive.png
    │           ├── scatter_stare.png
    │           ├── epochs_convergence.png
    │           ├── benchmark_results.json
    │           └── benchmark_results.xlsx
    └── stare/
        └── ...
```

---

## 8. Troubleshooting

### Program tidak menemukan dataset

```
[ERROR] Direktori tidak ditemukan: /content/drive/MyDrive/.../datasets/DRIVE/train/images
```

**Solusi:** Pastikan dataset sudah diupload ke Google Drive dengan struktur folder yang benar (lihat [Bagian 2](#2-struktur-dataset-yang-diperlukan)).

---

### Sesi Colab terputus saat training

Semua progress tersimpan di Google Drive. Cukup jalankan ulang `%run main.py` dan lanjutkan dari titik terakhir:

- Training: pilih loss yang belum selesai. Jika bobot ada, program akan tanya apakah ingin lanjutkan.
- Tuning: pilih "Lanjutkan (resume)" — sesi Optuna tersimpan di file `.db`.

---

### Error saat install requirements

```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.19.0
```

**Solusi:** Colab sudah include TensorFlow. Jika versi konflik, jalankan:

```python
!pip install -r requirements.txt --upgrade --quiet
```

---

### GPU tidak terdeteksi

```
[WARN] GPU tidak terdeteksi. Benchmark akan dijalankan di CPU (sangat lambat).
```

**Solusi:** Pastikan runtime Colab menggunakan GPU:

- Menu: `Runtime` → `Change runtime type` → Hardware accelerator: **T4 GPU** atau **A100**

---

### openpyxl tidak terinstall (Colab lama)

```
[WARN] openpyxl tidak terinstall, skip export Excel.
```

**Solusi:**

```python
!pip install openpyxl
```

Kemudian restart dan jalankan ulang `%run main.py`.

---

### Tuning sangat lambat / Colab timeout

**Rekomendasi:**

- Gunakan A100 GPU (Colab Pro/Pro+) untuk tuning
- Mulai dengan 30 trials, 30 epochs per trial
- Tuning dapat dilanjutkan (`resume`) jika sesi terputus — tidak perlu mulai ulang
- Jalankan tuning per fungsi loss, bukan semua sekaligus, untuk menghindari timeout

---

_Dibuat untuk Google Colab dengan runtime GPU. Terakhir diperbarui: Juli 2026._
