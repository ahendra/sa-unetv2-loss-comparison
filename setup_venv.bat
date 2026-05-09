@echo off
echo ============================================
echo  SA-UNetv2 Loss Comparison - Setup Venv
echo ============================================
echo.

REM -- Cek Python yang aktif kompatibel (3.9-3.12) ----------------------------
echo Memeriksa versi Python...
python --version 2>&1

python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and 9<=v.minor<=12 else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python yang aktif tidak kompatibel dengan TensorFlow 2.19.0.
    echo         TF 2.19.0 hanya mendukung Python 3.9 hingga 3.12.
    echo         Python 3.13 atau lebih baru TIDAK didukung.
    echo.
    echo Solusi: aktifkan conda environment Python 3.12 terlebih dahulu:
    echo   conda activate py312
    echo   setup_venv.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Python kompatibel.
echo.

REM -- 1. Buat virtual environment --------------------------------------------
echo [1/4] Membuat virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Gagal membuat venv.
    pause
    exit /b 1
)
echo [OK] venv berhasil dibuat.
echo.

REM -- 2. Aktifkan venv -------------------------------------------------------
echo [2/4] Mengaktifkan virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Gagal mengaktifkan venv.
    pause
    exit /b 1
)
echo [OK] venv aktif.
echo.

REM -- 3. Upgrade pip ---------------------------------------------------------
echo [3/4] Meng-upgrade pip...
python -m pip install --upgrade pip
echo.

REM -- 4. Install dependencies ------------------------------------------------
echo [4/4] Menginstall dependencies (mungkin memakan beberapa menit)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Gagal menginstall dependencies.
    echo         Coba jalankan manual: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup berhasil!
echo.
echo  Untuk menjalankan program selanjutnya:
echo    1. venv\Scripts\activate.bat
echo    2. python main.py
echo ============================================
pause
