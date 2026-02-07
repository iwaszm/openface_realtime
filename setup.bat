@echo off
TITLE OpenFace Setup (Windows)
echo ===================================================
echo      OpenFace Realtime - One-Click Setup
echo ===================================================

:: 1. Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b
)

:: 2. Create Virtual Environment
IF NOT EXIST "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
) ELSE (
    echo [INFO] Virtual environment exists.
)

:: 3. Install Dependencies
echo [INFO] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
:: Force binary pillow to avoid compilation issues
pip install "Pillow>=10.3.0" --only-binary=:all:
pip install numpy opencv-python torch torchvision pandas scipy seaborn tensorboardX timm tqdm scikit-image
pip install openface-test --no-deps

:: 4. Apply Patches (Copying pre-patched files)
echo [INFO] Applying fixes...
:: Determine site-packages path (a bit tricky in batch, assuming standard layout)
set SITE_PACKAGES=venv\Lib\site-packages

IF EXIST "openface_lib_patches" (
    xcopy "openface_lib_patches\openface" "%SITE_PACKAGES%\openface" /s /e /y /i
    echo [OK] Patches applied.
) ELSE (
    echo [WARN] Patch directory not found! Code might crash.
)

:: 5. Download Weights
if not exist "weights" mkdir weights
cd weights

echo [INFO] Checking weights...

if not exist "Alignment_RetinaFace.pth" (
    echo Downloading Alignment_RetinaFace.pth...
    curl -L -o Alignment_RetinaFace.pth "https://huggingface.co/nutPace/openface_weights/resolve/main/Alignment_RetinaFace.pth"
)

if not exist "Landmark_98.pkl" (
    echo Downloading Landmark_98.pkl...
    curl -L -o Landmark_98.pkl "https://huggingface.co/nutPace/openface_weights/resolve/main/Landmark_98.pkl"
)

if not exist "MTL_backbone.pth" (
    echo Downloading MTL_backbone.pth...
    curl -L -o MTL_backbone.pth "https://huggingface.co/nutPace/openface_weights/resolve/main/MTL_backbone.pth"
)

cd ..

echo.
echo ===================================================
echo [SUCCESS] Setup complete!
echo To run the app, double-click 'run.bat' or type:
echo venv\Scripts\python run.py
echo ===================================================
pause
