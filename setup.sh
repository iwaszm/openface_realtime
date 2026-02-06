#!/bin/bash

# OpenFace Realtime Setup Script
# Installs dependencies, downloads weights, and patches the OpenFace library for Python 3.12+ compatibility.

set -e # Exit on error

echo "🚀 Starting OpenFace Setup..."

# 1. Environment Setup
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

echo "📦 Installing dependencies..."
# Force binary Pillow to avoid compilation issues
pip install "Pillow>=10.3.0" --only-binary=:all:
# Install core libraries
pip install numpy opencv-python torch torchvision pandas scipy seaborn tensorboardX timm tqdm scikit-image
# Install OpenFace (without deps to avoid conflict)
pip install openface-test --no-deps

# 2. Apply Patches
echo "🔧 Patching libraries..."

VENV_LIB="venv/lib/python3.12/site-packages/openface"

# Patch 1: SciPy 'simps' removal (compat for SciPy 1.14+)
METRIC_FILE="$VENV_LIB/STAR/lib/metric/fr_and_auc.py"
if [ -f "$METRIC_FILE" ]; then
    sed -i 's/from scipy.integrate import simps/from scipy.integrate import simpson as simps/g' "$METRIC_FILE"
    echo "  ✅ Patched SciPy compatibility"
fi

# Patch 2: Fix hardcoded path in alignment.py
ALIGN_CONF="$VENV_LIB/STAR/conf/alignment.py"
if [ -f "$ALIGN_CONF" ]; then
    # Simple replace of the known hardcoded string
    sed -i "s|self.ckpt_dir = '/work/jiewenh/openFace/OpenFace-3.0/STAR'|self.ckpt_dir = './logs'|g" "$ALIGN_CONF"
    echo "  ✅ Patched log path"
fi

# Patch 3: Fix FaceDetector to support in-memory images (Video frames)
FACE_DET="$VENV_LIB/face_detection.py"
if [ -f "$FACE_DET" ]; then
    # Only patch if not already patched
    if ! grep -q "isinstance(image_path, np.ndarray)" "$FACE_DET"; then
        # Use python to perform the multiline string replacement safely
        python3 -c "
import sys
content = open('$FACE_DET').read()
old_code = '        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)'
new_code = '''        if isinstance(image_path, np.ndarray):
            img_raw = image_path
        else:
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)'''
content = content.replace(old_code, new_code)
open('$FACE_DET', 'w').write(content)
"
        echo "  ✅ Patched FaceDetector for video stream"
    else
        echo "  ℹ️ FaceDetector already patched"
    fi
fi

# 3. Download Weights
echo "⬇️ Checking Model Weights..."
mkdir -p weights

download_if_missing() {
    FILE="weights/$1"
    URL="$2"
    if [ ! -f "$FILE" ]; then
        echo "  ⏳ Downloading $1..."
        curl -L -o "$FILE" "$URL"
    else
        echo "  ✅ Found $1"
    fi
}

download_if_missing "Alignment_RetinaFace.pth" "https://huggingface.co/nutPace/openface_weights/resolve/main/Alignment_RetinaFace.pth"
download_if_missing "Landmark_98.pkl" "https://huggingface.co/nutPace/openface_weights/resolve/main/Landmark_98.pkl"
download_if_missing "MTL_backbone.pth" "https://huggingface.co/nutPace/openface_weights/resolve/main/MTL_backbone.pth"

echo "✅ Setup Complete!"
echo "👉 Run with: ./venv/bin/python3 run.py"
