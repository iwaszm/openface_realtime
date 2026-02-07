#!/bin/bash

# OpenFace Realtime Setup Script
# Installs dependencies, downloads weights, and applies pre-baked fixes for Python 3.12+ compatibility.

set -e

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

# 2. Apply Patches (The Golden Fixes)
echo "🔧 Applying fixes..."

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TARGET_LIB="$SITE_PACKAGES/openface"

if [ -d "openface_lib_patches/openface" ]; then
    # Copy our pre-fixed files directly over the installed library
    cp -r openface_lib_patches/openface/* "$TARGET_LIB/"
    echo "  ✅ Applied library patches (FaceDetector, RetinaFace, Alignment, SciPy)"
else
    echo "  ⚠️ Warning: Patch directory not found!"
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

# Ensure log dir exists (for our patch)
mkdir -p logs

echo "✅ Setup Complete!"
echo "👉 Run with: ./venv/bin/python3 run.py"
