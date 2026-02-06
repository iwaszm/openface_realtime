# OpenFace Realtime

A lightweight, real-time facial behavior analysis tool built on PyTorch. It provides face detection, landmark alignment, head pose estimation (gaze), and Action Unit (AU) recognition.

![OpenFace Demo](https://github.com/TadasBaltrusaitis/OpenFace/raw/master/imgs/multi_face.png)
*(Note: Conceptual demo image)*

## Features
- **Real-time**: Optimized for webcam stream input.
- **Multitask**: Simultaneous emotion recognition, gaze tracking, and AU detection.
- **Easy Setup**: Automated script handles dependencies, model weights, and library patches.

## Installation

This project requires Python 3.10+ (Tested on 3.12).

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd OpenFace-Realtime
   ```

2. **Run Setup Script**:
   This script will create a virtual environment, install dependencies, patch the underlying `openface` library for compatibility, and download the necessary model weights (~250MB).
   ```bash
   bash setup.sh
   ```

## Usage

Activate the environment and run the analyzer:

```bash
# Method 1: Using the venv python directly
./venv/bin/python3 run.py

# Method 2: Activate first
source venv/bin/activate
python run.py
```

Press **`q`** to exit the application.

## Troubleshooting

- **"No Face Detected"**: Ensure your lighting is good. The model expects RGB images (handled automatically in `run.py`).
- **Import Errors**: If you see errors related to `scipy` or `cv2`, try running `bash setup.sh` again to ensure patches are applied.

## Credits
Based on the [OpenFace 2.0/3.0](https://github.com/TadasBaltrusaitis/OpenFace) research and the PyTorch implementation.
