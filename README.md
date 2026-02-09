# OpenFace Realtime

A lightweight, real-time facial behavior analysis tool built on PyTorch. It provides face detection, landmark alignment, head pose estimation (gaze), and Action Unit (AU) recognition.

## Features
- **Real-time**: Optimized for webcam stream input with frame-skipping control.
- **Multitask**: Simultaneous emotion recognition, gaze tracking, and AU detection.
- **Configurable GUI**: User-friendly setup window to toggle features (AU, Gaze, etc.) and select processing device (CPU/GPU).
- **Data Recording**: Automatically saves analysis data (Emotions, AUs, Gaze) to CSV files in `logs/`.
- **Cross-Platform**: One-click setup for both Windows and Linux.

## Installation

This project requires Python 3.10+ (Tested on 3.12).

### 1. Clone the repository
```bash
git clone https://github.com/iwaszm/openface_realtime.git
cd openface_realtime
```

### 2. Setup & Install

#### 🪟 Windows Users
Simply double-click **`setup.bat`**.
*   This will create a virtual environment, install dependencies (including GPU-enabled PyTorch), download model weights (~250MB), and apply necessary patches.

#### 🐧 Linux / Mac Users
Run the setup script in your terminal:
```bash
bash setup.sh
```

## Usage

### 🪟 Windows
Double-click **`run_gui.bat`** to launch the configuration window.

### 🐧 Linux / Mac
Activate the environment and run the GUI launcher:
```bash
source venv/bin/activate
python run_gui.py
```

### ⚙️ Configuration
In the startup window, you can:
*   **Select Device**: Choose `Auto`, `CPU`, or `GPU (CUDA)` for processing.
*   **Adjust Speed**: Use the slider to skip frames for smoother performance on slower machines.
*   **Toggle Visuals**: Turn on/off landmarks, gaze lines, or emotion labels.
*   **Select AUs**: Choose which Action Units to display and record.

Press **`Start Analysis`** to begin. Press **`q`** in the video window to stop and exit.

## Troubleshooting

- **"No Face Detected"**: Ensure your lighting is good.
- **"GPU selected but not available"**: 
    - Ensure you have NVIDIA drivers installed.
    - If `setup.bat` installed the CPU version of PyTorch by mistake, run this manually in your project terminal:
      ```cmd
      venv\Scripts\pip uninstall torch torchvision -y
      venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
      ```

## Credits
Based on the [OpenFace 2.0/3.0](https://github.com/TadasBaltrusaitis/OpenFace) research and the PyTorch implementation.
