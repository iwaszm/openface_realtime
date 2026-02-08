import cv2
import torch
import numpy as np
import sys
import os
import traceback
import csv
import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# --- GUI Config Class ---
class ConfigDialog:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenFace Setup")
        self.root.geometry("450x600")
        
        # Configuration Variables
        self.show_au = tk.BooleanVar(value=True)
        self.show_emotion = tk.BooleanVar(value=True)
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_gaze = tk.BooleanVar(value=True)
        self.frame_skip = tk.IntVar(value=3)
        self.record_aus = [] # List of BooleanVars
        self.started = False

        self._create_widgets()

    def _create_widgets(self):
        # 1. Visualization Settings
        viz_frame = ttk.LabelFrame(self.root, text=" Visualization Settings ", padding=10)
        viz_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(viz_frame, text="Show Action Units (AU)", variable=self.show_au).pack(anchor="w")
        ttk.Checkbutton(viz_frame, text="Show Emotions", variable=self.show_emotion).pack(anchor="w")
        ttk.Checkbutton(viz_frame, text="Show Landmarks", variable=self.show_landmarks).pack(anchor="w")
        ttk.Checkbutton(viz_frame, text="Show Gaze", variable=self.show_gaze).pack(anchor="w")

        # 2. Recording Settings
        rec_frame = ttk.LabelFrame(self.root, text=" Recording Settings ", padding=10)
        rec_frame.pack(fill="x", padx=10, pady=5)
        
        # Frame Skip Control (Frequency)
        # 1 = 30 FPS (Full speed), 2 = 15 FPS, 3 = 10 FPS, etc.
        # We will label it as "Approx. FPS" for clarity
        fps_frame = ttk.Frame(rec_frame)
        fps_frame.pack(fill="x", pady=5)
        
        ttk.Label(fps_frame, text="Processing Speed:").pack(side="left")
        self.fps_label = ttk.Label(fps_frame, text="Max (Every Frame)", font=('Arial', 9, 'bold'), foreground="blue")
        self.fps_label.pack(side="right")
        
        self.frame_skip = tk.IntVar(value=3)
        scale = ttk.Scale(rec_frame, from_=1, to=10, variable=self.frame_skip, orient="horizontal", command=self._update_fps_label)
        scale.pack(fill="x")
        
        # Initialize label
        self._update_fps_label(3)

        ttk.Label(rec_frame, text="Select AUs to Record & Display:").pack(anchor="w", pady=(10, 0))
        au_frame = ttk.Frame(rec_frame)
        au_frame.pack(fill="x")
        
        # Assume standard 17 AUs for now (names or indices)
        au_names = [f"AU{i+1:02d}" for i in range(17)]
        for i, name in enumerate(au_names):
            var = tk.BooleanVar(value=True)
            self.record_aus.append(var)
            chk = ttk.Checkbutton(au_frame, text=name, variable=var)
            # Grid layout for AUs
            chk.grid(row=i//4, column=i%4, sticky="w")

        # 3. Instructions
        info_frame = ttk.LabelFrame(self.root, text=" Instructions ", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        info_text = (
            "• Press 'Start' to open the webcam window.\n"
            "• Press 'q' in the video window to Stop & Exit.\n"
            "• Data is saved automatically to 'logs/' folder."
        )
        ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w")

        # 4. Buttons
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Start Analysis", command=self.start).pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Exit", command=self.root.destroy).pack(fill="x")

    def _update_fps_label(self, val):
        skip = int(float(val))
        # Assuming webcam is ~30 FPS
        approx_fps = 30 / skip
        if skip == 1:
            self.fps_label.config(text=f"{approx_fps:.0f} FPS (Every Frame)")
        else:
            self.fps_label.config(text=f"~{approx_fps:.1f} FPS (Skip {skip-1})")

    def start(self):
        self.started = True
        self.root.destroy()

# --- Main Logic ---
def run_app():
    # Show GUI first
    root = tk.Tk()
    app = ConfigDialog(root)
    root.mainloop()

    if not app.started:
        print("❌ User cancelled setup.")
        return

    # Extract configs
    CONFIG = {
        'show_au': app.show_au.get(),
        'show_emotion': app.show_emotion.get(),
        'show_landmarks': app.show_landmarks.get(),
        'show_gaze': app.show_gaze.get(),
        'frame_skip': app.frame_skip.get(),
        'au_mask': [v.get() for v in app.record_aus]
    }

    # ... [Rest of the analysis code, adapted to use CONFIG] ...
    start_analysis(CONFIG)

def draw_bars(image, values, labels, x_start, y_start, color=(255, 0, 0)):
    """Draws AU bars in two columns to fit more data."""
    col_width = 160
    bar_max_width = 60
    row_height = 20
    
    mid = len(values) // 2 + len(values) % 2
    
    for i, (val, label) in enumerate(zip(values, labels)):
        if i < mid:
            cur_x = x_start
            cur_y = y_start + i * row_height
        else:
            cur_x = x_start + col_width
            cur_y = y_start + (i - mid) * row_height

        bar_w = int(val * bar_max_width)
        cv2.rectangle(image, (cur_x + 50, cur_y + 2), (cur_x + 50 + bar_max_width, cur_y + 12), (50, 50, 50), -1)
        cv2.rectangle(image, (cur_x + 50, cur_y + 2), (cur_x + 50 + bar_w, cur_y + 12), color, -1)
        cv2.putText(image, f"{label}", (cur_x, cur_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f"{val:.2f}", (cur_x + 50 + bar_max_width + 5, cur_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

def start_analysis(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading OpenFace models on {device}...")

    # Imports (Moved here to avoid delay before GUI)
    try:
        import openface
        from openface.face_detection import FaceDetector
        from openface.landmark_detection import LandmarkDetector
        from openface.multitask_model import MultitaskPredictor
    except ImportError:
        print("❌ Error importing OpenFace. Check setup.")
        return

    # Check weights
    weights_dir = "./weights"
    required_weights = ['Alignment_RetinaFace.pth', 'Landmark_98.pkl', 'MTL_backbone.pth']
    for w in required_weights:
        if not os.path.exists(os.path.join(weights_dir, w)):
            print(f"⚠️ Warning: Weight file not found: {w}")

    try:
        face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth', device=device)
        landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl', device=device)
        multitask_model = MultitaskPredictor(model_path='./weights/MTL_backbone.pth', device=device)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Logging Setup
    os.makedirs("logs", exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join("logs", f"record_{timestamp_str}.csv")
    csv_file = open(log_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    headers_written = False
    
    emotions_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🟢 Starting video loop.")
    frame_count = 0
    
    # Cache
    last_bbox = None
    last_landmarks = None
    last_emo = None
    last_gaze = None
    last_au = None
    last_au_labels = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        do_process = (frame_count % config['frame_skip'] == 0)

        if do_process:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_face, dets = face_detector.get_face(frame_rgb)
                
                if dets is None:
                    cropped_face, dets = face_detector.get_face(frame)

                if dets is not None and len(dets) > 0:
                    bbox_raw = dets[0]
                    last_bbox = tuple(map(int, bbox_raw[:4]))
                    
                    landmarks = landmark_detector.detect_landmarks(frame_rgb, dets)
                    if landmarks:
                        last_landmarks = landmarks[0]

                    if cropped_face is not None:
                        emo_logits, gaze, au = multitask_model.predict(cropped_face)
                        
                        last_emo = torch.softmax(emo_logits, dim=1).detach().cpu().numpy()[0]
                        yaw, pitch = gaze[0].detach().cpu().numpy()
                        last_gaze = (yaw, pitch)
                        last_au = au[0].detach().cpu().numpy()
                        
                        if last_au_labels is None:
                            last_au_labels = [f"AU{i+1:02d}" for i in range(len(last_au))]

                        # --- CSV Logging ---
                        if not headers_written:
                            # Filter headers based on AU selection
                            selected_au_headers = [label for i, label in enumerate(last_au_labels) 
                                                 if i < len(config['au_mask']) and config['au_mask'][i]]
                            
                            headers = ['Timestamp', 'Frame_ID'] + \
                                      [f"Emo_{label}" for label in emotions_labels] + \
                                      ['Gaze_Yaw', 'Gaze_Pitch'] + \
                                      selected_au_headers
                            csv_writer.writerow(headers)
                            headers_written = True
                        
                        # Filter AU data
                        selected_au_vals = [val for i, val in enumerate(last_au) 
                                          if i < len(config['au_mask']) and config['au_mask'][i]]

                        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        row = [current_time, frame_count] + \
                              last_emo.tolist() + \
                              [float(last_gaze[0]), float(last_gaze[1])] + \
                              selected_au_vals
                        csv_writer.writerow(row)
                else:
                    last_bbox = None
            except Exception:
                pass

        # --- Rendering ---
        if last_bbox:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if config['show_landmarks'] and last_landmarks is not None:
                for (x, y) in last_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
            
            if config['show_emotion'] and last_emo is not None:
                emo_idx = np.argmax(last_emo)
                label = emotions_labels[emo_idx]
                score = last_emo[emo_idx]
                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if config['show_gaze'] and last_gaze is not None:
                yaw, pitch = last_gaze
                center_x, center_y = (x1+x2)//2, (y1+y2)//2
                length = 100
                dx = -length * np.sin(yaw)
                dy = -length * np.sin(pitch)
                cv2.arrowedLine(frame, (center_x, center_y), (int(center_x + dx), int(center_y + dy)), (0, 0, 255), 2)
            
            if config['show_au'] and last_au is not None:
                # Filter AUs for DISPLAY based on selection
                if last_au_labels is not None:
                    # Create filtered lists for display
                    display_vals = []
                    display_labels = []
                    
                    for i, (val, label) in enumerate(zip(last_au, last_au_labels)):
                        # If mask is shorter than data (shouldn't happen), assume True
                        # If i is within mask range and mask is True, show it
                        if i < len(config['au_mask']) and config['au_mask'][i]:
                            display_vals.append(val)
                            display_labels.append(label)
                    
                    if display_vals:
                        draw_bars(frame, display_vals, display_labels, 10, 30)
        else:
            cv2.putText(frame, "Detecting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("OpenFace 3.0 Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    run_app()
