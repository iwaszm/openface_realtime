import cv2
import torch
import numpy as np
import sys
import os
import traceback
import csv
import datetime
import time

# Debug Imports
print("--------------------------------------------------")
print("🔍 Diagnostic: Checking imports...")
try:
    import openface
    print("✅ import openface: Success")
except ImportError as e:
    print(f"❌ import openface: Failed - {e}")

try:
    from openface.face_detection import FaceDetector
    from openface.landmark_detection import LandmarkDetector
    from openface.multitask_model import MultitaskPredictor
    print("✅ Core modules import: Success")
except ImportError as e:
    print(f"❌ Core modules import: Failed - {e}")
    traceback.print_exc()
print("--------------------------------------------------")

def draw_bars(image, values, labels, x_start, y_start, color=(255, 0, 0)):
    """Draws AU bars in two columns to fit more data."""
    col_width = 160
    bar_max_width = 60
    row_height = 20
    
    # Split into two columns
    mid = len(values) // 2 + len(values) % 2
    
    for i, (val, label) in enumerate(zip(values, labels)):
        # Determine column and row
        if i < mid:
            cur_x = x_start
            cur_y = y_start + i * row_height
        else:
            cur_x = x_start + col_width
            cur_y = y_start + (i - mid) * row_height

        # Bar width proportional to value (0-1) -> pixels
        bar_w = int(val * bar_max_width)
        
        # Background bar
        cv2.rectangle(image, (cur_x + 50, cur_y + 2), (cur_x + 50 + bar_max_width, cur_y + 12), (50, 50, 50), -1)
        # Active bar
        cv2.rectangle(image, (cur_x + 50, cur_y + 2), (cur_x + 50 + bar_w, cur_y + 12), color, -1)
        # Text Label
        cv2.putText(image, f"{label}", (cur_x, cur_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # Text Value
        cv2.putText(image, f"{val:.2f}", (cur_x + 50 + bar_max_width + 5, cur_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading OpenFace models on {device}...")

    if 'FaceDetector' not in globals() or 'LandmarkDetector' not in globals():
        print("❌ Critical modules missing. See import errors above.")
        return

    # --- Config ---
    FRAME_SKIP = 3  # Process 1 frame, then skip 2 (33% load)
    LOG_DIR = "logs"
    
    # --- Logging Setup ---
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_DIR, f"record_{timestamp_str}.csv")
    
    csv_file = open(log_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Headers will be written upon first detection when we know the AU labels
    headers_written = False
    
    emotions_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    try:
        # Check weights existence
        weights_dir = "./weights"
        required_weights = ['Alignment_RetinaFace.pth', 'Landmark_98.pkl', 'MTL_backbone.pth']
        for w in required_weights:
            path = os.path.join(weights_dir, w)
            if not os.path.exists(path):
                print(f"⚠️ Warning: Weight file not found: {path}")

        face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth', device=device)
        landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl', device=device)
        multitask_model = MultitaskPredictor(model_path='./weights/MTL_backbone.pth', device=device)
        
        print("✅ Models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🟢 Starting video loop. Press 'q' to exit.")
    print(f"📝 Logging data to: {log_file_path}")

    frame_count = 0
    
    # Cache for smoothing / frame skipping
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
        
        # Performance: Only run heavy detection every FRAME_SKIP frames
        do_process = (frame_count % FRAME_SKIP == 0)

        if do_process:
            try:
                # Some models expect RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cropped_face, dets = face_detector.get_face(frame_rgb)
                
                if dets is None:
                    # Retry raw
                    cropped_face, dets = face_detector.get_face(frame)

                if dets is not None and len(dets) > 0:
                    bbox_raw = dets[0]
                    # Update cache - convert map to tuple to persist data across frames
                    last_bbox = tuple(map(int, bbox_raw[:4]))
                    
                    # Landmarks
                    landmarks = landmark_detector.detect_landmarks(frame_rgb, dets)
                    if landmarks:
                        last_landmarks = landmarks[0]

                    # Emotions / Gaze / AU
                    if cropped_face is not None:
                        emo_logits, gaze, au = multitask_model.predict(cropped_face)
                        
                        # Process
                        emo_probs = torch.softmax(emo_logits, dim=1).detach().cpu().numpy()[0]
                        yaw, pitch = gaze[0].detach().cpu().numpy()
                        au_vals = au[0].detach().cpu().numpy()
                        
                        # AU Labels (usually fixed, but let's generate)
                        if last_au_labels is None:
                            last_au_labels = [f"AU_{i+1:02d}" for i in range(len(au_vals))]

                        # Update cache
                        last_emo = emo_probs
                        last_gaze = (yaw, pitch)
                        last_au = au_vals
                        
                        # --- CSV Logging ---
                        if not headers_written:
                            headers = ['Timestamp', 'Frame_ID'] + \
                                      [f"Emo_{label}" for label in emotions_labels] + \
                                      ['Gaze_Yaw', 'Gaze_Pitch'] + \
                                      last_au_labels
                            csv_writer.writerow(headers)
                            headers_written = True
                        
                        # Write Data Row
                        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        row = [current_time, frame_count] + \
                              last_emo.tolist() + \
                              [float(last_gaze[0]), float(last_gaze[1])] + \
                              last_au.tolist()
                        csv_writer.writerow(row)
                        
                else:
                    last_bbox = None # Lost face
            
            except Exception as e:
                print(f"⚠️ Warning: {e}")
                pass

        # --- Rendering (Every Frame) ---
        # Draw using cached data (this makes it look smooth even if we skipped processing)
        if last_bbox:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if last_landmarks is not None:
                for (x, y) in last_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
            
            if last_emo is not None:
                emo_idx = np.argmax(last_emo)
                label = emotions_labels[emo_idx]
                score = last_emo[emo_idx]
                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if last_gaze is not None:
                yaw, pitch = last_gaze
                center_x, center_y = (x1+x2)//2, (y1+y2)//2
                length = 100
                dx = -length * np.sin(yaw)
                dy = -length * np.sin(pitch)
                cv2.arrowedLine(frame, (center_x, center_y), (int(center_x + dx), int(center_y + dy)), (0, 0, 255), 2)
            
            if last_au is not None and last_au_labels is not None:
                # Draw ALL AUs in 2 columns
                draw_bars(frame, last_au, last_au_labels, 10, 30)
        else:
            cv2.putText(frame, "Detecting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("OpenFace 3.0 Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"💾 Data saved to {log_file_path}")

if __name__ == "__main__":
    main()
