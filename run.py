import cv2
import torch
import numpy as np
import sys
import os
import traceback

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
    print("✅ FaceDetector import: Success")
except ImportError as e:
    print(f"❌ FaceDetector import: Failed - {e}")
    traceback.print_exc()

try:
    from openface.landmark_detection import LandmarkDetector
    print("✅ LandmarkDetector import: Success")
except ImportError as e:
    print(f"❌ LandmarkDetector import: Failed - {e}")
    traceback.print_exc()

try:
    from openface.multitask_model import MultitaskPredictor
    print("✅ MultitaskPredictor import: Success")
except ImportError as e:
    print(f"❌ MultitaskPredictor import: Failed - {e}")
    traceback.print_exc()
print("--------------------------------------------------")

def draw_bars(image, values, labels, x_start, y_start, color=(255, 0, 0)):
    for i, (val, label) in enumerate(zip(values, labels)):
        bar_width = int(val * 100)
        cv2.rectangle(image, (x_start, y_start + i*20), (x_start + 100, y_start + i*20 + 15), (50, 50, 50), -1)
        cv2.rectangle(image, (x_start, y_start + i*20), (x_start + bar_width, y_start + i*20 + 15), color, -1)
        cv2.putText(image, f"{label}: {val:.2f}", (x_start + 110, y_start + i*20 + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading OpenFace models on {device}...")

    if 'FaceDetector' not in globals() or 'LandmarkDetector' not in globals():
        print("❌ Critical modules missing. See import errors above.")
        return

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
    emotions_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        try:
            # Debug: Print every 30 frames if no face is found to ensure it's running
            # Some models expect RGB, OpenCV gives BGR. Let's try passing RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try detection
            cropped_face, dets = face_detector.get_face(frame_rgb) # Try RGB first
            
            if dets is None or len(dets) == 0:
                # Fallback to BGR just in case
                cropped_face, dets = face_detector.get_face(frame)

            if dets is not None and len(dets) > 0:
                if frame_count % 30 == 0:
                    print(f"✅ Face detected! (Frame {frame_count})")
                
                bbox = dets[0] 
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = landmark_detector.detect_landmarks(frame_rgb, dets) # Use RGB for landmarks too
                if landmarks:
                    for (x, y) in landmarks[0]:
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)

                if cropped_face is not None:
                    # Multitask model usually expects normalized tensor, likely handled inside.
                    # But ensuring RGB consistency is good.
                    emo_logits, gaze, au = multitask_model.predict(cropped_face) 
                    emo_probs = torch.softmax(emo_logits, dim=1).detach().cpu().numpy()[0]
                    emo_idx = np.argmax(emo_probs)
                    
                    # Display Emotion
                    cv2.putText(frame, f"{emotions_labels[emo_idx]} ({emo_probs[emo_idx]:.2f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Display Gaze (Simple Arrow)
                    if gaze is not None:
                        yaw, pitch = gaze[0].detach().cpu().numpy()
                        center_x, center_y = (x1+x2)//2, (y1+y2)//2
                        length = 100
                        dx = -length * np.sin(yaw)
                        dy = -length * np.sin(pitch)
                        cv2.arrowedLine(frame, (center_x, center_y), (int(center_x + dx), int(center_y + dy)), (0, 0, 255), 2)

                    # Display AU (Bars)
                    if au is not None:
                        au_vals = au[0].detach().cpu().numpy()
                        au_labels = [f"AU{i}" for i in range(len(au_vals))]
                        draw_bars(frame, au_vals[:10], au_labels[:10], 10, 30)
            else:
                if frame_count % 30 == 0:
                    print(f"⚠️ No face detected. (Frame {frame_count}) - Check lighting?")
                cv2.putText(frame, "No Face Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            if frame_count % 30 == 0:
                print(f"❌ Runtime Error: {e}")
            pass # Ignore runtime errors to keep stream alive

        cv2.imshow("OpenFace 3.0", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
