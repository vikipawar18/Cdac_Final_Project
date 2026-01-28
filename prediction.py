import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

def predict_video(video_path, model_path='deepfake_model_rebuilt.h5', seq_length=10):
    """
    Optimized video prediction - processes fewer frames for faster results
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Initializing face detector...")
    detector = MTCNN()
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Optimize: Only process first 5 seconds or 150 frames max
    max_frames_to_check = min(total_frames, int(fps * 5) if fps > 0 else 150)
    interval = max(1, max_frames_to_check // seq_length)
    
    print(f"Processing {seq_length} frames from {max_frames_to_check} total frames...")
    
    frames = []
    curr_frame = 0
    frames_checked = 0
    
    while cap.isOpened() and len(frames) < seq_length and frames_checked < max_frames_to_check:
        ret, frame = cap.read()
        if not ret: break
        
        if curr_frame % interval == 0:
            frames_checked += 1
            # Resize frame first for faster face detection
            small_frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            results = detector.detect_faces(frame_rgb)
            if results:
                # Scale coordinates back to original frame size
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                
                x, y, w, h = max(results, key=lambda x: x['confidence'])['box']
                x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
                
                face = cv2.resize(frame[max(0,y):y+h, max(0,x):x+w], (128, 128))
                face = tf.keras.applications.resnet50.preprocess_input(face)
                frames.append(face)
                print(f"  Frame {len(frames)}/{seq_length} processed")
        
        curr_frame += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames with faces")

    # Pad with zeros if needed
    while len(frames) < seq_length: 
        frames.append(np.zeros((128, 128, 3)))
    
    print("Running model prediction...")
    input_tensor = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(prediction)
    
    label = "FAKE" if class_idx == 1 else "REAL"
    confidence = prediction[0][class_idx]
    
    print(f"Prediction complete: {label} ({confidence*100:.2f}%)")
    return label, confidence

if __name__ == "__main__":
    label, confidence = predict_video('path_to_video.mp4')
    print(f"Prediction: {label} ({confidence*100:.2f}%)")