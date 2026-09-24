"""
realtime_demo.py: Real-time ISL recognition loop with MediaPipe, Bi-GRU inference,
continuous gloss accumulator, and NLP sentence synthesis.
"""

import cv2
import json
import numpy as np
import keras
import mediapipe as mp
from model.src.extract_landmarks import normalize_frame_coordinates
from nlp_engine import synthesize_sentence, speak_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v for k, v in raw_map.items()}

# State trackers
frame_buffer = []
detected_glosses = []
final_sentence = ""
current_sign = "Waiting..."
current_conf = 0.0

CONFIDENCE_THRESHOLD = 0.60
TARGET_FRAMES = 32

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Draw hand landmarks on screen
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract 225-dim vector (LH: 63, Pose: 99, RH: 63)
        lh = np.zeros(63, dtype=np.float32)
        pose = np.zeros(99, dtype=np.float32)
        rh = np.zeros(63, dtype=np.float32)

        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()

        raw_vec = np.concatenate([lh, pose, rh])
        norm_vec = normalize_frame_coordinates(raw_vec)
        frame_buffer.append(norm_vec)

        # Maintain 32-frame rolling buffer
        if len(frame_buffer) > TARGET_FRAMES:
            frame_buffer.pop(0)

        # Predict every 10 frames once buffer is full
        if len(frame_buffer) == TARGET_FRAMES and len(frame_buffer) % 10 == 0:
            seq = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
            probs = model.predict(seq, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            current_conf = float(probs[top_idx])

            if current_conf >= CONFIDENCE_THRESHOLD:
                predicted_word = class_map.get(top_idx, "")
                current_sign = predicted_word

                # Append to sentence accumulator if different from last confirmed sign
                if not detected_glosses or detected_glosses[-1] != predicted_word:
                    detected_glosses.append(predicted_word)
            else:
                current_sign = "..."

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
        cv2.putText(frame, f"Sign: {current_sign.upper()} ({current_conf*100:4.1f}%)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
        cv2.putText(frame, f"Words: {' '.join(detected_glosses)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if final_sentence:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 100, 0), -1)
            cv2.putText(frame, f"Sentence: {final_sentence}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Indian Sign Language Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACEBAR to translate sequence to English sentence
            if detected_glosses:
                final_sentence = synthesize_sentence(detected_glosses)
                speak_sentence(final_sentence)
        elif key == ord('c'):  # 'C' to clear buffer
            detected_glosses.clear()
            final_sentence = ""
            current_sign = "Cleared"

cap.release()
cv2.destroyAllWindows()