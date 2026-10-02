"""
realtime_demo.py: True Live Real-Time Continuous ISL to Speech Pipeline.
Maintains a rolling 32-frame sliding window with temporal cooldown
to reliably capture sequential gestures directly from webcam.
"""

import cv2
import json
import time
import numpy as np
import keras
import mediapipe as mp
from collections import deque
from nlp_engine import synthesize_sentence, speak_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"
MAX_FRAMES = 32
NUM_FEATURES = 225
CONFIDENCE_THRESHOLD = 0.65  # Must be >= 65% confident
COOLDOWN_SECONDS = 1.6       # Time to transition between signs

print("\n" + "=" * 65)
print("   LIVE REAL-TIME ISL RECOGNITION & SPEECH PIPELINE")
print("=" * 65)

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v.lower() for k, v in raw_map.items()}

def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """Torso-relative translation and distance scaling."""
    ls_x, ls_y = coords[159], coords[160]
    rs_x, rs_y = coords[162], coords[163]

    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        anchor_x, anchor_y, torso_scale = 0.5, 0.5, 1.0

    normalized = coords.copy()
    for i in range(0, NUM_FEATURES, 3):
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale

    return normalized

# Rolling frame buffer (always holds last 32 frames)
frame_buffer = deque(maxlen=MAX_FRAMES)
detected_glosses = []
final_sentence = ""
last_pred_text = "Ready - Show sign"
last_pred_conf = 0.0
last_detection_time = 0

# Carry-forward memory for landmark stability
last_lh = np.zeros(63, dtype=np.float32)
last_rh = np.zeros(63, dtype=np.float32)
last_pose = np.zeros(99, dtype=np.float32)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_tick = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tick += 1
        h, w, _ = frame.shape
        display_frame = cv2.flip(frame, 1)

        # Process landmarks on unmirrored frame to preserve left/right orientation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        lh_active = results.left_hand_landmarks is not None
        rh_active = results.right_hand_landmarks is not None

        # Draw on mirrored display
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Update landmarks or hold last known position
        if lh_active:
            last_lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        if rh_active:
            last_rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.pose_landmarks:
            last_pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()

        raw_vec = np.concatenate([last_lh.copy(), last_rh.copy(), last_pose.copy()])
        norm_vec = normalize_frame_coordinates(raw_vec)
        frame_buffer.append(norm_vec)

        current_time = time.time()
        is_cooling_down = (current_time - last_detection_time) < COOLDOWN_SECONDS

        # Run inference every 8 frames when buffer is full and not in cooldown
        if len(frame_buffer) == MAX_FRAMES and frame_tick % 8 == 0 and not is_cooling_down:
            if lh_active or rh_active:
                batch_in = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
                probs = model.predict(batch_in, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                conf = float(probs[top_idx])

                if conf >= CONFIDENCE_THRESHOLD:
                    pred_word = class_map[top_idx]
                    last_pred_text = pred_word.upper()
                    last_pred_conf = conf * 100
                    last_detection_time = current_time

                    if not detected_glosses or detected_glosses[-1] != pred_word:
                        detected_glosses.append(pred_word)
                        print(f"\n[DETECTED]: {last_pred_text} ({last_pred_conf:.1f}%)")
                        print(f"Sequence So Far: {' -> '.join(detected_glosses)}")

        # --- UI Overlay ---
        cv2.rectangle(display_frame, (0, 0), (w, 110), (25, 25, 25), -1)

        if is_cooling_down:
            status_banner = f"LOCKED: {last_pred_text} ({last_pred_conf:.1f}%) - NEXT SIGN IN {COOLDOWN_SECONDS - (current_time - last_detection_time):.1f}s"
            banner_color = (0, 165, 255)
        else:
            status_banner = f"ACTIVE: {last_pred_text} ({last_pred_conf:.1f}%)" if last_pred_conf > 0 else "READY - PERFORM SIGN"
            banner_color = (0, 255, 120)

        cv2.putText(display_frame, status_banner, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, banner_color, 2)
        cv2.putText(display_frame, f"Accumulated: {' -> '.join(detected_glosses)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "[SPACE] Synthesize & Speak  |  [C] Clear  |  [ESC] Exit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if final_sentence:
            cv2.rectangle(display_frame, (0, h - 60), (w, h), (0, 100, 0), -1)
            cv2.putText(display_frame, f"Spoken Sentence: {final_sentence}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Indian Sign Language to Speech System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (32, 13):  # SPACE or ENTER -> Run NLP and Speak
            if detected_glosses:
                final_sentence = synthesize_sentence(detected_glosses)
                print(f"\n[NLP SYNTHESIS]: {final_sentence}")
                speak_sentence(final_sentence)
        elif key == ord('c'):  # 'C' -> Reset
            detected_glosses.clear()
            final_sentence = ""
            last_pred_text = "Cleared - Ready"
            last_pred_conf = 0.0

cap.release()
cv2.destroyAllWindows()