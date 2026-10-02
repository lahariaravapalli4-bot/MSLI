"""
realtime_demo.py: Real-time ISL recognition loop matching isl_model_best.keras.
Vector layout: Left Hand (63) -> Right Hand (63) -> Pose (99) = 225
Self-contained normalization avoids triggering model.train re-execution.
"""

import cv2
import json
import numpy as np
import keras
import mediapipe as mp
from nlp_engine import synthesize_sentence, speak_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"
MAX_FRAMES = 32
NUM_FEATURES = 225


def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Applies torso-relative spatial translation and distance scaling.
    Vector mapping:
      - left_hand:  indices 0..62    (21 keypoints * 3)
      - right_hand: indices 63..125  (21 keypoints * 3)
      - pose:       indices 126..224 (33 keypoints * 3)
    Keypoint 11 (Left Shoulder) in pose slice: offset = 126 + (11 * 3) = 159
    Keypoint 12 (Right Shoulder) in pose slice: offset = 126 + (12 * 3) = 162
    """
    ls_x = coords[159]
    ls_y = coords[160]
    rs_x = coords[162]
    rs_y = coords[163]

    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        anchor_x = 0.5
        anchor_y = 0.5
        torso_scale = 1.0

    normalized = coords.copy()

    for i in range(0, NUM_FEATURES, 3):
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale

    return normalized


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v for k, v in raw_map.items()}

# State trackers
is_signing = False
recorded_frames = []
detected_glosses = []
final_sentence = ""
last_prediction = "IDLE (Raise hands to sign)"
last_conf = 0.0

rest_frame_count = 0
REST_THRESHOLD = 10       # ~0.35s rest triggers classification
MIN_GESTURE_FRAMES = 12   # Minimum frames to qualify as a deliberate sign

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        display_frame = cv2.flip(frame, 1)

        # MediaPipe on unmirrored frame to preserve physical hand orientation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        left_hand_present = results.left_hand_landmarks is not None
        right_hand_present = results.right_hand_landmarks is not None
        hands_active = left_hand_present or right_hand_present

        # Landmarks display
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract 225-dim vector strictly in training order:
        # 1. Left Hand (0..62)
        # 2. Right Hand (63..125)
        # 3. Pose (126..224)
        lh = np.zeros(63, dtype=np.float32)
        rh = np.zeros(63, dtype=np.float32)
        pose = np.zeros(99, dtype=np.float32)

        if left_hand_present:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        if right_hand_present:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()

        # [LH (63), RH (63), POSE (99)]
        raw_vec = np.concatenate([lh, rh, pose])
        norm_vec = normalize_frame_coordinates(raw_vec)

        # Gesture boundary detection
        if hands_active:
            rest_frame_count = 0
            if not is_signing:
                is_signing = True
                recorded_frames = []
            recorded_frames.append(norm_vec)
        else:
            if is_signing:
                rest_frame_count += 1
                if rest_frame_count >= REST_THRESHOLD:
                    is_signing = False
                    n = len(recorded_frames)

                    if n >= MIN_GESTURE_FRAMES:
                        seq = np.array(recorded_frames, dtype=np.float32)

                        # Uniform resampling matching parse_parquet
                        if n == MAX_FRAMES:
                            processed_seq = seq
                        elif n > MAX_FRAMES:
                            idx = np.linspace(0, n - 1, MAX_FRAMES, dtype=int)
                            processed_seq = seq[idx]
                        else:
                            pad = MAX_FRAMES - n
                            processed_seq = np.pad(seq, ((0, pad), (0, 0)), mode="constant")

                        batch_in = np.expand_dims(processed_seq, axis=0)
                        probs = model.predict(batch_in, verbose=0)[0]

                        top3_idx = np.argsort(probs)[-3:][::-1]
                        top_idx = int(top3_idx[0])
                        last_conf = float(probs[top_idx]) * 100
                        word_pred = class_map.get(top_idx, "unknown")
                        last_prediction = word_pred.upper()

                        print(f"\n[DETECTED]: {last_prediction} ({last_conf:.1f}%)")
                        print(f"  Top 2 : {class_map.get(top3_idx[1])} ({probs[top3_idx[1]]*100:.1f}%)")
                        print(f"  Top 3 : {class_map.get(top3_idx[2])} ({probs[top3_idx[2]]*100:.1f}%)")

                        if last_conf >= 45.0 and (not detected_glosses or detected_glosses[-1] != word_pred):
                            detected_glosses.append(word_pred)

                    recorded_frames = []
                    rest_frame_count = 0

        # UI Overlay
        cv2.rectangle(display_frame, (0, 0), (w, 100), (20, 20, 20), -1)

        if is_signing:
            status_text = f"RECORDING SIGN... ({len(recorded_frames)} frames)"
            status_color = (0, 140, 255)
        else:
            status_text = f"SIGN: {last_prediction} ({last_conf:.1f}%)" if last_conf > 0 else last_prediction
            status_color = (0, 255, 180)

        cv2.putText(display_frame, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_frame, f"Words: {' -> '.join(detected_glosses)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if final_sentence:
            cv2.rectangle(display_frame, (0, h - 60), (w, h), (0, 120, 0), -1)
            cv2.putText(display_frame, f"Sentence: {final_sentence}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Indian Sign Language Recognition System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (13, 32):  # ENTER or SPACEBAR
            if detected_glosses:
                final_sentence = synthesize_sentence(detected_glosses)
                print(f"\n[NLP SENTENCE]: {final_sentence}")
                speak_sentence(final_sentence)
        elif key == ord('c'):  # 'C'
            detected_glosses.clear()
            final_sentence = ""
            last_prediction = "IDLE (Raise hands to sign)"
            last_conf = 0.0

cap.release()
cv2.destroyAllWindows()