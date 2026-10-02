"""
visual_demo.py: Flawless Visual Skeleton Demonstration for B.Tech Defense.
Renders real signing skeletons dynamically on screen, feeds vectors to Bi-GRU,
and speaks synthesized English sentences without needing a physical webcam.
"""

import os
import glob
import json
import time
import cv2
import numpy as np
import keras
import pyarrow.parquet as pq
from model.train import parse_parquet
from nlp_engine import synthesize_sentence, speak_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]

print("\n" + "=" * 65)
print("   INDIAN SIGN LANGUAGE TO SPEECH: SKELETON VISUAL DEMO")
print("=" * 65)

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v.lower() for k, v in raw_map.items()}

def parse_parquet_raw(file_path: str):
    """Loads raw landmark coordinates and sanitizes NaNs to 0.0."""
    table = pq.read_table(file_path)
    df = table.to_pandas()
    frames = sorted(df['frame'].unique())
    
    seq_raw = []
    for f in frames:
        f_df = df[df['frame'] == f]
        coords = np.zeros(225, dtype=np.float32)
        for _, row in f_df.iterrows():
            idx = int(row.get('landmark_index', 0))
            t = str(row.get('type', '')).lower()
            x = 0.0 if np.isnan(row.get('x', np.nan)) else float(row['x'])
            y = 0.0 if np.isnan(row.get('y', np.nan)) else float(row['y'])
            z = 0.0 if np.isnan(row.get('z', np.nan)) else float(row.get('z', 0.0))
            
            if 'left' in t and idx < 21:
                coords[idx*3 : idx*3+3] = [x, y, z]
            elif 'right' in t and idx < 21:
                coords[63 + idx*3 : 63 + idx*3+3] = [x, y, z]
            elif 'pose' in t and idx < 33:
                coords[126 + idx*3 : 126 + idx*3+3] = [x, y, z]
        seq_raw.append(coords)
    return np.nan_to_num(np.array(seq_raw, dtype=np.float32), nan=0.0)

def draw_skeleton(canvas, frame_coords, w=640, h=480):
    """Draws upper body and hand joints onto canvas with strict NaN protection."""
    lh = frame_coords[0:63].reshape(21, 3)
    rh = frame_coords[63:126].reshape(21, 3)
    pose = frame_coords[126:225].reshape(33, 3)

    # Draw Pose
    for start, end in POSE_CONNECTIONS:
        if (not np.isnan(pose[start, 0])) and (not np.isnan(pose[end, 0])):
            if pose[start, 0] > 0.0 and pose[end, 0] > 0.0:
                pt1 = (int(pose[start, 0] * w), int(pose[start, 1] * h))
                pt2 = (int(pose[end, 0] * w), int(pose[end, 1] * h))
                cv2.line(canvas, pt1, pt2, (255, 255, 0), 2)
                cv2.circle(canvas, pt1, 4, (0, 255, 255), -1)
                cv2.circle(canvas, pt2, 4, (0, 255, 255), -1)

    # Draw Left Hand
    for start, end in HAND_CONNECTIONS:
        if (not np.isnan(lh[start, 0])) and (not np.isnan(lh[end, 0])):
            if lh[start, 0] > 0.0 and lh[end, 0] > 0.0:
                pt1 = (int(lh[start, 0] * w), int(lh[start, 1] * h))
                pt2 = (int(lh[end, 0] * w), int(lh[end, 1] * h))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(canvas, pt1, 3, (0, 255, 128), -1)

    # Draw Right Hand
    for start, end in HAND_CONNECTIONS:
        if (not np.isnan(rh[start, 0])) and (not np.isnan(rh[end, 0])):
            if rh[start, 0] > 0.0 and rh[end, 0] > 0.0:
                pt1 = (int(rh[start, 0] * w), int(rh[start, 1] * h))
                pt2 = (int(rh[end, 0] * w), int(rh[end, 1] * h))
                cv2.line(canvas, pt1, pt2, (0, 140, 255), 2)
                cv2.circle(canvas, pt1, 3, (0, 180, 255), -1)

def play_and_infer_sign(word_name: str, file_path: str):
    raw_seq = parse_parquet_raw(file_path)
    n = len(raw_seq)

    # Run inference using the verified normalization routine
    features = parse_parquet(file_path)
    batch_in = np.expand_dims(features, axis=0)
    probs = model.predict(batch_in, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    pred_word = class_map[top_idx]
    conf = probs[top_idx] * 100

    window_name = "ISL Vision Pipeline: Skeleton Tracking & Gesture Decoding"
    cv2.namedWindow(window_name)

    for i in range(n):
        canvas = np.zeros((520, 640, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        # Header HUD
        cv2.rectangle(canvas, (0, 0), (640, 70), (40, 40, 40), -1)
        cv2.putText(canvas, f"Tracking Sign: {word_name.upper()} [Frame {i+1}/{n}]", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(canvas, "Feature Extraction: 225 Keypoints (Pose + Hands)", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Render Skeleton
        draw_skeleton(canvas, raw_seq[i], w=640, h=480)

        # Bottom Prediction Banner
        cv2.rectangle(canvas, (0, 460), (640, 520), (15, 60, 20), -1)
        cv2.putText(canvas, f"Bi-GRU Recognized: {pred_word.upper()} ({conf:.1f}%)", (20, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(40) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return pred_word, conf

demo_sentences = [
    ["i", "alive", "sad"],
    ["car", "shirt", "tuesday"]
]

print("\nPress Enter to begin the Visual Presentation Demo...")
input()

for sentence_words in demo_sentences:
    recognized_glosses = []
    print(f"\n--- Demonstrating Gesture Sequence: {sentence_words} ---")

    for word in sentence_words:
        matches = glob.glob(f"model/data/dataset_263/**/{word}/*.parquet", recursive=True)
        if not matches:
            continue

        print(f" -> Playing & Decoding: '{word.upper()}' ...")
        pred, conf = play_and_infer_sign(word, matches[0])
        print(f"    Recognized: {pred.upper()} with {conf:.2f}% confidence.")
        recognized_glosses.append(pred)
        time.sleep(0.4)

    if recognized_glosses:
        final_sentence = synthesize_sentence(recognized_glosses)
        print(f"\n[NLP Synthesizer]: \"{final_sentence}\"")
        print("Speaking via TTS...")
        speak_sentence(final_sentence)
        print("-" * 65)

print("\nVisual Showcase Complete.")