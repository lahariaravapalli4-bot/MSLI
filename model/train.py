import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

DATA_DIR = "model/data/dataset_263/keypoints"
CHECKPOINT_DIR = "model/checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "isl_model_best.keras")
CLASS_MAP_PATH = os.path.join(CHECKPOINT_DIR, "class_map.json")
CACHE_FILE = "model/data/preprocessed_data_normalized.npz"

MAX_FRAMES = 32
NUM_FEATURES = 225  # 21 Left Hand (63) + 21 Right Hand (63) + 33 Pose (99)
BATCH_SIZE = 64
EPOCHS = 40

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Applies torso-relative spatial translation and distance scaling.
    coords: 1D array of 225 floats representing (x, y, z) for:
      - indices 0..62:   Left Hand (21 keypoints * 3)
      - indices 63..125:  Right Hand (21 keypoints * 3)
      - indices 126..224: Upper Pose (33 keypoints * 3)
    """
    # Keypoint 11 (Left Shoulder) in pose slice: offset = 126 + (11 * 3) = 159
    ls_x = coords[159]
    ls_y = coords[160]

    # Keypoint 12 (Right Shoulder) in pose slice: offset = 126 + (12 * 3) = 162
    rs_x = coords[162]
    rs_y = coords[163]

    # If shoulders are detected, anchor to shoulder center
    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        # Fallback if pose tracking dropped shoulders in that specific frame
        anchor_x = 0.5
        anchor_y = 0.5
        torso_scale = 1.0

    normalized = coords.copy()

    # Shift (x, y) relative to torso anchor and scale by body size
    for i in range(0, NUM_FEATURES, 3):
        # Keep empty/missing joints at exact 0.0
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale
            # z remains relative depth or unshifted

    return normalized


def parse_parquet(file_path: str) -> np.ndarray:
    """Reads a parquet file, extracts skeletal features, normalizes coordinates, and standardizes to 32 frames."""
    df = pd.read_parquet(file_path)
    df = df[df["type"].isin(["left_hand", "right_hand", "pose"])]
    df = df.sort_values(by=["frame", "type", "landmark_index"])

    frames = []
    for _, group in df.groupby("frame"):
        coords = group[["x", "y", "z"]].to_numpy(dtype=np.float32).flatten()
        if len(coords) == NUM_FEATURES:
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            norm_coords = normalize_frame_coordinates(coords)
            frames.append(norm_coords)

    if not frames:
        return np.zeros((MAX_FRAMES, NUM_FEATURES), dtype=np.float32)

    seq = np.array(frames, dtype=np.float32)
    n = len(seq)

    # Resample or pad to fixed 32-frame sequence
    if n == MAX_FRAMES:
        return seq
    elif n > MAX_FRAMES:
        idx = np.linspace(0, n - 1, MAX_FRAMES, dtype=int)
        return seq[idx]
    else:
        pad = MAX_FRAMES - n
        return np.pad(seq, ((0, pad), (0, 0)), mode="constant")


# --- 1. Load or Generate Cached Data ---
if os.path.exists(CACHE_FILE):
    print(">> Loading normalized data directly from cache...")
    cache = np.load(CACHE_FILE)
    X, y, classes = cache["X"], cache["y"], cache["classes"]
else:
    print(">> Processing and normalizing parquet files into RAM cache...")
    parquet_files = glob.glob(f"{DATA_DIR}/*/*/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")

    classes = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in parquet_files)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    X_list, y_list = [], []
    total = len(parquet_files)

    for idx, f in enumerate(parquet_files):
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"   [{idx + 1}/{total}] Processing: {os.path.basename(f)}")
        label = os.path.basename(os.path.dirname(f))
        feats = parse_parquet(f)
        X_list.append(feats)
        y_list.append(class_to_idx[label])

    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_list, dtype=np.int32)
    
    print(f">> Saving normalized cache to {CACHE_FILE}...")
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=classes)
    print(">> Cache saved successfully.")

# Save class mapping
idx_to_class = {int(i): str(cls_name) for i, cls_name in enumerate(classes)}
with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, indent=2)

num_classes = len(classes)
print(f">> Dataset ready: {len(X)} samples, {num_classes} classes. NaNs remaining: {np.isnan(X).any()}")

# --- 2. Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Sequence Model Architecture ---
model = models.Sequential([
    layers.Input(shape=(MAX_FRAMES, NUM_FEATURES)),
    layers.BatchNormalization(),
    layers.Bidirectional(layers.GRU(128, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.GRU(64)),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 4. Callbacks & Training ---
cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
]

print(">> Starting model training on normalized spatial coordinates...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cb
)

print(f"\n>> Training complete! Generalized model saved to: {MODEL_SAVE_PATH}")