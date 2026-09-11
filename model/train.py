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
CACHE_FILE = "model/data/preprocessed_data.npz"

MAX_FRAMES = 32
NUM_FEATURES = 225  # (21 left + 21 right + 33 pose) * 3 coords
BATCH_SIZE = 64     # 64 processes significantly faster on CPU than 16 or 32
EPOCHS = 40

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_parquet(file_path):
    """Extracts LH, RH, and Pose coordinates, dropping Face."""
    df = pd.read_parquet(file_path)
    # Exclude face points to match the 225-feature vector
    df = df[df["type"].isin(["left_hand", "right_hand", "pose"])]
    
    # Sort to ensure consistent landmark order across frames
    df = df.sort_values(by=["frame", "type", "landmark_index"])
    
    frames = []
    for _, group in df.groupby("frame"):
        coords = group[["x", "y", "z"]].to_numpy(dtype=np.float32).flatten()
        if len(coords) == NUM_FEATURES:
            frames.append(coords)
            
    if not frames:
        return np.zeros((MAX_FRAMES, NUM_FEATURES), dtype=np.float32)
        
    seq = np.array(frames, dtype=np.float32)
    n = len(seq)
    
    # Temporal resampling or zero-padding to MAX_FRAMES
    if n == MAX_FRAMES:
        return seq
    elif n > MAX_FRAMES:
        idx = np.linspace(0, n - 1, MAX_FRAMES, dtype=int)
        return seq[idx]
    else:
        pad = MAX_FRAMES - n
        return np.pad(seq, ((0, pad), (0, 0)), mode="constant")

# --- 1. Load or Build RAM Cache ---
if os.path.exists(CACHE_FILE):
    print(">> Loading preprocessed data directly from RAM cache...")
    cache = np.load(CACHE_FILE)
    X, y, classes = cache["X"], cache["y"], cache["classes"]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
else:
    print(">> First-time processing: Caching landmarks into RAM (takes ~3-5 mins once)...")
    parquet_files = glob.glob(f"{DATA_DIR}/*/*/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")

    # Discover classes from parent directory names
    classes = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in parquet_files)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    X_list, y_list = [], []
    total = len(parquet_files)
    
    for idx, f in enumerate(parquet_files):
        label = os.path.basename(os.path.dirname(f))
        feats = parse_parquet(f)
        X_list.append(feats)
        y_list.append(class_to_idx[label])
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"   Processed {idx + 1}/{total} files...")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    # Save compact cache to disk for instant future loads (~35 MB)
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=classes)
    print(f">> Preprocessed cache saved to {CACHE_FILE}")

# Save class map
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, indent=2)

num_classes = len(classes)
print(f">> Dataset ready: {len(X)} samples, {num_classes} classes.")

# --- 2. Train/Val Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. CPU-Optimized Model Architecture ---
# Using GRU instead of LSTM: 30% fewer matrix ops, identical sequence accuracy
model = models.Sequential([
    layers.Input(shape=(MAX_FRAMES, NUM_FEATURES)),
    layers.Masking(mask_value=0.0),
    layers.Bidirectional(layers.GRU(64, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.GRU(32)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
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

print(">> Starting fast training on CPU...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cb
)

print(f"\n>> Training complete! Artifact saved to: {MODEL_SAVE_PATH}")