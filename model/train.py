import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("model", "data", "dataset_263")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")
CHECKPOINT_DIR = os.path.join("model", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_FRAMES = 32
NUM_FEATURES = 225  # 21 LH (63) + 21 RH (63) + 33 Pose (99)

# 1. Load label mapping
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Handle string or integer mappings cleanly
if isinstance(next(iter(label_map.values())), int):
    word_to_id = label_map
else:
    word_to_id = {v: int(k) for k, v in label_map.items()}

num_classes = len(word_to_id)
print(f"Total vocabulary classes: {num_classes}")

# 2. Collect parquet files & pair with labels
parquet_files = glob.glob(os.path.join(DATA_DIR, "keypoints", "**", "*.parquet"), recursive=True)
print(f"Found {len(parquet_files)} parquet takes.")

X_paths = []
y_labels = []

for file_path in parquet_files:
    # Path pattern: .../keypoints/<category>/<word>/<filename>.parquet
    word = os.path.basename(os.path.dirname(file_path)).lower()
    if word in word_to_id:
        X_paths.append(file_path)
        y_labels.append(word_to_id[word])

print(f"Matched {len(X_paths)} valid samples.")

# Train/Val Split (85% train, 15% val)
X_train, X_val, y_train, y_val = train_test_split(
    X_paths, y_labels, test_size=0.15, stratify=y_labels, random_state=42
)

# 3. Robust Parquet Reader for one take
def process_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        return np.zeros((MAX_FRAMES, NUM_FEATURES), dtype=np.float32)

    # Filter out face; retain left_hand, right_hand, pose
    df_body = df[df['type'].isin(['left_hand', 'right_hand', 'pose'])]
    df_body = df_body.fillna(0.0)

    frames = sorted(df_body['frame'].unique())
    if len(frames) == 0:
        return np.zeros((MAX_FRAMES, NUM_FEATURES), dtype=np.float32)

    seq = []
    for f in frames:
        f_df = df_body[df_body['frame'] == f]
        
        # Extract LH (21), RH (21), Pose (33)
        lh = f_df[f_df['type'] == 'left_hand'][['x', 'y', 'z']].values.flatten()
        rh = f_df[f_df['type'] == 'right_hand'][['x', 'y', 'z']].values.flatten()
        pose = f_df[f_df['type'] == 'pose'][['x', 'y', 'z']].values.flatten()

        # Pad to expected landmark dimensions if frames dropped points
        lh = np.pad(lh, (0, max(0, 63 - len(lh))))[:63]
        rh = np.pad(rh, (0, max(0, 63 - len(rh))))[:63]
        pose = np.pad(pose, (0, max(0, 99 - len(pose))))[:99]

        frame_vector = np.concatenate([lh, rh, pose])
        seq.append(frame_vector)

    seq = np.array(seq, dtype=np.float32)

    # Interpolate/pad to fixed MAX_FRAMES (32)
    n_frames = len(seq)
    if n_frames == MAX_FRAMES:
        return seq
    elif n_frames > MAX_FRAMES:
        idx = np.linspace(0, n_frames - 1, MAX_FRAMES, dtype=int)
        return seq[idx]
    else:
        pad_size = MAX_FRAMES - n_frames
        return np.pad(seq, ((0, pad_size), (0, 0)), mode='constant')

# 4. Batch Generator
def data_generator(paths, labels, batch_size=32):
    n = len(paths)
    while True:
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_x = np.array([process_parquet(paths[j]) for j in batch_idx])
            batch_y = tf.keras.utils.to_categorical([labels[j] for j in batch_idx], num_classes=num_classes)
            yield batch_x, batch_y

# 5. Bi-LSTM Architecture
model = models.Sequential([
    layers.Input(shape=(MAX_FRAMES, NUM_FEATURES)),
    layers.Masking(mask_value=0.0),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')]
)

model.summary()

# 6. Callbacks
ckpt_path = os.path.join(CHECKPOINT_DIR, "isl_model_best.keras")
checkpoint = callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Save id-to-word map for backend inference
id_to_word = {int(v): k for k, v in word_to_id.items()}
with open(os.path.join(CHECKPOINT_DIR, "class_map.json"), "w", encoding="utf-8") as f:
    json.dump(id_to_word, f, indent=2)

batch_size = 32
train_steps = max(1, len(X_train) // batch_size)
val_steps = max(1, len(X_val) // batch_size)

print("\nStarting model training...")
history = model.fit(
    data_generator(X_train, y_train, batch_size),
    steps_per_epoch=train_steps,
    validation_data=data_generator(X_val, y_val, batch_size),
    validation_steps=val_steps,
    epochs=40,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Export final model
final_path = os.path.join(CHECKPOINT_DIR, "isl_model_final.keras")
model.save(final_path)
print(f"\nModel training finished successfully!")
print(f"Best checkpoint saved to: {ckpt_path}")
print(f"Final model saved to: {final_path}")