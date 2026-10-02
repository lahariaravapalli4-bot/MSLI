import glob
import json
import os
import numpy as np
import keras
from model.train import parse_parquet

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v for k, v in raw_map.items()}

print(f"Total classes in model: {len(class_map)}")
print("Sample class mappings:", list(class_map.items())[:10])

# Test on 8 actual parquet files across different words
files = glob.glob("model/data/dataset_263/**/*.parquet", recursive=True)
print(f"Found {len(files)} parquet files. Evaluating samples...\n")

correct = 0
total = 0

for f in files[::max(1, len(files)//8)][:8]:
    parts = os.path.normpath(f).split(os.sep)
    true_label = parts[-2]

    feat = parse_parquet(f)
    inp = np.expand_dims(feat, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    pred_label = class_map.get(top_idx, "UNKNOWN")
    conf = probs[top_idx] * 100

    match = "PASS" if true_label.lower() == pred_label.lower() else "FAIL"
    if match == "PASS":
        correct += 1
    total += 1
    print(f"[{match}] True: {true_label:<15} | Pred: {pred_label:<15} ({conf:.1f}%)")

print(f"\nAccuracy on test batch: {correct}/{total}")
