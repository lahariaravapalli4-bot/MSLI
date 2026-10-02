"""
demo_showcase.py: Interactive Evaluation and Viva Showcase
Allows testing any word or sentence sequence on demand.
"""

import os
import glob
import json
import numpy as np
import keras
from model.train import parse_parquet
from nlp_engine import synthesize_sentence, speak_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"

print("\n" + "=" * 65)
print("   INDIAN SIGN LANGUAGE TO SPEECH: INTERACTIVE EVALUATION")
print("=" * 65)

model = keras.models.load_model(MODEL_PATH, compile=False)
raw_map = json.load(open(CLASS_MAP_PATH))
class_map = {int(k): v.lower() for k, v in raw_map.items()}

all_files = glob.glob("model/data/dataset_263/**/*.parquet", recursive=True)
available_words = sorted(list(set(os.path.normpath(f).split(os.sep)[-2].lower() for f in all_files)))

print(f"[INFO] System ready. Found {len(available_words)} testable sign classes.")
print("[TIP] Suggested test words: i, alive, sad, car, shirt, tuesday, dream")
print("=" * 65 + "\n")

while True:
    try:
        user_input = input("Enter sign word(s) separated by spaces (or 'q' to quit): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        break
    if user_input in ("q", "quit", "exit"):
        break
    if not user_input:
        continue

    query_words = user_input.split()
    detected_glosses = []

    for word in query_words:
        matches = glob.glob(f"model/data/dataset_263/**/{word}/*.parquet", recursive=True)
        if not matches:
            print(f"[-] Sign '{word}' not found in local dataset sample.")
            continue

        sample_file = matches[0]
        features = parse_parquet(sample_file)
        batch_in = np.expand_dims(features, axis=0)

        probs = model.predict(batch_in, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        pred_word = class_map[top_idx]
        conf = probs[top_idx] * 100

        print(f" -> Input: '{word.upper():<10}' | Bi-GRU Recognized: '{pred_word.upper():<10}' (Confidence: {conf:.2f}%)")
        detected_glosses.append(pred_word)

    if detected_glosses:
        final_sentence = synthesize_sentence(detected_glosses)
        print(f"\n[NLP Translation]: \"{final_sentence}\"")
        speak_sentence(final_sentence)
        print("-" * 65 + "\n")

print("\nExiting Showcase System.")