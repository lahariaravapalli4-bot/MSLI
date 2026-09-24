import os
import sys
import glob
import json
import numpy as np
import keras
from model.train import parse_parquet

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"
DATASET_DIR = "model/data/dataset_263/keypoints"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"Error: Class map not found at {CLASS_MAP_PATH}")
        sys.exit(1)

    print("Loading Bi-GRU model and class vocabulary...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    raw_map = json.load(open(CLASS_MAP_PATH))
    class_map = {int(k): v for k, v in raw_map.items()}
    vocab = {v.lower(): int(k) for k, v in class_map.items()}
    print(f"Ready: Loaded {len(class_map)} trained sign classes.\n")
    return model, class_map, vocab

def evaluate_word(target_word: str, model, class_map, vocab):
    query = target_word.strip().lower()
    
    # 1. Locate matching parquet files in dataset_263
    matches = glob.glob(f"{DATASET_DIR}/*{query}*/*.parquet") + \
              glob.glob(f"{DATASET_DIR}/*/*{query}*/*.parquet")

    if not matches:
        # Fallback: check if the exact word exists in vocab and search again
        if query in vocab:
            actual_name = class_map[vocab[query]]
            matches = glob.glob(f"{DATASET_DIR}/*{actual_name}*/*.parquet") + \
                      glob.glob(f"{DATASET_DIR}/*/*{actual_name}*/*.parquet")
    
    if not matches:
        print(f"[!] No sample files found for '{target_word}'.")
        similar = [w for w in vocab.keys() if query in w][:5]
        if similar:
            print(f"    Did you mean one of these? {similar}")
        return

    # Pick the latest sample to avoid bias
    sample_file = matches[-1]
    folder_name = os.path.basename(os.path.dirname(sample_file))

    # 2. Extract landmark sequence
    seq = np.expand_dims(parse_parquet(sample_file), axis=0)

    # 3. Model forward pass
    probs = model.predict(seq, verbose=0)[0]
    top5_idx = np.argsort(probs)[-5:][::-1]

    # 4. Display clean, professional results
    print("=" * 55)
    print(f" Evaluated Sign   : {query.upper()}")
    print(f" Dataset Folder   : {folder_name}")
    print(f" Parquet Sample   : {os.path.basename(sample_file)}")
    print("-" * 55)
    print(" Rank | Predicted Sign       | Confidence")
    print("-" * 55)
    for rank, idx in enumerate(top5_idx):
        sign_label = class_map.get(idx, f"ID_{idx}")
        confidence = probs[idx] * 100
        bar = "#" * int(confidence // 5)
        print(f"  #{rank+1}  | {sign_label:<20} | {confidence:6.2f}%  {bar}")
    print("=" * 55 + "\n")

def main():
    model, class_map, vocab = load_resources()
    print("Type any word from the 262 classes (or type 'quit' / 'exit' to stop).\n")

    while True:
        try:
            user_input = input("Enter sign to test >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting evaluator. Good luck on your review!")
                break
            evaluate_word(user_input, model, class_map, vocab)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()