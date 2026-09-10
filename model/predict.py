import os
import json
import numpy as np
import tensorflow as tf
from model.src.extract_landmarks import extract_landmarks_from_video

# Resolve file paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "isl_model_best.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "checkpoints", "class_map.json")

# Global variables to cache loaded model and labels in memory
_MODEL = None
_CLASS_MAP = None


def _load_artifacts():
    """Lazily loads the trained model weights and class mapping once."""
    global _MODEL, _CLASS_MAP
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {MODEL_PATH}. "
                "Ensure trained weights are placed in model/checkpoints/."
            )
        _MODEL = tf.keras.models.load_model(MODEL_PATH)

    if _CLASS_MAP is None:
        if not os.path.exists(CLASS_MAP_PATH):
            raise FileNotFoundError(
                f"Class mapping file not found at: {CLASS_MAP_PATH}. "
                "Ensure class_map.json exists in model/checkpoints/."
            )
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            # Ensure keys are integers (class indices)
            _CLASS_MAP = {int(k): v for k, v in raw_map.items()}


def predict(video_path: str) -> dict:
    """
    Public entrypoint for backend integration.
    
    Args:
        video_path (str): File path to an MP4 video clip.
        
    Returns:
        dict: {"sign": str, "confidence": float}
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    _load_artifacts()

    # 1. Extract (32, 225) landmarks using MediaPipe Holistic
    features = extract_landmarks_from_video(video_path, target_frames=32)

    # 2. Add batch dimension -> shape: (1, 32, 225)
    input_tensor = np.expand_dims(features, axis=0)

    # 3. Model inference
    probabilities = _MODEL.predict(input_tensor, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[best_idx])
    predicted_sign = _CLASS_MAP.get(best_idx, "UNKNOWN")

    return {
        "sign": predicted_sign,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_clip = sys.argv[1]
        print(f"Testing inference on {test_clip}...")
        try:
            res = predict(test_clip)
            print("Result:", res)
        except Exception as e:
            print("Inference error:", e)
    else:
        print("predict.py loaded successfully. Run with a video path argument to test.")