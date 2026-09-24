"""
stream_processor.py: Rolling window state machine for live streaming inference.
Designed for direct integration with FastAPI/Flask WebSockets.
"""

import json
import numpy as np
import keras
from model.src.extract_landmarks import normalize_frame_coordinates
from nlp_engine import synthesize_sentence

MODEL_PATH = "model/checkpoints/isl_model_best.keras"
CLASS_MAP_PATH = "model/checkpoints/class_map.json"


class LiveSignSession:
    def __init__(self, target_frames: int = 32, stride: int = 8, conf_threshold: float = 0.65):
        self.target_frames = target_frames
        self.stride = stride
        self.conf_threshold = conf_threshold
        
        self.model = keras.models.load_model(MODEL_PATH, compile=False)
        raw_map = json.load(open(CLASS_MAP_PATH))
        self.class_map = {int(k): v for k, v in raw_map.items()}
        
        self.buffer = []
        self.frame_counter = 0
        self.detected_glosses = []
        self.last_predicted_word = None

    def process_landmark_frame(self, raw_225_vec: list[float]) -> dict:
        """
        Receives a single 225-element landmark vector from the current frame.
        Returns a dict with detection status and current state.
        """
        norm_vec = normalize_frame_coordinates(np.array(raw_225_vec, dtype=np.float32))
        self.buffer.append(norm_vec)
        self.frame_counter += 1

        if len(self.buffer) > self.target_frames:
            self.buffer.pop(0)

        result = {
            "status": "buffering",
            "current_word": None,
            "confidence": 0.0,
            "gloss_history": list(self.detected_glosses)
        }

        # Run inference once buffer is full, every `stride` frames
        if len(self.buffer) == self.target_frames and (self.frame_counter % self.stride == 0):
            seq = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
            probs = self.model.predict(seq, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            conf = float(probs[top_idx])

            if conf >= self.conf_threshold:
                word = self.class_map.get(top_idx, "")
                result["current_word"] = word
                result["confidence"] = round(conf * 100, 2)
                result["status"] = "detected"

                # Debounce: only record if it's a new gesture or separated by a pause
                if not self.detected_glosses or self.detected_glosses[-1] != word:
                    self.detected_glosses.append(word)
                    result["gloss_history"] = list(self.detected_glosses)
            else:
                result["status"] = "idle"

        return result

    def finalize_sentence(self) -> str:
        """Converts accumulated glosses into a grammatically coherent sentence."""
        sentence = synthesize_sentence(self.detected_glosses)
        return sentence

    def reset(self):
        """Clears buffer and history."""
        self.buffer.clear()
        self.detected_glosses.clear()
        self.frame_counter = 0
        self.last_predicted_word = None