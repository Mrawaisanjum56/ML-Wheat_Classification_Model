import os
import pickle
import numpy as np
from typing import Dict, Tuple

from app.core.config import settings

# Fallback labels if not stored with model
DEFAULT_LABELS = ["Good", "Average", "Bad"]


class InferenceService:
    def __init__(self):
        self.model = None
        self.labels = DEFAULT_LABELS
        self.load_model()

    def load_model(self):
        if not os.path.exists(settings.MODEL_PATH):
            self.model = None
            return

        with open(settings.MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        # Support multiple save styles
        # 1) raw model
        # 2) dict: {"model": ..., "labels": [...]}
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.labels = obj.get("labels", DEFAULT_LABELS)
        else:
            self.model = obj

    def predict(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Works for sklearn-style classifiers with predict_proba.
        """
        if self.model is None:
            # Demo fallback when model file is absent
            probs = np.array([0.70, 0.20, 0.10], dtype=float)
            return self._format_output(probs)

        # If your model expects flattened input:
        # shape -> (1, H*W*C)
        sample = x.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(sample)[0]
        else:
            # fallback if no proba
            pred_idx = int(self.model.predict(sample)[0])
            probs = np.zeros(len(self.labels), dtype=float)
            probs[pred_idx] = 1.0

        return self._format_output(probs)

    def _format_output(self, probs: np.ndarray):
        pred_idx = int(np.argmax(probs))
        pred_class = self.labels[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            self.labels[i]: float(probs[i]) for i in range(min(len(self.labels), len(probs)))
        }

        return pred_class, confidence, probabilities


inference_service = InferenceService()