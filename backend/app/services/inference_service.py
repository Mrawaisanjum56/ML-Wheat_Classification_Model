import os
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from app.core.config import settings

DEFAULT_LABELS = ["Good", "Average", "Bad"]


class InferenceService:
    def __init__(self):
        self.model = None
        self.labels = DEFAULT_LABELS
        self.input_size = (224, 224)  # change if your model uses another size
        self.load_model()

    def load_model(self):
        if not os.path.exists(settings.MODEL_PATH):
            self.model = None
            return
        self.model = tf.keras.models.load_model(settings.MODEL_PATH)

    def predict(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        x is expected as image array [H,W,C] normalized 0..1
        """
        if self.model is None:
            probs = np.array([0.7, 0.2, 0.1], dtype=float)
            return self._format_output(probs)

        # Ensure expected shape [1,H,W,C]
        if x.ndim == 3:
            sample = np.expand_dims(x, axis=0)
        else:
            sample = x

        preds = self.model.predict(sample, verbose=0)[0]

        # If model output is logits, convert to softmax
        if np.any(preds < 0) or np.any(preds > 1) or not np.isclose(np.sum(preds), 1.0, atol=1e-2):
            exp = np.exp(preds - np.max(preds))
            preds = exp / exp.sum()

        probs = preds.astype(float)
        return self._format_output(probs)

    def _format_output(self, probs: np.ndarray):
        pred_idx = int(np.argmax(probs))
        pred_class = self.labels[pred_idx]
        confidence = float(probs[pred_idx])
        probabilities = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return pred_class, confidence, probabilities


inference_service = InferenceService()