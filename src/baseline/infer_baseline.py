import os
import pickle
from typing import Any

import numpy as np

from src.logger import get_logger


class BaselineClassifier:
    def __init__(self, model_dir: str = "models/baseline") -> None:
        logger = get_logger(__name__)
        model_path = os.path.join(model_dir, "tfidf_svc.pkl")
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)  # nosec B301
        with open(vec_path, "rb") as f:
            self._vectorizer = pickle.load(f)  # nosec B301

        logger.info("BaselineClassifier loaded", extra={"model_dir": model_dir})

    def classify(self, text: str) -> dict[str, Any]:
        vec = self._vectorizer.transform([text])
        label: int = int(self._clf.predict(vec)[0])
        proba: np.ndarray = self._clf.predict_proba(vec)[0]
        confidence: float = float(np.max(proba))
        probabilities: dict[str, float] = {
            "safe": float(proba[0]),
            "jailbreak": float(proba[1]),
            "indirect_injection": float(proba[2]),
        }
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }
