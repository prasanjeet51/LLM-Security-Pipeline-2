"""Calibrate Stage A confidence thresholds for the policy gate.

Produces block_confidence, allow_confidence, and an uncertain_band
that maximises jailbreak recall while holding FPR on safe < 5%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

THRESHOLDS_PATH = Path("models/manifests/thresholds.json")
SAFE_LABEL = 0
JAILBREAK_LABEL = 1


class _ProbaModel(Protocol):
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


def _max_threshold_with_fpr(
    scores: np.ndarray, is_safe: np.ndarray, max_fpr: float
) -> float:
    """Lowest threshold T s.t. FPR(safe flagged) < max_fpr when score >= T."""
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_is_safe = is_safe[order]
    n_safe = max(int(is_safe.sum()), 1)
    fp = 0
    best_t = 1.0
    for i, score in enumerate(sorted_scores):
        if sorted_is_safe[i]:
            fp += 1
            if fp / n_safe >= max_fpr:
                return float(score) + 1e-6
        best_t = float(score)
    return best_t


def calibrate_thresholds(
    model: _ProbaModel, val_data: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Compute Stage A thresholds from validation predictions.

    ``val_data`` must expose ``text`` and ``label`` iterables (DataFrame or
    mapping). Returns a dict with block_confidence, allow_confidence and
    uncertain_band; writes it to models/manifests/thresholds.json.
    """
    try:
        import mlflow

        mlflow_enabled = True
    except Exception:
        mlflow_enabled = False

    texts = list(val_data["text"])
    labels = np.asarray(list(val_data["label"]), dtype=int)
    probs = np.asarray(model.predict_proba(texts), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError("predict_proba returned incompatible shape")

    jailbreak_scores = probs[:, JAILBREAK_LABEL]
    safe_scores = probs[:, SAFE_LABEL]
    is_safe = labels == SAFE_LABEL
    is_jailbreak = labels == JAILBREAK_LABEL

    max_fpr = 0.05
    block_confidence = _max_threshold_with_fpr(jailbreak_scores, is_safe, max_fpr)

    jb_mask = is_jailbreak
    if jb_mask.any():
        allow_confidence = float(
            min(1.0, np.quantile(safe_scores[jb_mask], 0.95) + 1e-6)
        )
        allow_confidence = max(allow_confidence, float(np.median(safe_scores[is_safe])))
    else:
        allow_confidence = float(np.quantile(safe_scores[is_safe], 0.5))

    uncertain_low = min(block_confidence, 0.5)
    uncertain_high = block_confidence

    jb_flagged = jailbreak_scores >= block_confidence
    recall = float((jb_flagged & is_jailbreak).sum() / max(int(is_jailbreak.sum()), 1))
    fpr = float((jb_flagged & is_safe).sum() / max(int(is_safe.sum()), 1))

    thresholds: dict[str, Any] = {
        "block_confidence": float(block_confidence),
        "allow_confidence": float(allow_confidence),
        "uncertain_band": [float(uncertain_low), float(uncertain_high)],
        "metrics": {
            "jailbreak_recall": recall,
            "safe_fpr": fpr,
            "n_val": int(labels.shape[0]),
        },
    }

    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info(
        "thresholds calibrated",
        extra={"path": str(THRESHOLDS_PATH), **thresholds["metrics"]},
    )

    if mlflow_enabled:
        try:
            mlflow.log_metrics(
                {
                    "calibration_block_confidence": thresholds["block_confidence"],
                    "calibration_allow_confidence": thresholds["allow_confidence"],
                    "calibration_jailbreak_recall": recall,
                    "calibration_safe_fpr": fpr,
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("mlflow log skipped", extra={"err": str(exc)})

    return thresholds
