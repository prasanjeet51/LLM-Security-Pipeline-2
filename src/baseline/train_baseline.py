import json
import os
import pickle
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from src.logger import get_logger

_TRAIN_PATH = "data/processed/train.parquet"
_VAL_PATH = "data/processed/val.parquet"
_MODEL_DIR = "models/baseline"
_RESULTS_PATH = "reports/results.json"


def train_baseline(config: dict[str, Any]) -> None:
    logger = get_logger(__name__)

    logger.info("Loading training data", extra={"path": _TRAIN_PATH})
    train_df = pd.read_parquet(_TRAIN_PATH)
    val_df = pd.read_parquet(_VAL_PATH)

    baseline_cfg = config["baseline"]
    max_features: int = int(baseline_cfg["tfidf_max_features"])
    svc_c: float = float(baseline_cfg["svc_C"])

    logger.info("Fitting TF-IDF vectorizer", extra={"max_features": max_features})
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    logger.info("Training LinearSVC with calibration", extra={"C": svc_c})
    base_clf = LinearSVC(C=svc_c, max_iter=2000, random_state=42)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc: float = float(accuracy_score(y_val, y_pred))
    weighted_f1: float = float(f1_score(y_val, y_pred, average="weighted"))
    per_class_recall: np.ndarray = recall_score(
        y_val, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    safe_mask = y_val == 0
    fpr_safe: float = float(
        np.sum(y_pred[safe_mask] != 0) / max(int(np.sum(safe_mask)), 1)
    )

    logger.info(
        "Baseline metrics",
        extra={
            "accuracy": acc,
            "weighted_f1": weighted_f1,
            "safe_recall": float(per_class_recall[0]),
            "jailbreak_recall": float(per_class_recall[1]),
            "indirect_recall": float(per_class_recall[2]),
            "false_positive_rate_safe": fpr_safe,
        },
    )

    # Latency benchmark on up to 200 val samples
    sample_texts: list[str] = val_df["text"].head(200).tolist()
    latencies: list[float] = []
    for text in sample_texts:
        t0 = time.perf_counter()
        vec = vectorizer.transform([text])
        clf.predict(vec)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_p50: float = float(np.percentile(latencies, 50))
    lat_p95: float = float(np.percentile(latencies, 95))

    # Save model artefacts
    os.makedirs(_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(_MODEL_DIR, "tfidf_svc.pkl")
    vec_path = os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info("Models saved", extra={"model_path": model_path, "vec_path": vec_path})

    # MLflow logging
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    exp_name: str = str(mlflow_cfg.get("experiment_name", "p1-hybrid-jailbreak"))
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="baseline-tfidf-svc"):
        mlflow.log_params({"tfidf_max_features": max_features, "svc_C": svc_c})
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "weighted_f1": weighted_f1,
                "safe_recall": float(per_class_recall[0]),
                "jailbreak_recall": float(per_class_recall[1]),
                "indirect_recall": float(per_class_recall[2]),
                "false_positive_rate_safe": fpr_safe,
                "latency_p50_ms": lat_p50,
                "latency_p95_ms": lat_p95,
            }
        )

    # Update reports/results.json
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    benchmark_row: dict[str, str | float] = {
        "model": "TF-IDF + LinearSVC (baseline)",
        "accuracy": round(acc, 4),
        "weighted_f1": round(weighted_f1, 4),
        "jailbreak_recall": round(float(per_class_recall[1]), 4),
        "indirect_recall": round(float(per_class_recall[2]), 4),
        "false_positive_rate_safe": round(fpr_safe, 4),
        "latency_p50_ms": round(lat_p50, 2),
        "latency_p95_ms": round(lat_p95, 2),
        "date": "2026-04-12",
    }

    if os.path.exists(_RESULTS_PATH):
        with open(_RESULTS_PATH) as f:
            results: dict[str, Any] = json.load(f)
        results["benchmarks"] = [
            b
            for b in results["benchmarks"]
            if b.get("model") != "TF-IDF + LinearSVC (baseline)"
        ]
        results["benchmarks"].append(benchmark_row)
    else:
        results = {
            "project": "P1-Hybrid-Jailbreak-Detector",
            "benchmarks": [benchmark_row],
        }

    with open(_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written", extra={"path": _RESULTS_PATH})
    print(f"Accuracy: {acc:.4f}  Weighted F1: {weighted_f1:.4f}")
    print(
        classification_report(
            y_val, y_pred, target_names=["safe", "jailbreak", "indirect_injection"]
        )
    )


if __name__ == "__main__":
    from src.config import load_config

    train_baseline(load_config())
