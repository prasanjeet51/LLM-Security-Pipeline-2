"""Stage A model manifest: save, load, and build JSON metadata."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = Path("models/manifests/stage_a_manifest.json")

_REQUIRED_KEYS = (
    "model_name",
    "lora_r",
    "lora_alpha",
    "target_modules",
    "max_length",
    "training_samples",
    "epochs_completed",
    "best_val_f1",
    "training_device",
    "onnx_exported",
    "onnx_path",
    "onnx_speedup_ratio",
    "date",
)


def save_manifest(
    manifest: dict[str, Any],
    path: str | None = None,
) -> None:
    """Write manifest dict to JSON file, creating parent dirs as needed."""
    out = Path(path) if path else MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest_saved", extra={"path": str(out)})


def load_manifest(path: str | None = None) -> dict[str, Any]:
    """Read and return manifest dict from JSON file."""
    src = Path(path) if path else MANIFEST_PATH
    with open(src) as f:
        data: dict[str, Any] = json.load(f)
    return data


def build_manifest(
    config: dict[str, Any],
    training_samples: int,
    epochs_completed: int,
    best_val_f1: float,
    training_device: str = "T4",
    onnx_exported: bool = False,
    onnx_speedup_ratio: float = 1.0,
) -> dict[str, Any]:
    """Build a manifest dict from training results and config.

    All fields match the spec in TASK 5 of the Day 6 brief.
    """
    stage_a = config.get("model", {}).get("stage_a", {})
    onnx_path = config.get("onnx", {}).get("export_path", "models/stage_a_onnx/")
    return {
        "model_name": str(stage_a.get("model_name", "answerdotai/ModernBERT-base")),
        "lora_r": int(stage_a.get("lora_r", 16)),
        "lora_alpha": int(stage_a.get("lora_alpha", 32)),
        "target_modules": list(stage_a.get("target_modules", ["q_proj", "v_proj"])),
        "max_length": int(stage_a.get("max_length", 2048)),
        "training_samples": training_samples,
        "epochs_completed": epochs_completed,
        "best_val_f1": float(best_val_f1),
        "training_device": training_device,
        "onnx_exported": onnx_exported,
        "onnx_path": onnx_path,
        "onnx_speedup_ratio": float(onnx_speedup_ratio),
        "date": str(date.today()),
    }
