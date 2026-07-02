"""Stage A training, calibration, ONNX export, and manifest package.

Exports the LoRA training entrypoint, FPR-constrained threshold calibrator,
ONNX exporter, and manifest save/load used by the pipeline.
"""

from src.training.calibrate import calibrate_thresholds
from src.training.export_onnx import export_to_onnx
from src.training.manifest import save_manifest
from src.training.train import train_stage_a

__all__ = [
    "train_stage_a",
    "calibrate_thresholds",
    "export_to_onnx",
    "save_manifest",
]
