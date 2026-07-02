"""Export the locally merged Stage A model to ONNX.

Run AFTER merge_adapter.py has saved models/stage_a_merged/:
    py -3.10 scripts/export_onnx_local.py

Requires: pip install optimum[onnxruntime]
C40: falls back gracefully if optimum is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MERGED_PATH = "models/stage_a_merged"
_ONNX_PATH = "models/stage_a_onnx"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import load_config
    from src.training.export_onnx import export_to_onnx, verify_onnx_predictions

    cfg = load_config()

    merged = Path(_MERGED_PATH)
    if not merged.exists():
        print(
            f"ERROR: {_MERGED_PATH} not found. Run scripts/merge_adapter.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Exporting {_MERGED_PATH} -> {_ONNX_PATH} ...")
    export_to_onnx(_MERGED_PATH, _ONNX_PATH, cfg)
    print("Export complete.")

    print("Verifying ONNX predictions against PyTorch (tolerance 1e-4) …")
    ok = verify_onnx_predictions(_MERGED_PATH, _ONNX_PATH, cfg)
    if ok:
        print("ONNX verification PASSED.")
    else:
        print(
            "ONNX verification FAILED or skipped (optimum not installed). "
            "Stage A will fall back to PyTorch at inference time (C40).",
            file=sys.stderr,
        )
