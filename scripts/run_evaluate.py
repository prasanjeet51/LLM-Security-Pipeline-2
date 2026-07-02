"""End-to-end evaluation runner for P1 Hybrid Jailbreak Detector.

Runs the full evaluation pipeline:
  1. evaluate_model()  — accuracy / F1 / recall / FPR / confusion matrix
  2. generate_calibration_plot() — reliability diagram + ECE / MCE
  3. generate_latency_chart() — Stage A / A+B / ONNX latency bar chart
  4. compare_baseline_vs_hybrid() — side-by-side metrics + results.json
  5. add_vendor_notes() — qualitative Azure / AWS / Meta / P1 comparison

Usage:
    py -3.10 scripts/run_evaluate.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 evaluation pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import load_config
    from src.evaluation.evaluate import evaluate

    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
