"""Tests for ONNX export, benchmark, and prediction verification.

All heavy ML dependencies (optimum, transformers, torch) are mocked so
these tests run in CI without GPU access. C40 fallback paths are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch  # C25: module-level import; used in helper tests below

from src.training.export_onnx import (
    _run_onnx_inference,
    _run_pytorch_inference,
    benchmark_onnx,
    export_to_onnx,
    verify_onnx_predictions,
)

_CFG: dict[str, Any] = {
    "model": {
        "stage_a": {
            "model_name": "answerdotai/ModernBERT-base",
            "max_length": 128,
        }
    },
    "onnx": {"export_path": "models/stage_a_onnx/"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimum_mock() -> tuple[MagicMock, MagicMock]:
    """Return (optimum_module_mock, ort_instance_mock)."""
    ort_instance = MagicMock()
    ort_cls = MagicMock()
    ort_cls.from_pretrained.return_value = ort_instance
    mod = MagicMock()
    mod.ORTModelForSequenceClassification = ort_cls
    return mod, ort_instance


def _make_tokenizer_mock() -> MagicMock:
    """Return a tokenizer mock that produces valid pt and np tensors."""
    tokenizer = MagicMock()

    def _tok(
        texts: object, return_tensors: str = "pt", **kwargs: object
    ) -> dict[str, Any]:
        n = len(texts) if isinstance(texts, list) else 1  # type: ignore[arg-type]
        if return_tensors == "pt":
            return {
                "input_ids": torch.zeros(n, 8, dtype=torch.long),
                "attention_mask": torch.ones(n, 8, dtype=torch.long),
            }
        return {
            "input_ids": np.zeros((n, 8), dtype=np.int64),
            "attention_mask": np.ones((n, 8), dtype=np.int64),
        }

    tokenizer.side_effect = _tok
    return tokenizer


# ---------------------------------------------------------------------------
# export_to_onnx tests
# ---------------------------------------------------------------------------


def test_export_to_onnx_creates_output_directory(tmp_path: Path) -> None:
    """export_to_onnx creates the output directory before exporting."""
    out = tmp_path / "onnx_out"
    opt_mod, _ = _make_optimum_mock()

    with (
        patch.dict(
            sys.modules, {"optimum": MagicMock(), "optimum.onnxruntime": opt_mod}
        ),
        patch(
            "src.training.export_onnx.benchmark_onnx",
            return_value={
                "speedup_ratio": 2.5,
                "pytorch_ms_per_sample": 10.0,
                "onnx_ms_per_sample": 4.0,
            },
        ),
        patch("src.training.export_onnx.verify_onnx_predictions", return_value=True),
        patch("mlflow.log_metrics"),
    ):
        export_to_onnx("models/merged", str(out), _CFG)

    assert out.exists()


def test_export_to_onnx_calls_save_pretrained(tmp_path: Path) -> None:
    """export_to_onnx calls ort_model.save_pretrained with the output path."""
    out = tmp_path / "onnx_save"
    opt_mod, ort_inst = _make_optimum_mock()

    with (
        patch.dict(
            sys.modules, {"optimum": MagicMock(), "optimum.onnxruntime": opt_mod}
        ),
        patch(
            "src.training.export_onnx.benchmark_onnx",
            return_value={
                "speedup_ratio": 3.0,
                "pytorch_ms_per_sample": 12.0,
                "onnx_ms_per_sample": 4.0,
            },
        ),
        patch("src.training.export_onnx.verify_onnx_predictions", return_value=True),
        patch("mlflow.log_metrics"),
    ):
        export_to_onnx("models/merged", str(out), _CFG)

    ort_inst.save_pretrained.assert_called_once_with(str(out))


def test_export_to_onnx_graceful_on_missing_optimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C40: export_to_onnx does not raise when optimum is not installed."""
    monkeypatch.setitem(sys.modules, "optimum", None)  # type: ignore[arg-type]
    monkeypatch.setitem(  # type: ignore[arg-type]
        sys.modules, "optimum.onnxruntime", None
    )
    out = tmp_path / "onnx_fallback"
    # Must not raise
    export_to_onnx("models/merged", str(out), _CFG)


# ---------------------------------------------------------------------------
# benchmark_onnx tests
# ---------------------------------------------------------------------------


def test_benchmark_onnx_graceful_on_missing_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C40: benchmark_onnx returns default dict when optimum is missing."""
    monkeypatch.setitem(sys.modules, "optimum", None)  # type: ignore[arg-type]
    monkeypatch.setitem(  # type: ignore[arg-type]
        sys.modules, "optimum.onnxruntime", None
    )

    result = benchmark_onnx("models/merged", "models/onnx", _CFG, n_samples=5)

    assert "speedup_ratio" in result
    assert result["speedup_ratio"] == 1.0
    assert "pytorch_ms_per_sample" in result
    assert "onnx_ms_per_sample" in result


# ---------------------------------------------------------------------------
# verify_onnx_predictions tests
# ---------------------------------------------------------------------------


def test_verify_onnx_returns_false_on_missing_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C40: verify_onnx_predictions returns False (not raises) when optimum missing."""
    monkeypatch.setitem(sys.modules, "optimum", None)  # type: ignore[arg-type]
    monkeypatch.setitem(  # type: ignore[arg-type]
        sys.modules, "optimum.onnxruntime", None
    )

    result = verify_onnx_predictions("models/merged", "models/onnx", _CFG)

    assert result is False


# ---------------------------------------------------------------------------
# Internal helper tolerance tests (C40: 1e-4 tolerance check)
# ---------------------------------------------------------------------------


def test_run_pytorch_onnx_helpers_tolerance_check() -> None:
    """_run_pytorch_inference and _run_onnx_inference match within 1e-4 (C40)."""
    n_texts = 5
    fake_logits = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
            [0.05, 0.9, 0.05],
            [0.6, 0.2, 0.2],
        ],
        dtype=np.float32,
    )

    pt_model = MagicMock()
    pt_out = MagicMock()
    pt_out.logits = torch.tensor(fake_logits)
    pt_model.return_value = pt_out

    ort_model = MagicMock()
    ort_out = MagicMock()
    ort_out.logits = fake_logits
    ort_model.return_value = ort_out

    tokenizer = _make_tokenizer_mock()
    texts = ["test"] * n_texts

    pt_arr, pt_elapsed = _run_pytorch_inference(pt_model, tokenizer, texts, 32)
    onnx_arr, onnx_elapsed = _run_onnx_inference(ort_model, tokenizer, texts, 32)

    assert pt_arr.shape == onnx_arr.shape
    assert float(np.max(np.abs(pt_arr - onnx_arr))) <= 1e-4
    assert pt_elapsed >= 0.0
    assert onnx_elapsed >= 0.0
