"""Tests for Stage A model manifest save/load/build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.training.manifest import (
    _REQUIRED_KEYS,
    MANIFEST_PATH,
    build_manifest,
    load_manifest,
    save_manifest,
)

_TINY_CONFIG: dict[str, Any] = {
    "model": {
        "stage_a": {
            "model_name": "answerdotai/ModernBERT-base",
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "max_length": 2048,
        }
    },
    "onnx": {"export_path": "models/stage_a_onnx/"},
}


def test_save_manifest_creates_file(tmp_path: Path) -> None:
    """save_manifest writes a JSON file at the given path."""
    out = str(tmp_path / "m" / "manifest.json")
    data: dict[str, Any] = {"model_name": "test", "lora_r": 4}
    save_manifest(data, out)
    assert Path(out).exists()


def test_save_load_manifest_roundtrip(tmp_path: Path) -> None:
    """save_manifest then load_manifest preserves all fields."""
    out = str(tmp_path / "manifest.json")
    original: dict[str, Any] = {
        "model_name": "answerdotai/ModernBERT-base",
        "lora_r": 16,
        "lora_alpha": 32,
        "best_val_f1": 0.92,
        "onnx_exported": True,
    }
    save_manifest(original, out)
    loaded = load_manifest(out)
    assert loaded == original


def test_load_manifest_raises_on_missing_file(tmp_path: Path) -> None:
    """load_manifest raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_manifest(str(tmp_path / "does_not_exist.json"))


def test_build_manifest_has_all_required_keys() -> None:
    """build_manifest returns a dict with every key from _REQUIRED_KEYS."""
    manifest = build_manifest(
        config=_TINY_CONFIG,
        training_samples=5000,
        epochs_completed=3,
        best_val_f1=0.88,
        training_device="T4",
        onnx_exported=True,
        onnx_speedup_ratio=2.7,
    )
    for key in _REQUIRED_KEYS:
        assert key in manifest, f"Missing required key: {key}"


def test_build_manifest_reflects_config() -> None:
    """build_manifest picks up lora_r/alpha/target_modules from config."""
    manifest = build_manifest(
        config=_TINY_CONFIG,
        training_samples=1000,
        epochs_completed=5,
        best_val_f1=0.75,
    )
    assert manifest["lora_r"] == 16
    assert manifest["lora_alpha"] == 32
    assert manifest["target_modules"] == ["q_proj", "v_proj"]
    assert manifest["max_length"] == 2048
    assert manifest["model_name"] == "answerdotai/ModernBERT-base"


def test_build_manifest_date_is_iso_string() -> None:
    """build_manifest date field is a valid ISO date string."""
    manifest = build_manifest(
        config=_TINY_CONFIG,
        training_samples=100,
        epochs_completed=1,
        best_val_f1=0.5,
    )
    # Should parse without error
    from datetime import date

    date.fromisoformat(manifest["date"])


def test_save_manifest_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_manifest with no path argument uses MANIFEST_PATH."""
    target = tmp_path / "manifests" / "stage_a_manifest.json"
    monkeypatch.setattr("src.training.manifest.MANIFEST_PATH", target)
    data: dict[str, Any] = {"test": True}
    save_manifest(data)
    assert target.exists()
    assert json.loads(target.read_text())["test"] is True


def test_manifest_path_constant_name() -> None:
    """MANIFEST_PATH points to stage_a_manifest.json under models/manifests/."""
    assert MANIFEST_PATH.name == "stage_a_manifest.json"
    assert "manifests" in str(MANIFEST_PATH)
