"""Tests for src/data/pipeline.py — split_and_save + run_pipeline."""

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.pipeline import split_and_save


def _make_df(n: int = 90) -> pd.DataFrame:
    """Minimal DataFrame satisfying schema with balanced labels."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_id": f"id_{i}",
                "text": f"sample text number {i}",
                "label": i % 3,  # 0, 1, 2 in equal thirds
                "source_dataset": "test_source",
                "source_type": "user_input",
                "language": "en",
                "is_multiturn": False,
            }
        )
    return pd.DataFrame(rows)


_CONFIG: dict[str, Any] = {"data": {"train_split": 0.70, "seed": 42}}


def test_split_and_save_returns_three_splits(tmp_path: Any) -> None:
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        splits = split_and_save(_make_df(), _CONFIG)
    assert set(splits.keys()) == {"train", "val", "test"}


def test_split_dataset_respects_ratios(tmp_path: Any) -> None:
    df = _make_df(90)
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        splits = split_and_save(df, _CONFIG)
    total = len(df)
    assert len(splits["train"]) == pytest.approx(total * 0.70, abs=3)
    assert len(splits["val"]) == pytest.approx(total * 0.15, abs=3)
    assert len(splits["test"]) == pytest.approx(total * 0.15, abs=3)


def test_split_dataset_no_overlap(tmp_path: Any) -> None:
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        splits = split_and_save(_make_df(), _CONFIG)
    train_ids = set(splits["train"]["sample_id"])
    val_ids = set(splits["val"]["sample_id"])
    test_ids = set(splits["test"]["sample_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_dataset_covers_all_rows(tmp_path: Any) -> None:
    df = _make_df(90)
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        splits = split_and_save(df, _CONFIG)
    total_in_splits = sum(len(s) for s in splits.values())
    assert total_in_splits == len(df)


def test_split_and_save_writes_parquet_files(tmp_path: Any) -> None:
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        split_and_save(_make_df(), _CONFIG)
    for name in ("train", "val", "test"):
        assert (tmp_path / f"{name}.parquet").exists()


def test_split_preserves_all_labels(tmp_path: Any) -> None:
    with patch("src.data.pipeline.PROCESSED_DIR", str(tmp_path)):
        splits = split_and_save(_make_df(), _CONFIG)
    for name, split_df in splits.items():
        labels = set(split_df["label"].unique())
        assert labels == {0, 1, 2}, f"{name} split missing labels"
