"""
Dataset validation and checksum utilities.

C10: Use built-in open() for file I/O, never Path.open().
"""

import hashlib
import json
import logging
import os
from typing import Any

import pandas as pd

from src.data.schema import DATASET_SCHEMA

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a DataFrame against the canonical Pandera schema.

    Logs a summary of validation results. Raises SchemaError on failure.

    Returns the validated (and coerced) DataFrame.
    """
    logger.info(
        "Validating dataset with %d rows and %d columns", len(df), len(df.columns)
    )
    validated = DATASET_SCHEMA.validate(df, lazy=True)
    logger.info("Schema validation passed: %d rows", len(validated))
    logger.info(
        "Label distribution after validation: %s",
        validated["label"].value_counts().to_dict(),
    )
    assert not validated[
        "is_multiturn"
    ].any(), "is_multiturn must be False for all rows"
    return validated


def compute_checksums(
    df: pd.DataFrame, output_path: str = "data/raw/checksums.json"
) -> dict[str, Any]:
    """
    Compute SHA-256 checksums for data integrity.

    Saves checksums to output_path using built-in open() (C10).
    Returns the checksums dict.
    """
    checksums: dict[str, Any] = {}

    # Checksum of the full serialised content (CSV bytes)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    checksums["full_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()

    # Per-column checksums (sorted values)
    for col in df.columns:
        col_bytes = "|".join(df[col].astype(str).sort_values().values).encode("utf-8")
        checksums[f"col_{col}_sha256"] = hashlib.sha256(col_bytes).hexdigest()

    # Metadata
    checksums["num_rows"] = int(len(df))
    checksums["num_columns"] = int(len(df.columns))
    checksums["label_distribution"] = {
        str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(checksums, fh, indent=2)
    logger.info("Checksums saved to %s", output_path)

    return checksums
