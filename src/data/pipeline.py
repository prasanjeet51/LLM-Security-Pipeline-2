"""
Data pipeline: split, save parquet files, and build FAISS attack index.

C10: Use built-in open() for file I/O, never Path.open().
C37: FAISS similarity gate — index built here, queried at inference.
"""

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
FAISS_DIR = "data/faiss_index"


def split_and_save(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Stratified train/val/test split (70/15/15) saved as parquet files.

    Logs label distribution per split and samples per source.
    """
    data_cfg = config.get("data", {})
    train_frac: float = float(data_cfg.get("train_split", 0.70))
    seed: int = int(data_cfg.get("seed", 42))

    # val and test each take 50% of the non-train set (0.15 / 0.30 = 0.50)
    other_frac = 1.0 - train_frac

    train_df, temp_df = train_test_split(
        df, test_size=other_frac, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, split_df in splits.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        split_df.reset_index(drop=True).to_parquet(path, index=False)
        dist = split_df["label"].value_counts().to_dict()
        logger.info(
            "Split %-5s | rows: %5d | label dist: %s",
            name,
            len(split_df),
            dist,
        )

    logger.info("Total rows: %d", len(df))
    logger.info(
        "Samples per source:\n%s",
        df["source_dataset"].value_counts().to_string(),
    )

    return splits


def build_faiss_index(train_df: pd.DataFrame) -> None:
    """
    Build a FAISS IndexFlatIP (cosine similarity via normalised inner product)
    from all attack texts (label=1 and label=2) in the training set.

    Saves:
      data/faiss_index/attack_index.faiss
      data/faiss_index/attack_texts.json

    C37: This index is queried at inference by the similarity gate.
    """
    import faiss  # type: ignore[import-untyped]
    from sentence_transformers import (
        SentenceTransformer,  # type: ignore[import-untyped]
    )

    os.makedirs(FAISS_DIR, exist_ok=True)

    attack_df = train_df[train_df["label"].isin([1, 2])].copy()
    texts: list[str] = attack_df["text"].tolist()

    if not texts:
        logger.warning("No attack texts found in train split — skipping FAISS build")
        return

    logger.info("Building FAISS index from %d attack texts", len(texts))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise for cosine via inner product
        convert_to_numpy=True,
    )

    dim: int = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_path = os.path.join(FAISS_DIR, "attack_index.faiss")
    texts_path = os.path.join(FAISS_DIR, "attack_texts.json")

    faiss.write_index(index, index_path)
    logger.info(
        "FAISS index saved: path=%s | size=%d | dim=%d",
        index_path,
        index.ntotal,
        dim,
    )

    with open(texts_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "texts": texts,
                "labels": attack_df["label"].tolist(),
                "sources": attack_df["source_dataset"].tolist(),
            },
            fh,
            indent=2,
        )
    logger.info(
        "Attack texts mapping saved: path=%s | entries=%d", texts_path, len(texts)
    )


def run_pipeline(config: dict[str, Any]) -> None:
    """
    Full data pipeline:
      1. Load merged DataFrame (expects caller to pass it via config or re-collect)
      2. Validate
      3. Split and save
      4. Build FAISS index

    This is called from scripts/run_data_pipeline.py.
    """
    from src.data.collect import collect_all_sources
    from src.data.validate import compute_checksums, validate_dataset

    logger.info("=== Data Pipeline Start ===")

    df = collect_all_sources(config)
    df = validate_dataset(df)
    compute_checksums(df)

    splits = split_and_save(df, config)
    build_faiss_index(splits["train"])

    logger.info("=== Data Pipeline Complete ===")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build processed splits + FAISS index")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    _args = parser.parse_args()

    from src.config import load_config

    run_pipeline(load_config(_args.config))
