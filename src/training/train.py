"""Stage A training pipeline: ModernBERT + LoRA.

Run locally for smoke tests; full training runs on Kaggle T4 on Day 6.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)

ADAPTER_DIR = Path("models/stage_a_adapter")
MERGED_DIR = Path("models/stage_a_merged")


def _compute_class_weights(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Return inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels.astype(int), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = labels.shape[0] / (num_labels * counts)
    return weights.astype(np.float32)  # type: ignore[no-any-return]


def _tokenize_dataset(
    tokenizer: Any,
    texts: list[str],
    labels: list[int],
    max_length: int,
) -> Any:
    """Tokenize texts + labels into a PyTorch Dataset; warns on truncation."""
    import torch
    from torch.utils.data import Dataset

    over_limit = 0
    for t in texts:
        enc = tokenizer(t, truncation=False, add_special_tokens=True)
        if len(enc["input_ids"]) > max_length:
            over_limit += 1
    if over_limit:
        logger.warning(
            "truncating samples exceeding max_length",
            extra={"n_over": over_limit, "max_length": max_length},
        )

    class _DS(Dataset):  # type: ignore[type-arg,misc]
        """Minimal PyTorch Dataset wrapping tokenized inputs and labels."""

        def __init__(self) -> None:
            """Tokenize all texts at construction time."""
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            """Return the number of samples."""
            return int(self.labels.shape[0])

        def __getitem__(self, idx: int) -> dict[str, Any]:
            """Return input_ids, attention_mask and label for index idx."""
            return {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    return _DS()


def _early_stop_check(
    val_loss: float,
    best_val_loss: float,
    bad_epochs: int,
    patience: int,
    epoch: int,
) -> tuple[float, int, bool]:
    """Update early-stopping counters; return (new_best_loss, new_bad_epochs, stop)."""
    if val_loss < best_val_loss:
        return val_loss, 0, False
    bad_epochs += 1
    stop = bad_epochs >= patience
    if stop:
        logger.info("early stopping", extra={"epoch": epoch, "patience": patience})
    return best_val_loss, bad_epochs, stop


def _log_weave_epoch(
    epoch: int, val_loss: float, val_acc: float, val_f1: float
) -> None:
    """Optionally emit epoch metrics to W&B Weave when ENABLE_WEAVE=1."""
    if os.environ.get("ENABLE_WEAVE") != "1":
        return
    try:
        import weave

        weave.log(
            {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("weave log failed", extra={"err": str(exc)})


def _run_train_epoch(
    model: Any,
    train_loader: Any,
    optimizer: Any,
    scheduler: Any,
    loss_fn: Any,
    device: Any,
) -> float:
    """Run one full training epoch; return cumulative loss."""
    import torch
    from torch import nn

    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(out.logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += float(loss.item())
    return epoch_loss


def _run_validation(
    model: Any,
    val_loader: Any,
    loss_fn: Any,
    device: Any,
) -> tuple[float, float, float]:
    """Run validation; return (val_loss, val_acc, val_f1)."""
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    val_loss = 0.0
    correct = 0
    seen = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, enabled=(device.type == "cuda")):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                val_loss += float(loss_fn(out.logits, batch["labels"]).item())
            preds = out.logits.argmax(dim=-1)
            correct += int((preds == batch["labels"]).sum().item())
            seen += int(batch["labels"].shape[0])
            all_preds.extend([int(x) for x in preds.cpu().tolist()])
            all_labels.extend([int(x) for x in batch["labels"].cpu().tolist()])

    val_acc = correct / max(seen, 1)
    val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return val_loss, val_acc, val_f1


def train_stage_a(config: dict[str, Any]) -> None:
    """Train ModernBERT + LoRA Stage A classifier.

    Loads train/val parquet, fits LoRA adapters, logs to MLflow,
    and saves adapter + merged checkpoint.
    """
    import mlflow
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    stage_a = config["model"]["stage_a"]
    tcfg = config["training"]
    model_name = stage_a["model_name"]
    num_labels = int(stage_a.get("num_labels", 3))
    max_length = int(stage_a.get("max_length", 2048))

    data_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    logger.info(
        "loaded splits",
        extra={"n_train": len(train_df), "n_val": len(val_df)},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
    train_ds = _tokenize_dataset(
        tokenizer,
        train_df["text"].tolist(),
        train_df["label"].astype(int).tolist(),
        max_length,
    )
    val_ds = _tokenize_dataset(
        tokenizer,
        val_df["text"].tolist(),
        val_df["label"].astype(int).tolist(),
        max_length,
    )

    attn_impl = stage_a.get("attn_implementation", "sdpa")
    base = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
        model_name, num_labels=num_labels, attn_implementation=attn_impl
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(stage_a["lora_r"]),
        lora_alpha=int(stage_a["lora_alpha"]),
        lora_dropout=float(stage_a.get("lora_dropout", 0.1)),
        target_modules=list(stage_a["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("training device", extra={"device": str(device)})

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    logger.info("gradient checkpointing enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable parameters",
        extra={"trainable": trainable, "total": total, "pct": trainable / total},
    )

    class_weights_np = _compute_class_weights(
        train_df["label"].astype(int).to_numpy(), num_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    batch_size = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg["weight_decay"])
    warmup_ratio = float(tcfg["warmup_ratio"])
    patience = int(tcfg["early_stopping_patience"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "p1"))
    with mlflow.start_run(run_name="stage_a_lora"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "lora_r": stage_a["lora_r"],
                "lora_alpha": stage_a["lora_alpha"],
                "max_length": max_length,
                "n_train": len(train_df),
                "n_val": len(val_df),
                **{f"train_{k}": v for k, v in tcfg.items()},
            }
        )
        mlflow.log_metric("trainable_params", trainable)

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        bad_epochs = 0
        for epoch in range(epochs):
            epoch_loss = _run_train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device
            )
            val_loss, val_acc, val_f1 = _run_validation(
                model, val_loader, loss_fn, device
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            logger.info(
                "epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(len(train_loader), 1),
                    "val_loss": val_loss / max(len(val_loader), 1),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
            )
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            _log_weave_epoch(epoch, val_loss, val_acc, val_f1)

            best_val_loss, bad_epochs, stop = _early_stop_check(
                val_loss, best_val_loss, bad_epochs, patience, epoch
            )
            if stop:
                break

        mlflow.log_metric("best_val_f1", best_val_f1)

        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        logger.info("saved adapter", extra={"path": str(ADAPTER_DIR)})

        merged = model.merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        logger.info("saved merged model", extra={"path": str(MERGED_DIR)})

        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="stage_a_adapter")


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Train Stage A LoRA adapter")
    _parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config.yaml"
    )
    _args = _parser.parse_args()
    train_stage_a(load_config(_args.config))
