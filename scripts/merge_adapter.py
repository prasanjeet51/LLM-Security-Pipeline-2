"""Merge LoRA adapter into base ModernBERT model and save to models/stage_a_merged/.

Run this locally after Kaggle training:
    py -3.10 scripts/merge_adapter.py

The merged model is saved to models/stage_a_merged/ (gitignored, local only).
stage_a.py will load from there automatically on the next classify() call.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ADAPTER_PATH = "models/stage_a_adapter"
_OUTPUT_PATH = "models/stage_a_merged"


def merge_adapter(
    adapter_path: str,
    output_path: str,
    model_name: str,
    num_labels: int = 3,
) -> None:
    """Load base model + adapter, merge, and save to output_path.

    Args:
        adapter_path: Path to the saved LoRA adapter directory.
        output_path: Destination for merged full model.
        model_name: Base model name (e.g. 'answerdotai/ModernBERT-base').
        num_labels: Number of classification labels (default 3).
    """
    from peft import PeftModel
    from transformers import (  # nosec B615
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    adapter = Path(adapter_path)
    out = Path(output_path)

    if not adapter.exists():
        print(f"ERROR: adapter not found at {adapter_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
    base = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
        model_name, num_labels=num_labels
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter))  # nosec B615

    print("Merging adapter into base model (merge_and_unload)…")
    merged = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Merged model saved to: {out}")


if __name__ == "__main__":
    # Allow optional override: python merge_adapter.py <adapter_path> <output_path>
    if len(sys.argv) == 3:
        _adapter_path = sys.argv[1]
        _output_path = sys.argv[2]
    elif len(sys.argv) == 1:
        _adapter_path = _ADAPTER_PATH
        _output_path = _OUTPUT_PATH
    else:
        print("Usage: merge_adapter.py [adapter_path output_path]", file=sys.stderr)
        sys.exit(1)

    # Load model_name from config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import load_config

    _cfg = load_config()
    _model_name: str = str(
        _cfg.get("model", {})
        .get("stage_a", {})
        .get("model_name", "answerdotai/ModernBERT-base")
    )
    _num_labels: int = int(
        _cfg.get("model", {}).get("stage_a", {}).get("num_labels", 3)
    )
    merge_adapter(_adapter_path, _output_path, _model_name, _num_labels)
