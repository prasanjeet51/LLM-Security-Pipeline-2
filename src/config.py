from typing import Any

import yaml

from src.logger import get_logger

_REQUIRED_KEYS = ("model", "training", "data", "api")


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    logger = get_logger(__name__)
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in result]
    if missing:
        raise KeyError(f"config.yaml missing required keys: {missing}")
    logger.info("config loaded", extra={"path": path, "keys": list(result.keys())})
    return result
