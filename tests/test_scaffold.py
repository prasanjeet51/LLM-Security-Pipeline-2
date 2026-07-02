import logging

from src.config import load_config
from src.exceptions import (
    ClassificationError,
    DataLoadError,
    ModelNotFoundError,
    PolicyViolationError,
    ProjectBaseError,
    SchemaValidationError,
)
from src.logger import get_logger


def test_config_loads():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)


def test_config_has_required_keys():
    config = load_config("config/config.yaml")
    for key in ("model", "training", "data", "api"):
        assert key in config, f"Missing required key: {key}"


def test_logger_returns_logger():
    logger = get_logger("scaffold_test")
    assert isinstance(logger, logging.Logger)


def test_logger_no_duplicate_handlers():
    logger1 = get_logger("dedup_scaffold")
    count = len(logger1.handlers)
    logger2 = get_logger("dedup_scaffold")
    assert len(logger2.handlers) == count


def test_exceptions_hierarchy():
    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(ModelNotFoundError, ProjectBaseError)
    assert issubclass(ClassificationError, ProjectBaseError)
    assert issubclass(PolicyViolationError, ProjectBaseError)
    assert issubclass(SchemaValidationError, ProjectBaseError)
