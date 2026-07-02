"""Project-specific exception hierarchy for P1 Hybrid Jailbreak Detector."""


class ProjectBaseError(Exception):
    """Base class for all P1 project exceptions."""


class DataLoadError(ProjectBaseError):
    """Raised when a dataset or parquet file cannot be loaded."""


class ModelNotFoundError(ProjectBaseError):
    """Raised when a required model checkpoint is missing from disk."""


class ClassificationError(ProjectBaseError):
    """Raised when the classification pipeline fails unexpectedly."""


class PolicyViolationError(ProjectBaseError):
    """Raised when the policy gate detects a hard-block violation."""


class SchemaValidationError(ProjectBaseError):
    """Raised when input or output data fails pandera schema validation."""
