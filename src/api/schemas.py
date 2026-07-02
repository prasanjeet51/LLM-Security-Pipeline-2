"""Pydantic v2 request/response schemas for the classify and feedback API."""

from typing import Optional

from pydantic import BaseModel, field_validator

_VALID_SOURCE_TYPES = {"user_input", "external_doc", "api_call", "system_prompt"}
_VALID_FEEDBACK_TYPES = {"false_positive", "false_negative", "label_correction"}
_VALID_DECISIONS = {"allow", "block", "human_review"}
_VALID_LABELS = {"safe", "jailbreak", "indirect_injection"}


class ClassifyRequest(BaseModel):
    """Input payload for the /classify endpoint."""

    user_prompt: str
    external_context: Optional[str] = None
    source_type: str = "user_input"
    conversation_history: Optional[list[str]] = None

    @field_validator("user_prompt")
    @classmethod
    def user_prompt_must_not_be_empty(cls, v: str) -> str:
        """Reject blank prompts."""
        if not v.strip():
            raise ValueError("user_prompt must not be empty")
        return v

    @field_validator("source_type")
    @classmethod
    def source_type_must_be_valid(cls, v: str) -> str:
        """Enforce the closed source_type enumeration."""
        if v not in _VALID_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {_VALID_SOURCE_TYPES}")
        return v


class ClassifyResponse(BaseModel):
    """Output payload from the /classify endpoint."""

    label: str
    risk_scores: dict[str, float]
    decision: str
    confidence: float
    reason_tags: list[str]
    attack_type: Optional[str] = None
    stage_used: str
    similarity_score: Optional[float] = None
    perplexity_score: Optional[float] = None
    token_attributions: Optional[list[dict[str, str | float]]] = None


class BatchClassifyRequest(BaseModel):
    """Input payload for the /classify_batch endpoint."""

    requests: list[ClassifyRequest]

    @field_validator("requests")
    @classmethod
    def requests_must_not_be_empty(
        cls, v: list[ClassifyRequest]
    ) -> list[ClassifyRequest]:
        """Reject empty batch requests."""
        if not v:
            raise ValueError("requests must not be empty")
        return v


class BatchClassifyResponse(BaseModel):
    """Output payload for the /classify_batch endpoint."""

    responses: list[ClassifyResponse]


class HealthResponse(BaseModel):
    """Output payload for the /health endpoint."""

    status: str
    model_loaded: bool
    pipeline_ready: bool
    uptime_seconds: float
    memory_mb: float


class FeedbackRequest(BaseModel):
    """Input payload for the /feedback endpoint."""

    user_prompt: str
    original_label: str
    corrected_label: str
    original_decision: str
    original_confidence: float
    feedback_type: str

    @field_validator("original_label", "corrected_label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        """Enforce label enumeration."""
        if v not in _VALID_LABELS:
            raise ValueError(f"label must be one of {_VALID_LABELS}")
        return v

    @field_validator("original_decision")
    @classmethod
    def decision_must_be_valid(cls, v: str) -> str:
        """Enforce decision enumeration."""
        if v not in _VALID_DECISIONS:
            raise ValueError(f"decision must be one of {_VALID_DECISIONS}")
        return v

    @field_validator("feedback_type")
    @classmethod
    def feedback_type_must_be_valid(cls, v: str) -> str:
        """Enforce feedback_type enumeration."""
        if v not in _VALID_FEEDBACK_TYPES:
            raise ValueError(f"feedback_type must be one of {_VALID_FEEDBACK_TYPES}")
        return v

    @field_validator("original_confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        """Reject confidence values outside [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("original_confidence must be between 0 and 1")
        return v
