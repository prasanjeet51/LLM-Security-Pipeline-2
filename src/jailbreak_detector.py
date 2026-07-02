"""
One-liner public API for the hybrid jailbreak + prompt injection detector.

Usage:
    from jailbreak_detector import detect, DetectionResult
    result = detect("user prompt here")
    if result.blocked:
        raise ValueError(result.reason)
"""

import os
from dataclasses import dataclass
from typing import Optional

_pipeline = None


@dataclass
class DetectionResult:
    """Structured result from the detect() convenience function."""

    blocked: bool
    decision: str
    label: str
    confidence: float
    reason: Optional[str]
    attack_type: Optional[str]
    stage_used: str


def _get_pipeline():  # type: ignore[return]
    global _pipeline
    if _pipeline is None:
        from src.config import load_config
        from src.hybrid.pipeline import HybridPipeline

        config_path = os.environ.get(
            "JAILBREAK_DETECTOR_CONFIG",
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
        )
        _pipeline = HybridPipeline(load_config(config_path))
    return _pipeline


def detect(
    prompt: str,
    context: Optional[str] = None,
    history: Optional[list] = None,
) -> DetectionResult:
    """Classify a user prompt for jailbreak or prompt injection.

    Args:
        prompt: The user message to classify.
        context: Optional retrieved document context (RAG use case).
        history: Optional list of previous messages in the conversation.

    Returns:
        DetectionResult with blocked=True if the prompt should be blocked.
    """
    from src.api.schemas import ClassifyRequest

    pipeline = _get_pipeline()
    request = ClassifyRequest(
        user_prompt=prompt,
        external_context=context,
        conversation_history=history,
    )
    response = pipeline.classify(request)
    reason = ", ".join(response.reason_tags) if response.reason_tags else None
    return DetectionResult(
        blocked=response.decision == "block",
        decision=response.decision,
        label=response.label,
        confidence=response.confidence,
        reason=reason,
        attack_type=response.attack_type,
        stage_used=response.stage_used,
    )
