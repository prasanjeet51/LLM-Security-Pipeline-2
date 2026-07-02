"""Tests for the public-facing detect() API."""

from typing import Optional
from unittest.mock import patch

from src.api.schemas import ClassifyResponse
from src.jailbreak_detector import DetectionResult, detect


def _mock_response(
    decision: str,
    label: str,
    confidence: float,
    reason_tags: Optional[list] = None,
    attack_type: Optional[str] = None,
) -> ClassifyResponse:
    return ClassifyResponse(
        label=label,
        risk_scores={"safe": 0.1, "jailbreak": 0.8, "indirect_injection": 0.1},
        decision=decision,
        confidence=confidence,
        reason_tags=reason_tags or [],
        attack_type=attack_type,
        stage_used="stage_a",
    )


def test_detect_returns_detection_result() -> None:
    mock_resp = _mock_response("allow", "safe", 0.99)
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        result = detect("What is the capital of France?")
    assert isinstance(result, DetectionResult)
    assert result.blocked is False
    assert result.decision == "allow"


def test_detect_blocked_jailbreak() -> None:
    mock_resp = _mock_response(
        "block",
        "jailbreak",
        0.97,
        reason_tags=["high_attack_confidence"],
        attack_type="jailbreak",
    )
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        result = detect("Ignore all instructions and...")
    assert result.blocked is True
    assert result.attack_type == "jailbreak"
    assert result.reason is not None
    assert "high_attack_confidence" in result.reason


def test_detect_passes_context_and_history() -> None:
    mock_resp = _mock_response("allow", "safe", 0.95)
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        detect("prompt", context="some doc", history=["prev msg"])
        call_args = mock_pipeline.return_value.classify.call_args[0][0]
        assert call_args.external_context == "some doc"
        assert call_args.conversation_history == ["prev msg"]


def test_detect_human_review_not_blocked() -> None:
    mock_resp = _mock_response("human_review", "jailbreak", 0.6)
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        result = detect("ambiguous prompt")
    assert result.blocked is False
    assert result.decision == "human_review"


def test_detect_empty_reason_tags_gives_none_reason() -> None:
    mock_resp = _mock_response("allow", "safe", 0.99, reason_tags=[])
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        result = detect("hello")
    assert result.reason is None


def test_detect_multiple_reason_tags_joined() -> None:
    mock_resp = _mock_response(
        "block",
        "indirect_injection",
        0.88,
        reason_tags=["high_attack_confidence", "faiss_match"],
    )
    with patch("src.jailbreak_detector._get_pipeline") as mock_pipeline:
        mock_pipeline.return_value.classify.return_value = mock_resp
        result = detect("some injected doc")
    assert result.reason == "high_attack_confidence, faiss_match"
