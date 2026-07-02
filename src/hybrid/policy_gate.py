"""Deterministic policy gate — owns the final decision for every request."""

from typing import Any, Optional

from src.api.schemas import ClassifyRequest, ClassifyResponse
from src.logger import get_logger

_LABEL_MAP: dict[int, str] = {
    0: "safe",
    1: "jailbreak",
    2: "indirect_injection",
}

_OBFUSCATION_TAGS: frozenset[str] = frozenset(
    {"homoglyph_mapped", "zero_width_stripped", "leetspeak_normalized"}
)

_RISKY_SOURCE_TYPES: frozenset[str] = frozenset({"external_doc", "api_call"})


class PolicyGate:
    """5-row deterministic decision table. The model can be wrong, this can't."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise thresholds from config."""
        thresholds: dict[str, Any] = config["evaluation"]["thresholds"]
        self._block_conf: float = float(thresholds["block_confidence"])
        self._allow_conf: float = float(thresholds["allow_confidence"])
        band = thresholds["uncertain_band"]
        self._uncertain_lo: float = float(band[0])
        self._uncertain_hi: float = float(band[1])
        self._logger = get_logger(__name__)

    def decide(
        self,
        stage_a_result: Optional[dict[str, Any]],
        stage_b_result: Optional[dict[str, Any]],
        perplexity_result: Optional[dict[str, Any]],
        similarity_result: Optional[dict[str, Any]],
        request: ClassifyRequest,
        reason_tags_in: list[str],
    ) -> ClassifyResponse:
        """Apply decision table; return the first matching row's response."""
        tags: list[str] = list(reason_tags_in)
        ppl_score = (perplexity_result or {}).get("perplexity")
        sim_score = (similarity_result or {}).get("similarity_score")

        gate_resp = self._check_gates(
            perplexity_result, similarity_result, tags, ppl_score, sim_score
        )
        if gate_resp is not None:
            return gate_resp

        assert stage_a_result is not None, "stage_a_result required when gates pass"
        return self._check_model_results(
            stage_a_result, stage_b_result, request, tags, ppl_score, sim_score
        )

    def _check_gates(
        self,
        perplexity_result: Optional[dict[str, Any]],
        similarity_result: Optional[dict[str, Any]],
        tags: list[str],
        ppl_score: Optional[float],
        sim_score: Optional[float],
    ) -> Optional[ClassifyResponse]:
        """Return a block response for perplexity/similarity anomalies, or None."""
        if perplexity_result and perplexity_result.get("blocked"):
            tags.append("perplexity_anomaly")
            return ClassifyResponse(
                label="jailbreak",
                risk_scores={"safe": 0.0, "jailbreak": 0.5, "indirect_injection": 0.5},
                decision="block",
                confidence=1.0,
                reason_tags=tags,
                attack_type="gradient_suffix",
                stage_used="perplexity_gate",
                similarity_score=sim_score,
                perplexity_score=ppl_score,
            )

        if similarity_result and similarity_result.get("blocked"):
            tags.append("known_attack_similarity")
            return ClassifyResponse(
                label="jailbreak",
                risk_scores={"safe": 0.0, "jailbreak": 1.0, "indirect_injection": 0.0},
                decision="block",
                confidence=float(similarity_result["similarity_score"]),
                reason_tags=tags,
                attack_type="known_attack_pattern",
                stage_used="similarity_gate",
                similarity_score=sim_score,
                perplexity_score=ppl_score,
            )

        return None

    def _check_model_results(
        self,
        stage_a_result: dict[str, Any],
        stage_b_result: Optional[dict[str, Any]],
        request: ClassifyRequest,
        tags: list[str],
        ppl_score: Optional[float],
        sim_score: Optional[float],
    ) -> ClassifyResponse:
        """Apply rows 1-4 of the decision table using Stage A/B outputs."""
        probs: dict[str, float] = stage_a_result["probabilities"]
        attack_conf = float(probs["jailbreak"]) + float(probs["indirect_injection"])
        label_idx = int(stage_a_result["label"])
        label_name = _LABEL_MAP[label_idx]
        stage_used = "stage_b" if stage_b_result is not None else "stage_a"

        block_a = self._check_stage_a_block(
            probs,
            attack_conf,
            label_idx,
            label_name,
            stage_used,
            stage_a_result,
            tags,
            ppl_score,
            sim_score,
        )
        if block_a is not None:
            return block_a

        block_b = self._check_stage_b_block(
            stage_b_result,
            probs,
            label_idx,
            label_name,
            stage_used,
            tags,
            ppl_score,
            sim_score,
        )
        if block_b is not None:
            return block_b

        has_risk = self._has_risk_flags(request, tags)
        if float(probs["safe"]) >= self._allow_conf and not has_risk:
            return ClassifyResponse(
                label="safe",
                risk_scores=probs,
                decision="allow",
                confidence=float(probs["safe"]),
                reason_tags=tags,
                attack_type=None,
                stage_used=stage_used,
                similarity_score=sim_score,
                perplexity_score=ppl_score,
            )

        tags.append("uncertain_or_risk_context")
        return ClassifyResponse(
            label=label_name,
            risk_scores=probs,
            decision="human_review",
            confidence=float(stage_a_result["confidence"]),
            reason_tags=tags,
            attack_type=label_name if label_idx > 0 else None,
            stage_used=stage_used,
            similarity_score=sim_score,
            perplexity_score=ppl_score,
        )

    def _check_stage_a_block(
        self,
        probs: dict[str, float],
        attack_conf: float,
        label_idx: int,
        label_name: str,
        stage_used: str,
        stage_a_result: dict[str, Any],
        tags: list[str],
        ppl_score: Optional[float],
        sim_score: Optional[float],
    ) -> Optional[ClassifyResponse]:
        """Return a block response when Stage A attack confidence is high, else None."""
        if attack_conf < self._block_conf:
            return None
        tags.append("high_attack_confidence")
        attack_type = (
            "jailbreak"
            if probs["jailbreak"] >= probs["indirect_injection"]
            else "indirect_injection"
        )
        return ClassifyResponse(
            label=label_name if label_idx != 0 else "jailbreak",
            risk_scores=probs,
            decision="block",
            confidence=float(stage_a_result["confidence"]),
            reason_tags=tags,
            attack_type=attack_type,
            stage_used=stage_used,
            similarity_score=sim_score,
            perplexity_score=ppl_score,
        )

    def _check_stage_b_block(
        self,
        stage_b_result: Optional[dict[str, Any]],
        probs: dict[str, float],
        label_idx: int,
        label_name: str,
        stage_used: str,
        tags: list[str],
        ppl_score: Optional[float],
        sim_score: Optional[float],
    ) -> Optional[ClassifyResponse]:
        """Return a block response when Stage B finds a policy violation, else None."""
        if stage_b_result is None:
            return None
        if stage_b_result.get("is_safe", True) and not stage_b_result.get(
            "violation_categories"
        ):
            return None
        tags.append("policy_violation")
        for cat in stage_b_result.get("violation_categories", []):
            tags.append(str(cat))
        return ClassifyResponse(
            label=label_name if label_idx != 0 else "jailbreak",
            risk_scores=probs,
            decision="block",
            confidence=float(stage_b_result.get("risk_score", 0.9)),
            reason_tags=tags,
            attack_type=label_name if label_idx > 0 else "policy_violation",
            stage_used="stage_b",
            similarity_score=sim_score,
            perplexity_score=ppl_score,
        )

    def should_escalate(
        self,
        stage_a_result: dict[str, Any],
        request: ClassifyRequest,
        reason_tags: Optional[list[str]] = None,
    ) -> bool:
        """Return True when Stage B escalation is warranted."""
        confidence = float(stage_a_result.get("confidence", 0.0))
        if self._uncertain_lo <= confidence <= self._uncertain_hi:
            return True
        if request.external_context:
            return True
        if request.conversation_history:
            return True
        if reason_tags is not None:
            for tag in reason_tags:
                if tag in _OBFUSCATION_TAGS:
                    return True
        return False

    def _has_risk_flags(self, request: ClassifyRequest, reason_tags: list[str]) -> bool:
        """Return True when request context carries elevated risk."""
        if request.external_context:
            return True
        if request.conversation_history:
            return True
        if request.source_type in _RISKY_SOURCE_TYPES:
            return True
        for tag in reason_tags:
            if tag in _OBFUSCATION_TAGS:
                return True
        return False
