"""Hybrid detection pipeline + GPT-2 perplexity gate."""

from typing import Any, Callable, Optional

from src.api.schemas import ClassifyRequest, ClassifyResponse
from src.hybrid.explain import TokenExplainer
from src.hybrid.normalize import InputNormalizer
from src.hybrid.policy_gate import PolicyGate
from src.hybrid.similarity import SimilarityGate
from src.hybrid.stage_a import StageAClassifier
from src.hybrid.stage_b import StageBJudge
from src.logger import get_logger

# Lazy-loaded GPT-2 globals. Tests monkeypatch these module-level names
# with fakes so no network download happens during CI.
_PERPLEXITY_MODEL: Optional[Any] = None
_PERPLEXITY_TOKENIZER: Optional[Any] = None


def _load_perplexity_model() -> None:  # pragma: no cover - heavy download path
    """Download and cache GPT-2 model + tokenizer for perplexity scoring."""
    global _PERPLEXITY_MODEL, _PERPLEXITY_TOKENIZER
    from transformers import (  # type: ignore[attr-defined]
        GPT2LMHeadModel,
        GPT2TokenizerFast,
    )

    _PERPLEXITY_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")  # nosec B615
    _PERPLEXITY_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")  # nosec B615


def perplexity_gate(text: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run BEFORE Stage A on EVERY request including batch (C35).

    Returns {perplexity, blocked, reason_tag}. The caller (HybridPipeline)
    is responsible for attaching reason_tag to the final ClassifyResponse.
    Score is logged on every call regardless of decision.
    """
    logger = get_logger(__name__)
    threshold = float(config["model"].get("perplexity_threshold", 500.0))

    if _PERPLEXITY_MODEL is None or _PERPLEXITY_TOKENIZER is None:
        _load_perplexity_model()

    tokenizer = _PERPLEXITY_TOKENIZER
    model = _PERPLEXITY_MODEL
    assert tokenizer is not None
    assert model is not None

    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = encoded["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    ppl = float(torch.exp(loss).item())
    blocked = ppl > threshold
    reason_tag: Optional[str] = "perplexity_anomaly" if blocked else None

    logger.info(
        "perplexity_gate",
        extra={
            "perplexity": ppl,
            "threshold": threshold,
            "blocked": blocked,
        },
    )
    return {"perplexity": ppl, "blocked": blocked, "reason_tag": reason_tag}


class HybridPipeline:
    """Five-layer pipeline + deterministic policy gate.

    Order: Normalize -> Perplexity -> FAISS Similarity -> Stage A
           -> [Stage B if escalate] -> Policy Gate.
    Every component is injectable so tests can run without heavy models.
    """

    def __init__(
        self,
        config: dict[str, Any],
        normalizer: Optional[InputNormalizer] = None,
        similarity: Optional[SimilarityGate] = None,
        stage_a: Optional[StageAClassifier] = None,
        stage_b: Optional[StageBJudge] = None,
        policy_gate: Optional[PolicyGate] = None,
        perplexity_fn: Optional[Callable[[str, dict[str, Any]], dict[str, Any]]] = None,
        explainer: Optional[TokenExplainer] = None,
    ) -> None:
        """Wire pipeline components; each defaults to its prod implementation."""
        self._config = config
        self._normalizer = normalizer or InputNormalizer(config)
        self._similarity = similarity or SimilarityGate(config)
        self._stage_a = stage_a or StageAClassifier(config)
        self._stage_b = stage_b or StageBJudge(config)
        self._policy = policy_gate or PolicyGate(config)
        self._perplexity_fn = perplexity_fn or perplexity_gate
        self._explainer = explainer  # lazy — only built when explain=True (C38)
        self._logger = get_logger(__name__)

    def classify(
        self, request: ClassifyRequest, explain: bool = False
    ) -> ClassifyResponse:
        """Run the full five-layer pipeline and return a ClassifyResponse."""
        # 1. Normalize first (C36)
        normalized, norm_tags = self._normalizer.normalize(request.user_prompt)
        reason_tags: list[str] = list(norm_tags)

        # 2. Perplexity gate runs on EVERY request (C35)
        ppl_result = self._perplexity_fn(normalized, self._config)
        if ppl_result.get("blocked"):
            return self._policy.decide(
                stage_a_result=None,
                stage_b_result=None,
                perplexity_result=ppl_result,
                similarity_result=None,
                request=request,
                reason_tags_in=reason_tags,
            )

        # 3. FAISS similarity gate (C37)
        sim_result = self._similarity.check(normalized)
        if sim_result.get("blocked"):
            return self._policy.decide(
                stage_a_result=None,
                stage_b_result=None,
                perplexity_result=ppl_result,
                similarity_result=sim_result,
                request=request,
                reason_tags_in=reason_tags,
            )

        # 4. Conversation history concatenation (C33)
        if request.conversation_history:
            full_text = "\n".join(list(request.conversation_history) + [normalized])
        else:
            full_text = normalized

        # 5. Stage A
        stage_a_result = self._stage_a.classify(full_text)

        # 6. Escalate?
        stage_b_result: Optional[dict[str, Any]] = None
        if self._policy.should_escalate(stage_a_result, request, reason_tags):
            stage_b_result = self._stage_b.judge(full_text, stage_a_result)

        # 7. Policy gate decides
        response = self._policy.decide(
            stage_a_result=stage_a_result,
            stage_b_result=stage_b_result,
            perplexity_result=ppl_result,
            similarity_result=sim_result,
            request=request,
            reason_tags_in=reason_tags,
        )

        # 8. Explain (lazy Captum, C38) — only when actually requested
        if explain and response.decision in ("block", "human_review"):
            if self._explainer is None:
                self._explainer = TokenExplainer(
                    self._stage_a._model, self._stage_a._tokenizer
                )
            response.token_attributions = self._explainer.explain(full_text)

        return response

    def classify_batch(self, requests: list[ClassifyRequest]) -> list[ClassifyResponse]:
        """Classify a list of requests; every request goes through the full pipeline."""
        # C35: every request gets the full gate stack — no batch shortcuts.
        return [self.classify(r) for r in requests]
