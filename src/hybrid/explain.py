"""Token-level explainability via Captum integrated gradients.

Captum is NEVER imported at module load (C38) — it is pulled in only when
explain() is called and the underlying LayerIntegratedGradients is missing.
"""

from typing import Any, Optional

from src.logger import get_logger


class TokenExplainer:
    """Wrap a sequence classifier with Captum LayerIntegratedGradients."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        lig: Optional[Any] = None,
    ) -> None:
        """Store model/tokenizer; lig is lazily built on first explain() call."""
        self._model = model
        self._tokenizer = tokenizer
        self._lig = lig
        self._logger = get_logger(__name__)

    def _ensure_lig(self) -> None:
        """Build LayerIntegratedGradients on first call; no-op thereafter."""
        if self._lig is not None:
            return
        # pragma: no cover - heavy captum import path
        from captum.attr import LayerIntegratedGradients  # pragma: no cover

        embeddings_layer = self._model.get_input_embeddings()  # pragma: no cover
        self._lig = LayerIntegratedGradients(  # pragma: no cover
            self._forward_fn, embeddings_layer
        )

    def _forward_fn(self, input_ids: Any) -> Any:  # pragma: no cover
        """Forward pass returning max logit — required by LayerIntegratedGradients."""
        return self._model(input_ids).logits.max(dim=-1).values

    def explain(self, text: str) -> list[dict[str, str | float]]:
        """Return token-level attribution scores via integrated gradients."""
        self._ensure_lig()
        assert self._lig is not None

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"]
        attributions = self._lig.attribute(input_ids)
        # Sum over embedding dim, drop batch dim, convert to plain floats.
        scores = attributions.sum(dim=-1).squeeze(0).tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])

        out: list[dict[str, str | float]] = []
        for tok, sc in zip(tokens, scores):
            out.append({"token": str(tok), "score": float(sc)})

        self._logger.info(
            "explain_computed",
            extra={"n_tokens": len(out)},
        )
        return out
