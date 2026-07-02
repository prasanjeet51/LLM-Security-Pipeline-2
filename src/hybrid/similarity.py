"""FAISS similarity gate against a known-attack embedding index."""

import os
from typing import Any, Optional

import numpy as np

from src.logger import get_logger


class SimilarityGate:
    """FAISS IndexFlatIP search against known attack embeddings (cosine via IP)."""

    def __init__(
        self,
        config: dict[str, Any],
        encoder: Optional[Any] = None,
        index: Optional[Any] = None,
        attack_texts: Optional[list[str]] = None,
    ) -> None:
        """Load similarity config; encoder/index are injectable for testing."""
        sim_cfg: dict[str, Any] = config["model"]["similarity"]
        self._model_name: str = str(sim_cfg["model_name"])
        self._threshold: float = float(sim_cfg["threshold"])
        self._index_path: str = str(sim_cfg["index_path"])
        self._encoder = encoder
        self._index = index
        self._attack_texts: list[str] = list(attack_texts or [])
        self._logger = get_logger(__name__)

    def _ensure_encoder(self) -> None:
        """Lazy-load SentenceTransformer encoder on first use."""
        if self._encoder is None:  # pragma: no cover - heavy download path
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name)

    def build_index(self, attack_texts: list[str]) -> None:  # pragma: no cover
        """Encode attacks, build a FAISS IP index, persist to config path.

        Excluded from coverage because some CI hosts trip a faiss/AVX2
        import-time access violation; index construction is exercised
        end-to-end by integration tests on dedicated runners.
        """
        import faiss

        self._ensure_encoder()
        assert self._encoder is not None
        embeddings = self._encoder.encode(
            attack_texts, normalize_embeddings=True
        ).astype("float32")
        dim = int(embeddings.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._attack_texts = list(attack_texts)

        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        self._logger.info(
            "faiss_index_built",
            extra={"n_attacks": len(attack_texts), "dim": dim},
        )

    def check(self, text: str) -> dict[str, Any]:
        """Return similarity score + blocked flag against the attack index."""
        if self._index is None or not self._attack_texts:
            return {
                "similarity_score": 0.0,
                "blocked": False,
                "nearest_attack": "",
            }

        self._ensure_encoder()
        assert self._encoder is not None
        query = self._encoder.encode([text], normalize_embeddings=True).astype(
            "float32"
        )
        if not isinstance(query, np.ndarray):
            query = np.asarray(query, dtype="float32")

        scores, indices = self._index.search(query, 1)
        score = float(scores[0][0])
        nearest_idx = int(indices[0][0])
        nearest = (
            self._attack_texts[nearest_idx]
            if 0 <= nearest_idx < len(self._attack_texts)
            else ""
        )
        blocked = score >= self._threshold

        self._logger.info(
            "similarity_checked",
            extra={
                "similarity_score": score,
                "blocked": blocked,
                "threshold": self._threshold,
            },
        )
        return {
            "similarity_score": score,
            "blocked": blocked,
            "nearest_attack": nearest,
        }
