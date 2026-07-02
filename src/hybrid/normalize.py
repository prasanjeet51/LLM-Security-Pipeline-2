"""Adversarial input normalization: zero-width, homoglyph, leetspeak."""

import re
from typing import Any

from src.logger import get_logger

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Curated Cyrillic / Greek lookalikes that commonly appear in evasion payloads.
_HOMOGLYPH_MAP: dict[str, str] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "А": "A",
    "Е": "E",
    "О": "O",
    "Р": "P",
    "С": "C",
    "У": "Y",
    "Х": "X",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "ԁ": "d",
    "ԛ": "q",
    "ԝ": "w",
    "α": "a",
    "β": "b",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
}

_LEET_MAP: dict[str, str] = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}


class InputNormalizer:
    """Homoglyph mapping, zero-width char stripping, leetspeak normalization."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Load normalization flags from config.model.normalization."""
        norm_cfg: dict[str, Any] = config.get("model", {}).get("normalization", {})
        self._strip_zw: bool = bool(norm_cfg.get("strip_zero_width", True))
        self._map_homoglyphs: bool = bool(norm_cfg.get("normalize_homoglyphs", True))
        self._normalize_leet: bool = bool(norm_cfg.get("normalize_leetspeak", True))
        self._logger = get_logger(__name__)

    def normalize(self, text: str) -> tuple[str, list[str]]:
        """Return (normalized_text, applied_normalizations).

        applied_normalizations feeds directly into ClassifyResponse.reason_tags
        so downstream consumers can audit exactly which adversarial layers fired.
        """
        applied: list[str] = []
        result = text

        if self._strip_zw:
            stripped = _ZERO_WIDTH_RE.sub("", result)
            if stripped != result:
                applied.append("zero_width_stripped")
                result = stripped

        if self._map_homoglyphs:
            mapped_chars: list[str] = []
            changed = False
            for ch in result:
                replacement = _HOMOGLYPH_MAP.get(ch)
                if replacement is not None:
                    mapped_chars.append(replacement)
                    changed = True
                else:
                    mapped_chars.append(ch)
            if changed:
                applied.append("homoglyph_mapped")
                result = "".join(mapped_chars)

        if self._normalize_leet:
            leet = self._convert_leetspeak(result)
            if leet != result:
                applied.append("leetspeak_normalized")
                result = leet

        if applied:
            self._logger.info(
                "input_normalized",
                extra={"tags": applied, "original_len": len(text)},
            )

        return result, applied

    def _convert_leetspeak(self, text: str) -> str:
        """Replace leet digits adjacent to letters; leave standalone numerics."""
        # Only convert digits that sit adjacent to a letter — prevents mangling
        # legitimate numerics like "ip 127.0.0.1" or "year 2024".
        chars = list(text)
        out: list[str] = []
        for i, ch in enumerate(chars):
            if ch in _LEET_MAP:
                left = chars[i - 1] if i > 0 else ""
                right = chars[i + 1] if i < len(chars) - 1 else ""
                if left.isalpha() or right.isalpha():
                    out.append(_LEET_MAP[ch])
                    continue
            out.append(ch)
        return "".join(out)
