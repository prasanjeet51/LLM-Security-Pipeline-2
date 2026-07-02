"""Tests for src/ui/theme.py and src/ui/gradio_app.py (C4)."""

import gradio as gr
import pytest

from src.api.schemas import ClassifyResponse
from src.ui.gradio_app import (
    _attributions_to_highlights,
    build_app,
    classify_stream,
)
from src.ui.theme import get_css, get_theme


class _MockPipeline:
    """Minimal pipeline stub — no model loading, deterministic output."""

    def classify(self, request: object, explain: bool = False) -> ClassifyResponse:
        return ClassifyResponse(
            label="safe",
            risk_scores={"safe": 0.95, "jailbreak": 0.03, "indirect_injection": 0.02},
            decision="allow",
            confidence=0.95,
            reason_tags=[],
            stage_used="stage_a",
            perplexity_score=120.5,
            similarity_score=0.12,
            token_attributions=[
                {"token": "Hello", "attribution": 0.05},
                {"token": "world", "attribution": 0.01},
            ],
        )


class _MockBlockPipeline(_MockPipeline):
    """Returns a BLOCK decision for testing decision card."""

    def classify(self, request: object, explain: bool = False) -> ClassifyResponse:
        return ClassifyResponse(
            label="jailbreak",
            risk_scores={"safe": 0.02, "jailbreak": 0.92, "indirect_injection": 0.06},
            decision="block",
            confidence=0.92,
            reason_tags=["jailbreak_pattern"],
            attack_type="role_play",
            stage_used="stage_a",
            perplexity_score=380.0,
            similarity_score=0.87,
            token_attributions=[
                {"token": "ignore", "attribution": 0.8},
                {"token": "rules", "attribution": 0.6},
            ],
        )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_gradio_app_launches() -> None:
    """build_app() with a mock pipeline must return a gr.Blocks without crash."""
    app = build_app(pipeline=_MockPipeline())  # type: ignore[arg-type]
    assert app is not None
    # gr.Blocks is the base class for Gradio apps
    assert hasattr(app, "launch")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_quick_check_returns_decision() -> None:
    """classify_stream must eventually yield a tuple containing a decision card."""
    pipe = _MockBlockPipeline()  # type: ignore[arg-type]
    stages = list(classify_stream("test prompt", "", pipeline=pipe))
    assert len(stages) >= 3
    final_html, final_highlights = stages[-1]
    assert (
        "BLOCK" in final_html or "ALLOW" in final_html or "HUMAN REVIEW" in final_html
    )
    assert isinstance(final_highlights, list)
    assert len(final_highlights) > 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_theme_loads() -> None:
    """get_theme() must return a valid gr.themes.Base instance."""
    theme = get_theme()
    assert isinstance(theme, gr.themes.Base)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_css_not_empty() -> None:
    """get_css() must return a non-empty string with required selectors."""
    css = get_css()
    assert isinstance(css, str)
    assert len(css) > 500
    for selector in ("shimmer", "slideUpFadeIn", "decision-allow", "stage-step"):
        assert selector in css, f"Missing CSS selector: {selector}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_streaming_yields_stages() -> None:
    """Generator must yield normalization, perplexity, and FAISS stages."""
    pipe = _MockPipeline()  # type: ignore[arg-type]
    stages = list(classify_stream("hello world", "", pipeline=pipe))
    all_html = " ".join(html for html, _ in stages)
    assert "Normalizing" in all_html
    assert "Perplexity" in all_html
    assert "FAISS" in all_html
    assert "Stage A" in all_html


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_highlighted_text_format() -> None:
    """_attributions_to_highlights must produce valid HighlightedText tuples."""
    attributions = [
        {"token": "hello", "attribution": 0.05},
        {"token": "bad", "attribution": 0.9},
        {"token": "word", "attribution": -0.3},
        {"token": ".", "attribution": 0.0},
    ]
    result = _attributions_to_highlights(attributions)
    assert isinstance(result, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
    tokens = [t[0] for t in result]
    labels = [t[1] for t in result]
    assert "hello" in tokens
    assert "high" in labels, "score 0.9 should map to 'high'"
    assert "low" in labels, "score -0.3 should map to 'low'"
    # score 0.0 maps to None (neutral)
    neutral = [lbl for tok, lbl in result if tok == "."]
    assert neutral[0] is None

    # empty attributions fallback
    fallback = _attributions_to_highlights(None)
    assert len(fallback) == 1
    assert fallback[0][0] == "No attribution data available"
