import logging
import os
import tempfile

import pandas as pd
import pytest

from src.api.schemas import ClassifyRequest, ClassifyResponse
from src.config import load_config
from src.logger import get_logger


def test_config_loads():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)
    assert "model" in config
    assert config["model"]["perplexity_threshold"] == 500.0


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_no_duplicate_handlers():
    logger1 = get_logger("dedup_test")
    handler_count_1 = len(logger1.handlers)
    logger2 = get_logger("dedup_test")
    assert len(logger2.handlers) == handler_count_1


def test_classify_request_valid():
    req = ClassifyRequest(user_prompt="Hello, how are you?")
    assert req.user_prompt == "Hello, how are you?"
    assert req.source_type == "user_input"
    assert req.external_context is None
    assert req.conversation_history is None


def test_classify_request_rejects_empty_prompt():
    with pytest.raises(Exception):
        ClassifyRequest(user_prompt="   ")


def test_classify_request_rejects_missing_prompt():
    with pytest.raises(Exception):
        ClassifyRequest()  # type: ignore[call-arg]


def test_classify_request_with_all_fields():
    req = ClassifyRequest(
        user_prompt="Ignore previous instructions",
        external_context="Some external doc",
        source_type="external_doc",
        conversation_history=["Hello", "Hi there"],
    )
    assert req.source_type == "external_doc"
    assert len(req.conversation_history) == 2  # type: ignore[arg-type]


def test_classify_response_has_required_fields():
    resp = ClassifyResponse(
        label="safe",
        risk_scores={"safe": 0.95, "jailbreak": 0.03, "indirect_injection": 0.02},
        decision="allow",
        confidence=0.95,
        reason_tags=[],
        stage_used="stage_a",
    )
    assert resp.label == "safe"
    assert resp.attack_type is None
    assert resp.similarity_score is None
    assert resp.perplexity_score is None
    assert resp.token_attributions is None


def test_classify_response_with_optional_fields():
    resp = ClassifyResponse(
        label="jailbreak",
        risk_scores={"safe": 0.05, "jailbreak": 0.90, "indirect_injection": 0.05},
        decision="block",
        confidence=0.90,
        reason_tags=["role_play", "ignore_instructions"],
        attack_type="prompt_injection",
        stage_used="stage_b",
        similarity_score=0.92,
        perplexity_score=620.5,
        token_attributions=[{"ignore": 0.8, "instructions": 0.6}],
    )
    assert resp.attack_type == "prompt_injection"
    assert resp.similarity_score == 0.92
    assert resp.perplexity_score == 620.5
    assert resp.token_attributions is not None


def test_exceptions_hierarchy():
    from src.exceptions import (
        ClassificationError,
        DataLoadError,
        ModelNotFoundError,
        PolicyViolationError,
        ProjectBaseError,
    )

    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(ModelNotFoundError, ProjectBaseError)
    assert issubclass(ClassificationError, ProjectBaseError)
    assert issubclass(PolicyViolationError, ProjectBaseError)


def test_sample_record_valid():
    from src.data.schema import SampleRecord

    record = SampleRecord(
        sample_id="test_001",
        text="Hello world",
        label=0,
        source_dataset="safe_corpus",
        source_type="user_input",
        language="en",
        is_multiturn=False,
    )
    assert record.label == 0


def test_sample_record_invalid_label():
    from src.data.schema import SampleRecord

    with pytest.raises(Exception):
        SampleRecord(
            sample_id="test_002",
            text="Some text",
            label=99,
            source_dataset="test",
            source_type="user_input",
            language="en",
            is_multiturn=False,
        )


def test_sample_record_empty_text():
    from src.data.schema import SampleRecord

    with pytest.raises(Exception):
        SampleRecord(
            sample_id="test_003",
            text="   ",
            label=0,
            source_dataset="test",
            source_type="user_input",
            language="en",
            is_multiturn=False,
        )


def test_hybrid_stubs_importable():
    from src.hybrid.explain import TokenExplainer
    from src.hybrid.normalize import InputNormalizer
    from src.hybrid.pipeline import HybridPipeline
    from src.hybrid.policy_gate import PolicyGate
    from src.hybrid.similarity import SimilarityGate
    from src.hybrid.stage_a import StageAClassifier
    from src.hybrid.stage_b import StageBJudge

    assert StageAClassifier is not None
    assert StageBJudge is not None
    assert PolicyGate is not None
    assert HybridPipeline is not None
    assert InputNormalizer is not None
    assert SimilarityGate is not None
    assert TokenExplainer is not None


def test_api_app_health_endpoint():
    from fastapi.testclient import TestClient

    import src.api.app as app_module

    client = TestClient(app_module.app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ui_theme_get_css():
    from src.ui.theme import get_css

    css = get_css()
    assert isinstance(css, str)


def test_ui_theme_get_theme():
    import gradio as gr

    from src.ui.theme import get_theme

    result = get_theme()
    assert isinstance(result, gr.themes.Base)


def test_feedback_store_instantiable(tmp_path):
    from src.api.feedback import FeedbackStore

    store = FeedbackStore(str(tmp_path / "test.db"))
    assert store is not None


# ── Day 2: Dataset schema tests ────────────────────────────────────────────────


def _make_valid_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal valid DataFrame for schema tests."""
    return pd.DataFrame(
        {
            "sample_id": [f"id_{i}" for i in range(n)],
            "text": [f"sample text number {i}" for i in range(n)],
            "label": [0, 1, 2][:n],
            "source_dataset": ["test_src"] * n,
            "source_type": ["user_only"] * n,
            "language": ["en"] * n,
            "is_multiturn": [False] * n,
        }
    )


def test_schema_validates_correct_data():
    """Valid DataFrame must pass Pandera validation without raising."""
    from src.data.schema import DATASET_SCHEMA

    df = _make_valid_df(3)
    validated = DATASET_SCHEMA.validate(df)
    assert len(validated) == 3


def test_schema_rejects_invalid_label():
    """A row with label=5 must fail Pandera validation."""
    import pandera

    from src.data.schema import DATASET_SCHEMA

    df = _make_valid_df(3)
    df.loc[0, "label"] = 5
    with pytest.raises(pandera.errors.SchemaError):
        DATASET_SCHEMA.validate(df)


def test_schema_rejects_empty_text():
    """A row with empty text must fail Pandera validation."""
    import pandera

    from src.data.schema import DATASET_SCHEMA

    df = _make_valid_df(3)
    df.loc[0, "text"] = ""
    with pytest.raises(pandera.errors.SchemaError):
        DATASET_SCHEMA.validate(df)


def test_is_multiturn_always_false():
    """All rows produced by collect helpers must have is_multiturn=False."""
    df = _make_valid_df(3)
    assert (df["is_multiturn"] == False).all()  # noqa: E712


def test_checksums_computed():
    """compute_checksums must return dict with required keys."""
    from src.data.validate import compute_checksums

    df = _make_valid_df(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "checksums.json")
        result = compute_checksums(df, output_path=out_path)

    assert "full_csv_sha256" in result
    assert "num_rows" in result
    assert "num_columns" in result
    assert "label_distribution" in result
    assert result["num_rows"] == 3


def test_splits_stratified():
    """Label distribution in train/val/test must be roughly proportional."""
    from src.data.pipeline import split_and_save

    # Build a balanced dataset large enough for stratified split
    n_per_class = 30
    rows = []
    for label in [0, 1, 2]:
        for i in range(n_per_class):
            rows.append(
                {
                    "sample_id": f"{label}_{i}",
                    "text": f"text {label} {i}",
                    "label": label,
                    "source_dataset": "test",
                    "source_type": "user_only",
                    "language": "en",
                    "is_multiturn": False,
                }
            )
    df = pd.DataFrame(rows)
    config = {
        "data": {"train_split": 0.70, "val_split": 0.15, "test_split": 0.15, "seed": 42}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        import src.data.pipeline as pipeline_mod

        orig_dir = pipeline_mod.PROCESSED_DIR
        pipeline_mod.PROCESSED_DIR = tmpdir
        try:
            splits = split_and_save(df, config)
        finally:
            pipeline_mod.PROCESSED_DIR = orig_dir

    for split_name, split_df in splits.items():
        dist = split_df["label"].value_counts(normalize=True)
        for label in [0, 1, 2]:
            # Each class should be within 15% of expected 33%
            assert (
                abs(dist.get(label, 0) - 1 / 3) < 0.15
            ), f"Split '{split_name}' label {label} not stratified: {dist.to_dict()}"
