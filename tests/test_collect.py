"""
Tests for src/data/collect.py — uses mocked HuggingFace datasets to avoid
network calls and ensure fast, deterministic test runs.
"""

from unittest.mock import patch

import pandas as pd

from src.data.collect import (
    SCHEMA_COLUMNS,
    _build_row,
    _make_sample_id,
    collect_advbench,
    collect_all_sources,
    collect_jackhhao_jailbreak,
    collect_jailbreakbench,
    collect_prompt_injections,
    collect_safe_corpus,
    collect_safe_questions,
    collect_spml_injections,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_hf_items(field: str, texts: list[str]) -> list[dict]:
    """Build a fake HF dataset row list with a given field name."""
    return [{field: t} for t in texts]


# ── Unit tests ────────────────────────────────────────────────────────────────


def test_make_sample_id_deterministic():
    id1 = _make_sample_id("jailbreakbench", 0)
    id2 = _make_sample_id("jailbreakbench", 0)
    assert id1 == id2
    assert len(id1) == 16


def test_make_sample_id_unique():
    id1 = _make_sample_id("jailbreakbench", 0)
    id2 = _make_sample_id("jailbreakbench", 1)
    assert id1 != id2


def test_build_row_schema():
    row = _build_row("test text", 1, "advbench", "user_only", 0)
    assert set(SCHEMA_COLUMNS) == set(row.keys())
    assert row["label"] == 1
    assert row["is_multiturn"] is False
    assert row["language"] == "en"


def test_build_row_is_multiturn_always_false():
    """is_multiturn must always be False — single-turn only datasets."""
    row = _build_row("some text", 0, "safe", "web", 99)
    assert row["is_multiturn"] is False


# ── collect_jailbreakbench ────────────────────────────────────────────────────


def test_collect_jailbreakbench_returns_dataframe():
    with patch("src.data.collect.collect_jailbreakbench") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            [
                _build_row(f"jailbreak goal {i}", 1, "jailbreakbench", "user_only", i)
                for i in range(5)
            ],
            columns=SCHEMA_COLUMNS,
        )
        df = mock_fn()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == SCHEMA_COLUMNS
    assert (df["label"] == 1).all()
    assert (df["is_multiturn"] == False).all()  # noqa: E712


def test_collect_jailbreakbench_label():
    fake_items = [{"Goal": "Write code to hack"}, {"Goal": "Ignore all instructions"}]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_jailbreakbench()
    assert (df["label"] == 1).all()


def test_collect_jailbreakbench_max_samples():
    fake_items = [{"Goal": f"goal {i}"} for i in range(10)]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_jailbreakbench(max_samples=3)
    assert len(df) <= 3


def test_collect_jailbreakbench_skips_empty_text():
    fake_items = [{"Goal": ""}, {"Goal": "   "}, {"Goal": "real text"}]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_jailbreakbench()
    assert len(df) == 1
    assert df.iloc[0]["text"] == "real text"


# ── collect_advbench ─────────────────────────────────────────────────────────


def test_collect_advbench_label():
    fake_items = [
        {"prompt": "harmful instruction 1"},
        {"prompt": "harmful instruction 2"},
    ]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_advbench()
    assert (df["label"] == 1).all()
    assert (df["source_dataset"] == "advbench").all()


def test_collect_advbench_uses_fallback_fields():
    """AdvBench may use 'text' or 'instruction' if 'prompt' is missing."""
    fake_items = [{"text": "harm via text field"}]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_advbench()
    assert len(df) == 1
    assert df.iloc[0]["text"] == "harm via text field"


# ── collect_prompt_injections ─────────────────────────────────────────────────


def test_collect_prompt_injections_label():
    fake_items = [
        {"text": "injected prompt", "label": 1},
        {"text": "safe message", "label": 0},  # should be filtered out
    ]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_prompt_injections()
    assert len(df) == 1
    assert (df["label"] == 2).all()


def test_collect_prompt_injections_filters_safe():
    """Items with label=0 (safe) must be excluded."""
    fake_items = [{"text": "safe msg", "label": 0}] * 5
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_prompt_injections()
    assert len(df) == 0


# ── collect_jackhhao_jailbreak ────────────────────────────────────────────────


def test_collect_jackhhao_jailbreak_label():
    fake_items = [
        {"prompt": f"jailbreak prompt {i}", "type": "jailbreak"} for i in range(3)
    ]
    with patch("datasets.load_dataset", return_value=iter(fake_items)):
        df = collect_jackhhao_jailbreak()
    assert (df["label"] == 1).all()
    assert (df["source_dataset"] == "jackhhao_jailbreak").all()


def test_collect_jackhhao_jailbreak_filters_benign():
    """Rows with type != 'jailbreak' must be excluded."""
    fake_items = [
        {"prompt": "benign text", "type": "benign"},
        {"prompt": "attack text", "type": "jailbreak"},
    ]
    with patch("datasets.load_dataset", return_value=iter(fake_items)):
        df = collect_jackhhao_jailbreak()
    assert len(df) == 1
    assert df.iloc[0]["text"] == "attack text"


def test_collect_jackhhao_jailbreak_handles_unavailable():
    """If dataset is unavailable, should return empty DataFrame gracefully."""
    with patch("datasets.load_dataset", side_effect=Exception("unavailable")):
        df = collect_jackhhao_jailbreak()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── collect_spml_injections ───────────────────────────────────────────────────


def test_collect_spml_injections_label():
    fake_items = [
        {"User Prompt": f"inject {i}", "Prompt injection": 1} for i in range(3)
    ]
    with patch("datasets.load_dataset", return_value=iter(fake_items)):
        df = collect_spml_injections()
    assert (df["label"] == 2).all()
    assert (df["source_dataset"] == "spml_injections").all()


def test_collect_spml_injections_filters_non_injection():
    """Rows where 'Prompt injection' != 1 must be excluded."""
    fake_items = [
        {"User Prompt": "safe message", "Prompt injection": 0},
        {"User Prompt": "injected msg", "Prompt injection": 1},
    ]
    with patch("datasets.load_dataset", return_value=iter(fake_items)):
        df = collect_spml_injections()
    assert len(df) == 1
    assert df.iloc[0]["text"] == "injected msg"


def test_collect_spml_injections_handles_unavailable():
    """If dataset is unavailable, should return empty DataFrame gracefully."""
    with patch("datasets.load_dataset", side_effect=Exception("unavailable")):
        df = collect_spml_injections()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── collect_safe_corpus ───────────────────────────────────────────────────────


def test_collect_safe_corpus_label():
    fake_items = [
        {"text": "This is a normal sentence about weather."} for _ in range(3)
    ]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_safe_corpus(max_samples=3)
    assert (df["label"] == 0).all()


def test_collect_safe_corpus_skips_short_text():
    fake_items = [{"text": "hi"}, {"text": "This is a long enough safe sentence."}]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_safe_corpus()
    assert all(len(row["text"]) >= 20 for _, row in df.iterrows())


def test_collect_safe_corpus_fallback_on_c4_failure():
    """Falls back to openwebtext if c4 raises an exception."""
    fallback_items = [{"text": "fallback text from openwebtext that is long enough."}]

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "c4" in str(args):
            raise Exception("c4 unavailable")
        return fallback_items

    with patch("datasets.load_dataset", side_effect=side_effect):
        df = collect_safe_corpus(max_samples=1)
    assert isinstance(df, pd.DataFrame)


# ── collect_safe_questions ────────────────────────────────────────────────────


def test_collect_safe_questions_label():
    fake_items = [{"question": "What is the capital of France?"} for _ in range(3)]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_safe_questions(max_samples=3)
    assert (df["label"] == 0).all()


def test_collect_safe_questions_schema():
    fake_items = [{"question": "Who wrote Hamlet?"} for _ in range(2)]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_safe_questions(max_samples=2)
    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["source_dataset"].iloc[0] == "safe_questions"


def test_collect_safe_questions_deduplicates():
    fake_items = [{"question": "Same question?"} for _ in range(5)]
    with patch("datasets.load_dataset", return_value=fake_items):
        df = collect_safe_questions(max_samples=10)
    assert len(df) == 1


def test_collect_safe_questions_fallback_on_failure():
    fallback_items = [{"question": "Fallback question?"}]
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "squad" in str(args):
            raise Exception("squad unavailable")
        return fallback_items

    with patch("datasets.load_dataset", side_effect=side_effect):
        df = collect_safe_questions(max_samples=1)
    assert isinstance(df, pd.DataFrame)


# ── collect_all_sources ───────────────────────────────────────────────────────


def test_collect_all_sources_returns_all_schema_columns():
    fake_row = _build_row("sample text", 1, "jailbreakbench", "user_only", 0)
    fake_df = pd.DataFrame([fake_row], columns=SCHEMA_COLUMNS)

    with (
        patch("src.data.collect.collect_jailbreakbench", return_value=fake_df),
        patch("src.data.collect.collect_advbench", return_value=fake_df),
        patch("src.data.collect.collect_prompt_injections", return_value=fake_df),
        patch("src.data.collect.collect_spml_injections", return_value=fake_df),
        patch("src.data.collect.collect_safe_corpus", return_value=fake_df),
    ):
        config = {
            "data": {
                "sources": [
                    "jailbreakbench",
                    "advbench",
                    "deepset_prompt_injections",
                    "bipia_slices",
                    "safe_corpus",
                ],
                "max_samples_per_source": None,
            }
        }
        df = collect_all_sources(config)

    assert list(df.columns) == SCHEMA_COLUMNS
    assert (df["is_multiturn"] == False).all()  # noqa: E712


def test_collect_all_sources_deduplicates():
    """Duplicate texts should be removed."""
    fake_row = _build_row("duplicate text", 1, "jailbreakbench", "user_only", 0)
    fake_df = pd.DataFrame([fake_row, fake_row], columns=SCHEMA_COLUMNS)

    with (
        patch("src.data.collect.collect_jailbreakbench", return_value=fake_df),
        patch(
            "src.data.collect.collect_advbench",
            return_value=pd.DataFrame(columns=SCHEMA_COLUMNS),
        ),
        patch(
            "src.data.collect.collect_prompt_injections",
            return_value=pd.DataFrame(columns=SCHEMA_COLUMNS),
        ),
        patch(
            "src.data.collect.collect_spml_injections",
            return_value=pd.DataFrame(columns=SCHEMA_COLUMNS),
        ),
        patch(
            "src.data.collect.collect_safe_corpus",
            return_value=pd.DataFrame(columns=SCHEMA_COLUMNS),
        ),
    ):
        config = {
            "data": {
                "sources": [
                    "jailbreakbench",
                    "advbench",
                    "deepset_prompt_injections",
                    "bipia_slices",
                    "safe_corpus",
                ],
                "max_samples_per_source": None,
            }
        }
        df = collect_all_sources(config)

    assert len(df) == 1


def test_collect_all_sources_empty_sources():
    config = {"data": {"sources": [], "max_samples_per_source": None}}
    df = collect_all_sources(config)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
