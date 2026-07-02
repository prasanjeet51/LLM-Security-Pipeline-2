import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")
MODEL_CARD = os.path.join(REPO_ROOT, "MODEL_CARD.md")
HF_MODEL_CARD = os.path.join(REPO_ROOT, "hf_space", "MODEL_CARD.md")

FORBIDDEN_TERMS = ["Claude", "Sonnet", "Opus", "Anthropic"]


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_readme_exists() -> None:
    assert os.path.isfile(README), "README.md must exist at repo root"
    assert _read(README).strip(), "README.md must not be empty"


def test_model_card_exists() -> None:
    assert os.path.isfile(MODEL_CARD), "MODEL_CARD.md must exist at repo root"
    assert _read(MODEL_CARD).strip(), "MODEL_CARD.md must not be empty"


def test_model_card_has_limitations() -> None:
    body = _read(MODEL_CARD)
    assert re.search(
        r"^##\s+Known Limitations\s*$", body, flags=re.MULTILINE
    ), "MODEL_CARD.md must contain a '## Known Limitations' section"


def test_no_claude_mentions() -> None:
    for path in (README, MODEL_CARD):
        body = _read(path)
        hits = [t for t in FORBIDDEN_TERMS if re.search(rf"\b{t}\b", body)]
        assert not hits, f"{os.path.basename(path)} contains forbidden terms: {hits}"


def test_hf_space_has_model_card() -> None:
    assert os.path.isfile(HF_MODEL_CARD), (
        "hf_space/MODEL_CARD.md must exist so the live HF Space can render it "
        "in the Model Card tab"
    )


def test_hf_model_card_in_sync_with_root() -> None:
    assert _read(HF_MODEL_CARD) == _read(MODEL_CARD), (
        "hf_space/MODEL_CARD.md drifted from MODEL_CARD.md — "
        "re-copy after editing the root model card"
    )
