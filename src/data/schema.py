"""
Pandera schema + Pydantic model for the unified dataset.

C5: use from pandera.pandas import Column, DataFrameSchema (not top-level pandera).
C6: Pydantic v2 @field_validator + @classmethod only.
"""

import pandera
from pandera.pandas import Column, DataFrameSchema
from pydantic import BaseModel, field_validator

# Canonical Pandera schema — validates every DataFrame before save/load.
DATASET_SCHEMA = DataFrameSchema(
    {
        "sample_id": Column(str, unique=True, nullable=False),
        "text": Column(
            str,
            checks=pandera.Check(
                lambda s: s.str.strip().str.len() > 0, name="non_empty_text"
            ),
            nullable=False,
        ),
        "label": Column(int, pandera.Check.isin([0, 1, 2]), nullable=False),
        "source_dataset": Column(str, nullable=False),
        "source_type": Column(str, nullable=False),
        "language": Column(str, nullable=False),
        "is_multiturn": Column(bool, nullable=False),
    }
)


class SampleRecord(BaseModel):
    """Pydantic v2 model for a single dataset row."""

    sample_id: str
    text: str
    label: int
    source_dataset: str
    source_type: str
    language: str
    is_multiturn: bool

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(
                "label must be 0 (safe), 1 (jailbreak), or 2 (indirect_injection)"
            )
        return v

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v
