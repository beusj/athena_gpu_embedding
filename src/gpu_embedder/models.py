"""Pydantic models, filter spec, and DuckDB schema DDL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Athena concept row
# ---------------------------------------------------------------------------

class ConceptRow(BaseModel):
    """One row from an Athena CONCEPT.csv file."""

    concept_id: int
    concept_name: str
    domain_id: str | None = None
    vocabulary_id: str | None = None
    concept_class_id: str | None = None
    standard_concept: str | None = None  # "S", "C", or None
    concept_code: str | None = None
    valid_start_date: str | None = None
    valid_end_date: str | None = None
    invalid_reason: str | None = None  # None means valid

    @field_validator(
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, v: object) -> object:
        """Treat empty strings and the literal string 'NULL' as None."""
        if isinstance(v, str) and (v == "" or v.upper() == "NULL"):
            return None
        return v


# ---------------------------------------------------------------------------
# Embedded row (ConceptRow + vector + metadata)
# ---------------------------------------------------------------------------

class EmbeddedRow(BaseModel):
    """A ConceptRow plus its embedding vector and run metadata."""

    concept: ConceptRow
    embedding: list[float]
    embed_text: str        # exact string passed to the tokenizer
    model_version: str     # SHA-256 digest of model weights
    embedded_at: datetime


# ---------------------------------------------------------------------------
# Filter spec
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    """Per-column include-lists.  Empty list means "accept all"."""

    vocabulary_ids: list[str] = field(default_factory=list)
    domain_ids: list[str] = field(default_factory=list)
    concept_class_ids: list[str] = field(default_factory=list)
    standard_concepts: list[str | None] = field(default_factory=list)
    invalid_reasons: list[str | None] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS concept_embeddings (
    concept_id          BIGINT    NOT NULL,
    concept_name        TEXT      NOT NULL,
    domain_id           TEXT,
    vocabulary_id       TEXT,
    concept_class_id    TEXT,
    standard_concept    TEXT,
    concept_code        TEXT,
    invalid_reason      TEXT,
    embedding           FLOAT[768] NOT NULL,
    embed_text          TEXT      NOT NULL,
    model_version       TEXT      NOT NULL,
    embedded_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (concept_id, model_version)
);
"""
