"""Dataclass models, filter spec, and DuckDB schema DDL.

`ConceptRow`/`EmbeddedRow` are plain ``slots`` dataclasses rather than Pydantic
models: at ingest time we build millions of these per run, and Pydantic
`BaseModel` instantiation is ~4× slower than a slots dataclass (it dominated CSV
load).  The light coercion the old validators did — ``concept_id`` → int and
empty/``"NULL"`` → ``None`` — is now pushed into the DuckDB scan (see
``ingest.py``), so the row objects are built from already-typed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Default namespace for OHDSI Athena standard/target concepts.  Source-concept
# datasets (local codes, free text) pass a distinct namespace so they cannot
# collide with Athena concept_ids on the (namespace, concept_id, model_version)
# primary key.
DEFAULT_NAMESPACE = "athena"

# ---------------------------------------------------------------------------
# Athena concept row
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConceptRow:
    """One concept row from an Athena CONCEPT.csv file or a source dataset.

    Field values are expected pre-coerced: ``concept_id`` is an ``int`` and the
    nullable string columns are ``None`` (not ``""``/``"NULL"``).  Ingestion is
    responsible for that coercion.
    """

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
    # Provenance/identity dimension; part of the embedding primary key.
    namespace: str = DEFAULT_NAMESPACE


# ---------------------------------------------------------------------------
# Embedded row (ConceptRow + vector + metadata)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EmbeddedRow:
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
    namespace           TEXT      NOT NULL DEFAULT 'athena',
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
    PRIMARY KEY (namespace, concept_id, model_version)
);
"""

# Tracks which (csv_path, model_version, filter_hash) triples have been fully
# ingested so subsequent runs can skip the expensive CSV load when the source
# file has not changed.  filter_hash is a SHA-256 of the canonical FilterSpec
# so changing filters correctly invalidates the fingerprint.
CSV_FINGERPRINTS_DDL = """
CREATE TABLE IF NOT EXISTS csv_fingerprints (
    csv_path      TEXT      NOT NULL,
    model_version TEXT      NOT NULL,
    filter_hash   TEXT      NOT NULL,
    size_bytes    BIGINT    NOT NULL,
    mtime_ns      BIGINT    NOT NULL,
    sha256        TEXT      NOT NULL,
    row_count     BIGINT    NOT NULL,
    completed_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (csv_path, model_version, filter_hash)
);
"""

# Caches the model_version digest keyed by (model_id, revision, pooling) so
# compute_model_version does not re-hash the ~440 MB weights file on every run.
# Invalidated automatically when the user changes --model, --model-revision, or
# --pooling.  pooling is part of the key because it is folded into the digest:
# without it a mean run would get a stale cls cache hit and silently reuse the
# cls model_version (the exact collision the fold prevents).
MODEL_VERSION_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS model_version_cache (
    model_id  TEXT NOT NULL,
    revision  TEXT NOT NULL,
    pooling   TEXT NOT NULL DEFAULT 'cls',
    sha256    TEXT NOT NULL,
    PRIMARY KEY (model_id, revision, pooling)
);
"""
