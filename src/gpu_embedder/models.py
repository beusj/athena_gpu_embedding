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
    # Source-dataset provenance (NULL for Athena concepts).  These round-trip the
    # concept-mapper ``source_concepts`` natural key so embedded source rows can
    # be rejoined on ``(mapping_wave, source_id)``.  ``concept_id`` for a source
    # row is only a hashed surrogate (see ``ingest._stable_source_concept_id``)
    # and cannot reconstruct ``source_id``, so it is carried explicitly here.
    source_id: str | None = None
    mapping_wave: str | None = None


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
# Default "highest-yield" vocabularies
# ---------------------------------------------------------------------------

# When `embed` is run without any `--vocabulary-id`, the filter would otherwise
# match *every* vocabulary in CONCEPT.csv (millions of rows, much of it low-value
# for downstream concept mapping). Instead we default to this curated set of the
# highest-yield Athena vocabularies, covering conditions, procedures, drugs, labs,
# demographics, and provider taxonomies.
#
# These are the exact, case-sensitive Athena ``vocabulary_id`` strings (DuckDB
# string equality is case-sensitive). To bypass this default and embed every
# vocabulary, pass the reserved sentinel ``--vocabulary-id all``.
DEFAULT_VOCABULARY_IDS: tuple[str, ...] = (
    "SNOMED",            # conditions, clinical findings (standard)
    "ICD9CM",            # legacy US diagnoses
    "ICD10CM",           # US diagnoses
    "ICD9Proc",          # legacy US inpatient procedures
    "ICD10PCS",          # US inpatient procedures
    "CPT4",              # outpatient procedures (names require the `cpt4` step)
    "LOINC",             # lab tests, measurements, vitals
    "RxNorm",            # drugs (standard)
    "RxNorm Extension",  # OHDSI drugs not covered by RxNorm
    "NDC",               # drug product codes
    "Race",              # demographics
    "Ethnicity",         # demographics
    "ABMS",              # provider board certifications / specialties
    "NUCC",              # provider taxonomy / specialties
    "Medicare Specialty",  # CMS provider/supplier specialty codes
)

# Reserved sentinel value for `--vocabulary-id` that disables the highest-yield
# default and embeds every vocabulary present in the source CSV(s).
ALL_VOCABULARIES_SENTINEL = "all"


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
    -- Source-dataset provenance; NULL for Athena concepts.  Carries the
    -- concept-mapper source_concepts key so embedded source rows round-trip
    -- back on (mapping_wave, source_id).  Not part of the primary key (the
    -- hashed concept_id surrogate already disambiguates within a namespace).
    source_id           TEXT,
    mapping_wave        TEXT,
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

# Caches the SHA-256 digest of HuggingFace model weights keyed by
# (model_id, revision) so compute_model_version does not re-hash the
# ~440 MB weights file on every run.  Invalidated automatically when
# the user changes --model or --model-revision.
MODEL_VERSION_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS model_version_cache (
    model_id  TEXT NOT NULL,
    revision  TEXT NOT NULL,
    sha256    TEXT NOT NULL,
    PRIMARY KEY (model_id, revision)
);
"""
