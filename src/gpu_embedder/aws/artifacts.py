"""Artifact (de)serialization for the AWS execution path.

Two NDJSON artifact shapes move across S3:

* **input shards** — :class:`~gpu_embedder.models.ConceptRow` records produced by
  the local ingest/filter step (the "athena vocab + source concepts" moved to
  AWS);
* **output shards** — :class:`~gpu_embedder.models.EmbeddedRow` records produced
  by the remote worker (the embeddings "exported back").

NDJSON is used (rather than parquet) to avoid a heavyweight ``pyarrow``
dependency and to match the staging format already used by
:mod:`gpu_embedder.store`. The :class:`RunManifest` ties a run's shards,
filters, and model identity together so :func:`collect_run` can validate before
importing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gpu_embedder.models import ConceptRow, EmbeddedRow

# Metadata columns carried alongside concept_id / concept_name on an input shard.
_CONCEPT_FIELDS = (
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


class RunManifest(BaseModel):
    """Describes one AWS embedding run: shards, model identity, and embed params.

    Uploaded to S3 next to the input shards so the remote worker reads the exact
    same embedding parameters the submitter intended, and so ``collect`` can
    validate outputs against the run's expected model version.
    """

    run_id: str
    created_at: datetime
    model: str
    model_revision: str | None
    model_version: str | None  # SHA-256 of weights; may be unknown at submit time
    embedding_dim: int
    text_fields: list[str]
    separator: str
    max_length: int
    batch_size: int
    num_shards: int
    total_rows: int

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        return cls.model_validate_json(text)


# ---------------------------------------------------------------------------
# ConceptRow <-> dict / NDJSON  (input shards)
# ---------------------------------------------------------------------------


def concept_row_to_dict(row: ConceptRow) -> dict[str, Any]:
    return {field: getattr(row, field) for field in _CONCEPT_FIELDS}


def concept_row_from_dict(payload: dict[str, Any]) -> ConceptRow:
    return ConceptRow.model_validate(payload)


def write_concept_rows(path: Path, rows: list[ConceptRow]) -> None:
    """Write *rows* as NDJSON (one ConceptRow per line)."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(concept_row_to_dict(row)))
            handle.write("\n")


def read_concept_rows(path: Path) -> list[ConceptRow]:
    """Read an NDJSON input shard back into ConceptRow objects."""
    rows: list[ConceptRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(concept_row_from_dict(json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# EmbeddedRow <-> dict / NDJSON  (output shards)
# ---------------------------------------------------------------------------


def embedded_row_to_dict(row: EmbeddedRow) -> dict[str, Any]:
    return {
        "concept_id": row.concept.concept_id,
        "concept_name": row.concept.concept_name,
        "domain_id": row.concept.domain_id,
        "vocabulary_id": row.concept.vocabulary_id,
        "concept_class_id": row.concept.concept_class_id,
        "standard_concept": row.concept.standard_concept,
        "concept_code": row.concept.concept_code,
        "valid_start_date": row.concept.valid_start_date,
        "valid_end_date": row.concept.valid_end_date,
        "invalid_reason": row.concept.invalid_reason,
        "embedding": row.embedding,
        "embed_text": row.embed_text,
        "model_version": row.model_version,
        "embedded_at": row.embedded_at.isoformat(),
    }


def embedded_row_from_dict(payload: dict[str, Any]) -> EmbeddedRow:
    concept = ConceptRow.model_validate(
        {field: payload.get(field) for field in _CONCEPT_FIELDS}
    )
    return EmbeddedRow(
        concept=concept,
        embedding=[float(x) for x in payload["embedding"]],
        embed_text=payload["embed_text"],
        model_version=payload["model_version"],
        embedded_at=datetime.fromisoformat(payload["embedded_at"]),
    )


def write_embedded_rows(path: Path, rows: list[EmbeddedRow]) -> None:
    """Write *rows* as NDJSON (one EmbeddedRow per line)."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(embedded_row_to_dict(row)))
            handle.write("\n")


def read_embedded_rows(path: Path) -> list[EmbeddedRow]:
    """Read an NDJSON output shard back into EmbeddedRow objects."""
    rows: list[EmbeddedRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(embedded_row_from_dict(json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embedded_rows(
    rows: list[EmbeddedRow],
    *,
    embedding_dim: int,
    expected_model_version: str | None = None,
) -> None:
    """Raise ``ValueError`` if any row violates dimension or model-version rules.

    Mirrors the import-validation invariant from the runbook: dimension must
    match and (when pinned) all rows must share the expected model version.
    """
    for row in rows:
        if len(row.embedding) != embedding_dim:
            raise ValueError(
                f"concept_id={row.concept.concept_id} has embedding dimension "
                f"{len(row.embedding)}, expected {embedding_dim}"
            )
        if (
            expected_model_version is not None
            and row.model_version != expected_model_version
        ):
            raise ValueError(
                f"concept_id={row.concept.concept_id} has model_version "
                f"{row.model_version[:16]}…, expected "
                f"{expected_model_version[:16]}…"
            )
