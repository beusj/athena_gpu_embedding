"""CSV ingestion: read Athena CONCEPT.csv and apply column filters.

Both public functions are pure (no I/O side effects in filter_rows).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gpu_embedder.models import ConceptRow, FilterSpec

logger = logging.getLogger(__name__)

# Columns Athena always provides in CONCEPT.csv
_ATHENA_COLUMNS = [
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
]


def read_csv(path: Path) -> list[ConceptRow]:
    """Read a single Athena CONCEPT.csv (tab-separated) and return validated rows.

    All columns are read as strings first; type coercion happens inside
    ConceptRow validators (empty / "NULL" → None, concept_id → int).
    """
    logger.info("Reading %s", path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,  # don't let pandas silently convert "" to NaN
        usecols=lambda c: c in _ATHENA_COLUMNS,
    )
    # Ensure all expected columns are present (fill with empty string if absent)
    for col in _ATHENA_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    rows: list[ConceptRow] = []
    for record in df.to_dict(orient="records"):
        try:
            rows.append(ConceptRow.model_validate(record))
        except Exception:
            logger.warning("Skipping malformed row: %s", record)
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def filter_rows(rows: list[ConceptRow], spec: FilterSpec) -> list[ConceptRow]:
    """Filter rows according to FilterSpec.

    Logic:
    - OR within each column's include-list.
    - AND across different columns (only rows that pass every non-empty filter).
    - An empty include-list for a column means "accept all values".
    - For invalid_reason the string "valid" is treated as None (null / empty).
    """
    result: list[ConceptRow] = []
    for row in rows:
        if spec.vocabulary_ids and row.vocabulary_id not in spec.vocabulary_ids:
            continue
        if spec.domain_ids and row.domain_id not in spec.domain_ids:
            continue
        if spec.concept_class_ids and row.concept_class_id not in spec.concept_class_ids:
            continue
        if spec.standard_concepts and row.standard_concept not in spec.standard_concepts:
            continue
        if spec.invalid_reasons:
            # normalise "valid" sentinel → None before comparing
            normalised = [None if v == "valid" else v for v in spec.invalid_reasons]
            if row.invalid_reason not in normalised:
                continue
        result.append(row)

    logger.info("filter_rows: %d → %d rows after filtering", len(rows), len(result))
    return result
