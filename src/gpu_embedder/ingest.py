"""CSV ingestion: read Athena CONCEPT.csv and apply DuckDB-backed filters.

`read_csv()` and `filter_rows()` are both pure. DuckDB is the default engine
for CSV scanning and filtering so large Athena files can be narrowed before
Pydantic validation and embedding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import duckdb

from gpu_embedder.models import DEFAULT_NAMESPACE, ConceptRow, FilterSpec

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


def _sql_quote(value: str) -> str:
    """Return a safely quoted SQL string literal for DuckDB expressions."""
    return "'" + value.replace("'", "''") + "'"


def _nullish_predicate(column: str) -> str:
    """Match Athena nullish strings: NULL, empty string, or actual NULL."""
    return f"({column} IS NULL OR {column} = '' OR UPPER({column}) = 'NULL')"


def _in_predicate(column: str, values: list[str]) -> str:
    """Build an IN predicate for a non-empty list of string values."""
    literals = ", ".join(_sql_quote(value) for value in values)
    return f"{column} IN ({literals})"


def _nullable_predicate(column: str, values: list[str | None]) -> str | None:
    """Build a predicate for fields that may legitimately be nullish."""
    non_null_values = [value for value in values if value is not None]
    include_null = any(value is None for value in values)

    predicates: list[str] = []
    if non_null_values:
        predicates.append(_in_predicate(column, non_null_values))
    if include_null:
        predicates.append(_nullish_predicate(column))

    if not predicates:
        return None
    return "(" + " OR ".join(predicates) + ")"


def _build_where_clause(spec: FilterSpec | None) -> str:
    """Translate a FilterSpec into a DuckDB WHERE clause."""
    if spec is None:
        return ""

    predicates: list[str] = []
    if spec.vocabulary_ids:
        predicates.append(_in_predicate("vocabulary_id", spec.vocabulary_ids))
    if spec.domain_ids:
        predicates.append(_in_predicate("domain_id", spec.domain_ids))
    if spec.concept_class_ids:
        predicates.append(_in_predicate("concept_class_id", spec.concept_class_ids))
    if spec.standard_concepts:
        predicate = _nullable_predicate("standard_concept", spec.standard_concepts)
        if predicate is not None:
            predicates.append(predicate)
    if spec.invalid_reasons:
        normalized_invalid_reasons = [
            None if value == "valid" else value for value in spec.invalid_reasons
        ]
        predicate = _nullable_predicate("invalid_reason", normalized_invalid_reasons)
        if predicate is not None:
            predicates.append(predicate)

    if not predicates:
        return ""
    return " WHERE " + " AND ".join(predicates)


def _nullish_to_none(value: str | None) -> str | None:
    """Treat empty strings and the literal string 'NULL' (any case) as None."""
    if value is None or value == "" or value.upper() == "NULL":
        return None
    return value


def _coerced_scan_columns() -> str:
    """SELECT list that coerces a raw all-varchar Athena scan to typed columns.

    Replaces the old per-row Pydantic validators: ``concept_id`` is cast to
    BIGINT (uncastable → NULL → row dropped) and the nullable string columns
    map empty/``"NULL"`` to SQL NULL.  Doing this in DuckDB lets the row objects
    be built from already-typed values, which is far cheaper than validating
    millions of rows in Python.
    """
    parts: list[str] = []
    for col in _ATHENA_COLUMNS:
        if col == "concept_id":
            parts.append("TRY_CAST(concept_id AS BIGINT) AS concept_id")
        elif col == "concept_name":
            parts.append("concept_name")
        else:
            parts.append(
                f"CASE WHEN {col} IS NULL OR {col} = '' OR upper({col}) = 'NULL' "
                f"THEN NULL ELSE {col} END AS {col}"
            )
    return ", ".join(parts)


def _records_to_concepts(
    records: list[tuple[object, ...]],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> list[ConceptRow]:
    """Build ConceptRow objects from pre-coerced row tuples.

    Tuples are in ``_ATHENA_COLUMNS`` order with ``concept_id`` already an int
    (or None) and nullable strings already None.  Rows with a null
    ``concept_id`` or ``concept_name`` are dropped (they cannot satisfy the
    NOT NULL store columns).
    """
    rows: list[ConceptRow] = []
    skipped = 0
    for record in records:
        if record[0] is None or record[1] is None:
            skipped += 1
            continue
        rows.append(ConceptRow(*record, namespace=namespace))
    if skipped:
        logger.warning("Skipped %d malformed row(s) with null concept_id/concept_name", skipped)
    return rows


def count_csv_rows(path: Path) -> int:
    """Count rows in an Athena TSV via DuckDB without loading them all into Python."""
    logger.info("Counting rows in %s", path)
    with duckdb.connect() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM read_csv(?, delim='\t', header=true, all_varchar=true)",
            [str(path)],
        ).fetchone()
    return int(result[0]) if result is not None else 0


def _read_csv_python(path: Path, *, namespace: str = DEFAULT_NAMESPACE) -> list[ConceptRow]:
    """Read Athena TSV rows in pure Python as a fallback engine."""
    rows: list[ConceptRow] = []
    skipped = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            raw_id = record.get("concept_id")
            try:
                concept_id = int(raw_id) if raw_id not in (None, "") else None
            except (TypeError, ValueError):
                concept_id = None
            concept_name = record.get("concept_name") or ""
            if concept_id is None:
                skipped += 1
                continue
            rows.append(
                ConceptRow(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    domain_id=_nullish_to_none(record.get("domain_id")),
                    vocabulary_id=_nullish_to_none(record.get("vocabulary_id")),
                    concept_class_id=_nullish_to_none(record.get("concept_class_id")),
                    standard_concept=_nullish_to_none(record.get("standard_concept")),
                    concept_code=_nullish_to_none(record.get("concept_code")),
                    valid_start_date=_nullish_to_none(record.get("valid_start_date")),
                    valid_end_date=_nullish_to_none(record.get("valid_end_date")),
                    invalid_reason=_nullish_to_none(record.get("invalid_reason")),
                    namespace=namespace,
                )
            )
    if skipped:
        logger.warning("Skipped %d malformed row(s) with null concept_id", skipped)
    return rows


def read_csv(
    path: Path,
    spec: FilterSpec | None = None,
    engine: str = "duckdb",
    namespace: str = DEFAULT_NAMESPACE,
) -> list[ConceptRow]:
    """Read a single Athena CONCEPT.csv and return ConceptRow objects.

    DuckDB is the default scanner/filter engine. Columns are loaded as strings
    first; type coercion (concept_id → BIGINT, empty/"NULL" → NULL) happens in
    the scan SELECT so the Python objects are built from typed values.
    All rows are tagged with *namespace* (default ``athena``).
    """
    logger.info("Reading %s", path)
    if engine == "python":
        rows = _read_csv_python(path, namespace=namespace)
        return filter_rows(rows, spec) if spec is not None else rows

    sql = (
        "SELECT "
        + _coerced_scan_columns()
        + " FROM read_csv(?, delim='\\t', header=true, all_varchar=true)"
        + _build_where_clause(spec)
    )
    with duckdb.connect() as conn:
        records = conn.execute(sql, [str(path)]).fetchall()

    rows = _records_to_concepts(records, namespace=namespace)
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def filter_rows(rows: list[ConceptRow], spec: FilterSpec) -> list[ConceptRow]:
    """Filter in-memory rows using DuckDB semantics.

    This keeps filtering behavior consistent with `read_csv(..., spec=...)`,
    which pushes the same logic down into DuckDB for the default ingest path.
    """
    where_clause = _build_where_clause(spec)
    if not rows or not where_clause:
        logger.info("filter_rows: %d → %d rows after filtering", len(rows), len(rows))
        return list(rows)

    # Rows passed in already share one namespace (read_csv tags uniformly);
    # preserve it through the DuckDB round-trip.
    namespace = rows[0].namespace

    with duckdb.connect() as conn:
        conn.execute(
            """
            CREATE TEMP TABLE concept_rows (
                concept_id VARCHAR,
                concept_name VARCHAR,
                domain_id VARCHAR,
                vocabulary_id VARCHAR,
                concept_class_id VARCHAR,
                standard_concept VARCHAR,
                concept_code VARCHAR,
                valid_start_date VARCHAR,
                valid_end_date VARCHAR,
                invalid_reason VARCHAR
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO concept_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row.concept_id),
                    row.concept_name,
                    row.domain_id,
                    row.vocabulary_id,
                    row.concept_class_id,
                    row.standard_concept,
                    row.concept_code,
                    row.valid_start_date,
                    row.valid_end_date,
                    row.invalid_reason,
                )
                for row in rows
            ],
        )
        records = conn.execute(
            "SELECT "
            + _coerced_scan_columns()
            + " FROM concept_rows"
            + where_clause
        ).fetchall()

    result = _records_to_concepts(records, namespace=namespace)
    logger.info("filter_rows: %d → %d rows after filtering", len(rows), len(result))
    return result


# ---------------------------------------------------------------------------
# CSV fingerprinting
# ---------------------------------------------------------------------------

def compute_csv_fingerprint(path: Path) -> dict[str, object]:
    """Return a fingerprint dict for *path* containing size_bytes, mtime_ns, and sha256.

    Used to detect whether the source CSV has changed since the last
    successful ingest run, so we can skip the expensive read_csv() call.
    """
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": h.hexdigest(),
    }


def filter_spec_hash(spec: FilterSpec | None) -> str:
    """Return a stable SHA-256 hex digest of *spec*.

    The digest changes whenever any filter value is added or removed, so
    a fingerprint recorded under one filter spec will not suppress a run
    that uses a different spec (e.g. adding a new --vocabulary-id).
    """
    canonical: dict[str, list[str]] = {
        "vocabulary_ids": sorted(spec.vocabulary_ids if spec else []),
        "domain_ids": sorted(spec.domain_ids if spec else []),
        "concept_class_ids": sorted(spec.concept_class_ids if spec else []),
        "standard_concepts": sorted(str(v) for v in (spec.standard_concepts if spec else [])),
        "invalid_reasons": sorted(str(v) for v in (spec.invalid_reasons if spec else [])),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
