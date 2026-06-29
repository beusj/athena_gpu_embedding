"""CSV ingestion: read Athena CONCEPT.csv and apply DuckDB-backed filters.

`read_csv()` and `filter_rows()` are both pure. DuckDB is the default engine
for CSV scanning and filtering so large Athena files can be narrowed before
Pydantic validation and embedding.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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

_SOURCE_PARQUET_COLUMNS = [
    "source_id",
    "source_name",
    "source_description",
    "source_domain",
    "ehr_codes",
    "sample_units",
    "sample_values",
    "data_type",
]

_SOURCE_TEXT_FIELD_ALIASES = {
    "concept_name": "source_name",
}

_SOURCE_TEXT_FIELDS = {
    "source_id",
    "source_name",
    "source_description",
    "source_domain",
    "ehr_codes",
    "sample_units",
    "sample_values",
    "data_type",
}


@dataclass(slots=True)
class SourceParquetRows:
    """Source-concept parquet rows adapted to `ConceptRow` plus embed texts."""

    rows: list[ConceptRow]
    embed_texts: dict[int, str]


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


def _stable_source_concept_id(source_id: str) -> int:
    """Return a deterministic positive BIGINT-safe surrogate key for `source_id`."""
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _parse_source_ehr_codes(raw: object) -> list[dict[str, str]]:
    """Normalize source `ehr_codes` values to a list of `{system, code}` dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    elif isinstance(raw, list):
        parsed = raw
    else:
        return []

    result: list[dict[str, str]] = []
    if not isinstance(parsed, list):
        return result
    for item in parsed:
        if not isinstance(item, dict):
            continue
        system = item.get("system")
        code = item.get("code")
        if not isinstance(system, str) or not isinstance(code, str):
            continue
        if not system.strip() or not code.strip():
            continue
        result.append({"system": system.strip(), "code": code.strip()})
    return result


def _normalize_source_text_fields(text_fields: list[str]) -> list[str]:
    """Resolve source-text aliases and validate the requested source fields."""
    normalized = [_SOURCE_TEXT_FIELD_ALIASES.get(field, field) for field in text_fields]
    invalid = sorted({field for field in normalized if field not in _SOURCE_TEXT_FIELDS})
    if invalid:
        allowed = ", ".join(sorted(_SOURCE_TEXT_FIELDS | set(_SOURCE_TEXT_FIELD_ALIASES)))
        raise ValueError(
            "Unsupported source text field(s): "
            + ", ".join(invalid)
            + f". Allowed values: {allowed}"
        )
    return normalized


def _format_source_text_value(field_name: str, value: object) -> str | None:
    """Return one source field rendered for embed-text concatenation."""
    if field_name == "ehr_codes":
        codes = _parse_source_ehr_codes(value)
        if not codes:
            return None
        return ", ".join(f"{code['system']}:{code['code']}" for code in codes)

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _build_source_embed_text(
    record: dict[str, object],
    *,
    text_fields: list[str],
    separator: str,
) -> str:
    """Build the tokenizer input string for one source parquet row."""
    parts = [
        rendered
        for field_name in text_fields
        if (rendered := _format_source_text_value(field_name, record.get(field_name))) is not None
    ]
    if parts:
        return separator.join(parts)

    fallback_name = _format_source_text_value("source_name", record.get("source_name"))
    if fallback_name is not None:
        return fallback_name

    source_id = record.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()

    return ""


def read_source_parquet(
    path: Path,
    *,
    namespace: str,
    text_fields: list[str] | None = None,
    separator: str = " ",
) -> SourceParquetRows:
    """Read Stage-0-style source-concept parquet rows and adapt them to `ConceptRow`.

    The parquet schema is the `concept-mapper` `source_concepts` contract
    (`source_id`, `source_name`, `source_description`, `source_domain`,
    `ehr_codes`, `sample_units`, `sample_values`, `data_type`).
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)

    source_text_fields = _normalize_source_text_fields(text_fields or ["source_name"])

    # ``mapping_wave`` is not an embed-text field but must round-trip so the
    # vectors can be rejoined to concept-mapper's ``source_concepts`` on
    # ``(mapping_wave, source_id)``.  Select it alongside the embed columns,
    # guarding against source parquets that predate the column (NULL fallback).
    scan_columns = [*_SOURCE_PARQUET_COLUMNS, "mapping_wave"]

    with duckdb.connect() as conn:
        available = {
            str(row[0])
            for row in conn.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
        }
        select_list = ", ".join(
            col if col in available else f"NULL AS {col}" for col in scan_columns
        )
        sql = f"SELECT {select_list} FROM read_parquet(?)"
        records = conn.execute(sql, [str(path)]).fetchall()

    rows: list[ConceptRow] = []
    embed_texts: dict[int, str] = {}
    source_ids_by_concept_id: dict[int, str] = {}
    skipped = 0

    for record in records:
        raw = dict(zip(scan_columns, record, strict=True))
        source_id = raw["source_id"]
        if not isinstance(source_id, str) or not source_id.strip():
            skipped += 1
            continue
        source_id = source_id.strip()

        concept_id = _stable_source_concept_id(source_id)
        prior_source_id = source_ids_by_concept_id.get(concept_id)
        if prior_source_id is not None and prior_source_id != source_id:
            raise ValueError(
                "Stable hash collision while adapting source parquet: "
                f"{prior_source_id!r} and {source_id!r} map to concept_id {concept_id}"
            )
        source_ids_by_concept_id.setdefault(concept_id, source_id)

        ehr_codes = _parse_source_ehr_codes(raw["ehr_codes"])
        first_code = ehr_codes[0] if ehr_codes else None
        source_name = _nullish_to_none(raw["source_name"] if isinstance(raw["source_name"], str) else None)
        source_domain = _nullish_to_none(
            raw["source_domain"] if isinstance(raw["source_domain"], str) else None
        )
        data_type = _nullish_to_none(raw["data_type"] if isinstance(raw["data_type"], str) else None)

        mapping_wave = _nullish_to_none(
            raw["mapping_wave"] if isinstance(raw["mapping_wave"], str) else None
        )

        row = ConceptRow(
            concept_id=concept_id,
            concept_name=source_name or source_id,
            domain_id=source_domain,
            vocabulary_id=first_code["system"] if first_code is not None else None,
            concept_class_id=data_type,
            standard_concept=None,
            concept_code=first_code["code"] if first_code is not None else source_id,
            valid_start_date=None,
            valid_end_date=None,
            invalid_reason=None,
            namespace=namespace,
            source_id=source_id,
            mapping_wave=mapping_wave,
        )
        rows.append(row)
        embed_texts.setdefault(
            concept_id,
            _build_source_embed_text(raw, text_fields=source_text_fields, separator=separator),
        )

    if skipped:
        logger.warning("Skipped %d malformed source parquet row(s) with blank source_id", skipped)

    logger.info("Loaded %d source parquet rows from %s", len(rows), path)
    return SourceParquetRows(rows=rows, embed_texts=embed_texts)


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


def filter_spec_hash(
    spec: FilterSpec | None,
    namespace: str = DEFAULT_NAMESPACE,
    *,
    extra: dict[str, object] | None = None,
) -> str:
    """Return a stable SHA-256 hex digest of *(spec, namespace)*.

    The digest changes whenever any filter value or the namespace is added or
    removed, so a fingerprint recorded under one filter spec (or namespace) will
    not suppress a run that uses a different one (e.g. ingesting the same file
    under a second namespace, or adding a new --vocabulary-id).
    """
    canonical: dict[str, object] = {
        "namespace": namespace,
        "vocabulary_ids": sorted(spec.vocabulary_ids if spec else []),
        "domain_ids": sorted(spec.domain_ids if spec else []),
        "concept_class_ids": sorted(spec.concept_class_ids if spec else []),
        "standard_concepts": sorted(str(v) for v in (spec.standard_concepts if spec else [])),
        "invalid_reasons": sorted(str(v) for v in (spec.invalid_reasons if spec else [])),
    }
    if extra:
        canonical["extra"] = extra
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
