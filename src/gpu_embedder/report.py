"""Embedding coverage reporting: DB summaries and CSV-vs-DB gap analysis.

All public functions are pure (no CLI side effects) and fully unit-testable
with in-memory DuckDB connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelVersionInfo:
    """One stored model version with its row count and embed timestamp range."""

    model_version: str
    count: int
    first_embedded_at: datetime
    last_embedded_at: datetime

    @property
    def short_hash(self) -> str:
        """First 16 hex chars of the SHA-256 digest, for display."""
        return self.model_version[:16]


@dataclass
class VocabCoverage:
    """Coverage statistics for one (vocabulary_id, domain_id) combination."""

    vocabulary_id: str
    domain_id: str
    total: int       # concepts present in the source CSV (or DB when DB-only)
    embedded: int    # concepts with a stored embedding

    @property
    def gap(self) -> int:
        """Number of source concepts that have not been embedded."""
        return self.total - self.embedded

    @property
    def pct(self) -> float:
        """Percentage of source concepts that have been embedded (0–100)."""
        return 100.0 * self.embedded / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# DB-only queries
# ---------------------------------------------------------------------------


def list_model_versions(conn: duckdb.DuckDBPyConnection) -> list[ModelVersionInfo]:
    """Return all stored model versions, most-recently-embedded first.

    Returns an empty list when the table does not yet exist.
    """
    try:
        rows = conn.execute(
            """
            SELECT
                model_version,
                COUNT(*)                    AS cnt,
                MIN(embedded_at)            AS first_at,
                MAX(embedded_at)            AS last_at
            FROM concept_embeddings
            GROUP BY model_version
            ORDER BY last_at DESC
            """
        ).fetchall()
    except duckdb.CatalogException:
        logger.debug("concept_embeddings table does not exist yet")
        return []

    return [
        ModelVersionInfo(
            model_version=r[0],
            count=int(r[1]),
            first_embedded_at=r[2],
            last_embedded_at=r[3],
        )
        for r in rows
    ]


def embedded_summary(
    conn: duckdb.DuckDBPyConnection,
    model_version: str | None = None,
) -> list[VocabCoverage]:
    """Return per-(vocabulary_id, domain_id) counts from concept_embeddings.

    When *model_version* is ``None``, aggregates across all model versions.
    Returns an empty list when the table does not yet exist.

    ``total`` and ``embedded`` are both set to the count because only stored
    concepts are known here (no source CSV is consulted).
    """
    where = "WHERE model_version = ?" if model_version else ""
    params = [model_version] if model_version else []
    try:
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(vocabulary_id, '')  AS vocabulary_id,
                COALESCE(domain_id, '')      AS domain_id,
                COUNT(*)                     AS cnt
            FROM concept_embeddings
            {where}
            GROUP BY vocabulary_id, domain_id
            ORDER BY vocabulary_id, domain_id
            """,
            params,
        ).fetchall()
    except duckdb.CatalogException:
        logger.debug("concept_embeddings table does not exist yet")
        return []

    return [
        VocabCoverage(vocabulary_id=r[0], domain_id=r[1], total=r[2], embedded=r[2])
        for r in rows
    ]


# ---------------------------------------------------------------------------
# CSV-vs-DB gap analysis
# ---------------------------------------------------------------------------


def coverage_report(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    model_version: str | None = None,
) -> list[VocabCoverage]:
    """Compare *csv_path* against concept_embeddings and return per-group coverage.

    For each (vocabulary_id, domain_id) pair found in the source CSV:

    - ``total``    = total concepts in that group in the CSV
    - ``embedded`` = subset that has a stored embedding (for *model_version*)

    When *model_version* is ``None``, a concept is considered embedded if it
    appears in concept_embeddings under **any** model version.

    The CSV is scanned entirely inside DuckDB alongside the existing
    concept_embeddings table, so no Python-level iteration is needed.
    The table may not yet exist; in that case all rows report embedded=0.
    """
    # Build the embedded-ID subquery
    if model_version:
        emb_subquery = (
            "SELECT DISTINCT concept_id FROM concept_embeddings "
            "WHERE model_version = ?"
        )
        params: list[object] = [str(csv_path), model_version]
    else:
        try:
            conn.execute("SELECT 1 FROM concept_embeddings LIMIT 1")
        except duckdb.CatalogException:
            # Table doesn't exist at all — every concept is a gap
            return _coverage_from_csv_only(conn, csv_path)
        emb_subquery = "SELECT DISTINCT concept_id FROM concept_embeddings"
        params = [str(csv_path)]

    try:
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(src.vocabulary_id, '')  AS vocabulary_id,
                COALESCE(src.domain_id, '')      AS domain_id,
                COUNT(*)                         AS total,
                COUNT(emb.concept_id)            AS embedded
            FROM read_csv(
                ?,
                delim='\t',
                header=true,
                all_varchar=true
            ) AS src
            LEFT JOIN ({emb_subquery}) AS emb
                ON TRY_CAST(src.concept_id AS BIGINT) = emb.concept_id
            GROUP BY src.vocabulary_id, src.domain_id
            ORDER BY src.vocabulary_id, src.domain_id
            """,
            params,
        ).fetchall()
    except duckdb.CatalogException:
        return _coverage_from_csv_only(conn, csv_path)

    return [
        VocabCoverage(
            vocabulary_id=r[0],
            domain_id=r[1],
            total=int(r[2]),
            embedded=int(r[3]),
        )
        for r in rows
    ]


def _coverage_from_csv_only(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
) -> list[VocabCoverage]:
    """Return coverage rows with embedded=0 for every group in the CSV."""
    rows = conn.execute(
        """
        SELECT
            COALESCE(vocabulary_id, '')  AS vocabulary_id,
            COALESCE(domain_id, '')      AS domain_id,
            COUNT(*)                     AS total
        FROM read_csv(?, delim='\t', header=true, all_varchar=true)
        GROUP BY vocabulary_id, domain_id
        ORDER BY vocabulary_id, domain_id
        """,
        [str(csv_path)],
    ).fetchall()
    return [
        VocabCoverage(vocabulary_id=r[0], domain_id=r[1], total=int(r[2]), embedded=0)
        for r in rows
    ]
