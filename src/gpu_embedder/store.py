"""DuckDB store: schema management, existence checks, and upserts.

The DuckDB connection is opened once per CLI invocation and passed down.
Modules do not open their own connections.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from gpu_embedder.models import SCHEMA_DDL, EmbeddedRow

logger = logging.getLogger(__name__)


def open_db(path: Path) -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    logger.info("Opened DuckDB at %s", path)
    return conn


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the concept_embeddings table if it does not already exist.

    Safe to call multiple times (idempotent).
    """
    conn.execute(SCHEMA_DDL)
    logger.debug("Schema ensured")


def get_existing_ids(conn: duckdb.DuckDBPyConnection, model_version: str) -> set[int]:
    """Return concept_ids that already have an embedding for *model_version*."""
    rows = conn.execute(
        "SELECT concept_id FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchall()
    ids = {r[0] for r in rows}
    logger.info("Found %d existing concept_ids for model_version=%s", len(ids), model_version[:8])
    return ids


def upsert_rows(conn: duckdb.DuckDBPyConnection, rows: list[EmbeddedRow]) -> None:
    """Insert or replace a batch of EmbeddedRow objects.

    Uses a staged merge strategy for better DuckDB performance on large vector
    payloads:
    1) insert incoming rows into a temporary table,
    2) delete matching (concept_id, model_version) keys from target,
    3) insert staged rows into target.

    This avoids the per-row conflict handling overhead of INSERT OR REPLACE.
    """
    if not rows:
        return

    conn.execute("BEGIN")
    try:
        conn.execute("DROP TABLE IF EXISTS temp_embeddings")
        conn.execute(
            """
            CREATE TEMP TABLE temp_embeddings AS
            SELECT * FROM concept_embeddings WHERE 1=0
            """
        )

        chunk_size = 1024
        for chunk_start in range(0, len(rows), chunk_size):
            chunk = rows[chunk_start : chunk_start + chunk_size]
            records = [
                (
                    r.concept.concept_id,
                    r.concept.concept_name,
                    r.concept.domain_id,
                    r.concept.vocabulary_id,
                    r.concept.concept_class_id,
                    r.concept.standard_concept,
                    r.concept.concept_code,
                    r.concept.invalid_reason,
                    r.embedding,
                    r.embed_text,
                    r.model_version,
                    r.embedded_at,
                )
                for r in chunk
            ]

            conn.executemany(
                """
                INSERT INTO temp_embeddings (
                    concept_id, concept_name, domain_id, vocabulary_id,
                    concept_class_id, standard_concept, concept_code,
                    invalid_reason, embedding, embed_text, model_version, embedded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            logger.debug(
                "Staged chunk of %d rows (%d/%d)",
                len(chunk),
                chunk_start + len(chunk),
                len(rows),
            )

        conn.execute(
            """
            DELETE FROM concept_embeddings AS t
            USING temp_embeddings AS s
            WHERE t.concept_id = s.concept_id
              AND t.model_version = s.model_version
            """
        )

        conn.execute(
            """
            INSERT INTO concept_embeddings
            SELECT * FROM temp_embeddings
            """
        )

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("DROP TABLE IF EXISTS temp_embeddings")

    logger.info("Upserted %d rows in total", len(rows))


def count_rows(conn: duckdb.DuckDBPyConnection, model_version: str) -> int:
    """Return the number of stored embeddings for *model_version*."""
    result = conn.execute(
        "SELECT COUNT(*) FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchone()
    return int(result[0]) if result else 0
