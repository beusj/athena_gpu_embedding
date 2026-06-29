"""DuckDB-backed query layer over parquet-sharded embedding storage.

The DuckDB connection is opened once per CLI invocation and passed down.
Data is persisted in parquet shards partitioned by model_version.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from glob import glob
from pathlib import Path
from typing import Literal

import duckdb

from gpu_embedder.models import EmbeddedRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StoreContext:
    parquet_root: Path
    legacy_db_path: Path | None = None


_CONTEXTS: dict[int, _StoreContext] = {}


def _resolve_paths(path: Path) -> _StoreContext:
    if path.suffix.lower() == ".duckdb":
        parquet_root = path.with_suffix("")
        legacy_db_path = path if path.exists() and path.is_file() else None
        return _StoreContext(parquet_root=parquet_root, legacy_db_path=legacy_db_path)

    if path.exists() and path.is_file():
        raise ValueError(f"Expected directory path for parquet store, found file: {path}")

    return _StoreContext(parquet_root=path, legacy_db_path=None)


def _get_context(conn: duckdb.DuckDBPyConnection) -> _StoreContext:
    ctx = _CONTEXTS.get(id(conn))
    if ctx is None:
        raise RuntimeError("Store connection context not found")
    return ctx


def _parquet_glob(parquet_root: Path) -> str:
    return str((parquet_root / "model_version=*" / "*.parquet").as_posix())


def _has_parquet_data(parquet_root: Path) -> bool:
    return bool(glob(_parquet_glob(parquet_root)))


def _create_empty_view(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE VIEW concept_embeddings AS
        SELECT
            CAST(NULL AS BIGINT) AS concept_id,
            CAST(NULL AS VARCHAR) AS concept_name,
            CAST(NULL AS VARCHAR) AS domain_id,
            CAST(NULL AS VARCHAR) AS vocabulary_id,
            CAST(NULL AS VARCHAR) AS concept_class_id,
            CAST(NULL AS VARCHAR) AS standard_concept,
            CAST(NULL AS VARCHAR) AS concept_code,
            CAST(NULL AS VARCHAR) AS invalid_reason,
            CAST(NULL AS FLOAT[768]) AS embedding,
            CAST(NULL AS VARCHAR) AS embed_text,
            CAST(NULL AS VARCHAR) AS model_version,
            CAST(NULL AS TIMESTAMP) AS embedded_at
        WHERE FALSE
        """
    )


def _refresh_view(conn: duckdb.DuckDBPyConnection, parquet_root: Path) -> None:
    if not _has_parquet_data(parquet_root):
        _create_empty_view(conn)
        return

    parquet_pattern = _parquet_glob(parquet_root).replace("'", "''")
    conn.execute(
        """
        CREATE OR REPLACE VIEW concept_embeddings AS
        SELECT
            concept_id,
            concept_name,
            domain_id,
            vocabulary_id,
            concept_class_id,
            standard_concept,
            concept_code,
            invalid_reason,
            CAST(embedding AS FLOAT[768]) AS embedding,
            embed_text,
            model_version,
            CAST(embedded_at AS TIMESTAMP) AS embedded_at
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY concept_id, model_version
                    ORDER BY CAST(embedded_at AS TIMESTAMP) DESC, filename DESC
                ) AS _rownum
            FROM read_parquet(
                '__PARQUET_GLOB__',
                hive_partitioning=true,
                union_by_name=true,
                filename=true
            )
        ) dedup
        WHERE _rownum = 1
        """.replace("__PARQUET_GLOB__", parquet_pattern),
    )


def _migrate_legacy_if_needed(conn: duckdb.DuckDBPyConnection, ctx: _StoreContext) -> None:
    if ctx.legacy_db_path is None or _has_parquet_data(ctx.parquet_root):
        return

    logger.info("Migrating legacy DuckDB store %s -> %s", ctx.legacy_db_path, ctx.parquet_root)
    escaped_legacy = str(ctx.legacy_db_path).replace("'", "''")
    conn.execute(f"ATTACH '{escaped_legacy}' AS legacy (READ_ONLY)")
    try:
        has_table = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_catalog = 'legacy'
              AND table_schema = 'main'
              AND table_name = 'concept_embeddings'
            """
        ).fetchone()
        if not has_table or int(has_table[0]) == 0:
            logger.info("No legacy concept_embeddings table found; skipping migration")
            return

        model_versions = conn.execute(
            "SELECT DISTINCT model_version FROM legacy.concept_embeddings"
        ).fetchall()
        if not model_versions:
            logger.info("Legacy concept_embeddings is empty; skipping migration")
            return

        migrated_files = 0
        for row in model_versions:
            model_version = row[0]
            partition_dir = ctx.parquet_root / f"model_version={model_version}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            shard_path = partition_dir / (
                f"migrated-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.parquet"
            )
            escaped_shard = shard_path.as_posix().replace("'", "''")
            conn.execute(
                f"""
                COPY (
                    SELECT
                        concept_id,
                        concept_name,
                        domain_id,
                        vocabulary_id,
                        concept_class_id,
                        standard_concept,
                        concept_code,
                        invalid_reason,
                        embedding,
                        embed_text,
                        model_version,
                        embedded_at
                    FROM legacy.concept_embeddings
                    WHERE model_version = ?
                ) TO '{escaped_shard}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                [model_version],
            )
            migrated_files += 1

        logger.info("Migrated %d model_version shard(s) from legacy DuckDB", migrated_files)
    finally:
        conn.execute("DETACH legacy")


def open_db(path: Path) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB query engine for a parquet-backed store rooted at *path*."""
    ctx = _resolve_paths(path)
    ctx.parquet_root.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(":memory:")
    _CONTEXTS[id(conn)] = ctx
    logger.info("Opened parquet-backed store at %s", ctx.parquet_root)
    return conn


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure view-backed schema over parquet shards (idempotent)."""
    ctx = _get_context(conn)
    ctx.parquet_root.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_if_needed(conn, ctx)
    _refresh_view(conn, ctx.parquet_root)
    logger.debug("Parquet-backed schema ensured")


def get_existing_ids(conn: duckdb.DuckDBPyConnection, model_version: str) -> set[int]:
    """Return concept_ids that already have an embedding for *model_version*."""
    rows = conn.execute(
        "SELECT concept_id FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchall()
    ids = {r[0] for r in rows}
    logger.info("Found %d existing concept_ids for model_version=%s", len(ids), model_version[:8])
    return ids


def _append_rows_as_parquet_shards(conn: duckdb.DuckDBPyConnection, rows: list[EmbeddedRow]) -> None:
    if not rows:
        return

    ctx = _get_context(conn)
    conn.execute("DROP TABLE IF EXISTS temp_embeddings")
    conn.execute(
        """
        CREATE TEMP TABLE temp_embeddings (
            concept_id BIGINT,
            concept_name VARCHAR,
            domain_id VARCHAR,
            vocabulary_id VARCHAR,
            concept_class_id VARCHAR,
            standard_concept VARCHAR,
            concept_code VARCHAR,
            invalid_reason VARCHAR,
            embedding FLOAT[768],
            embed_text VARCHAR,
            model_version VARCHAR,
            embedded_at TIMESTAMP
        )
        """
    )

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
        for r in rows
    ]

    conn.executemany(
        """
        INSERT INTO temp_embeddings (
            concept_id,
            concept_name,
            domain_id,
            vocabulary_id,
            concept_class_id,
            standard_concept,
            concept_code,
            invalid_reason,
            embedding,
            embed_text,
            model_version,
            embedded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    model_versions = conn.execute(
        "SELECT DISTINCT model_version FROM temp_embeddings"
    ).fetchall()
    for row in model_versions:
        model_version = row[0]
        partition_dir = ctx.parquet_root / f"model_version={model_version}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        shard_path = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
        escaped_shard = shard_path.as_posix().replace("'", "''")

        conn.execute(
            f"""
            COPY (
                SELECT
                    concept_id,
                    concept_name,
                    domain_id,
                    vocabulary_id,
                    concept_class_id,
                    standard_concept,
                    concept_code,
                    invalid_reason,
                    embedding,
                    embed_text,
                    model_version,
                    embedded_at
                FROM temp_embeddings
                WHERE model_version = ?
            ) TO '{escaped_shard}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [model_version],
        )

    conn.execute("DROP TABLE IF EXISTS temp_embeddings")
    _refresh_view(conn, ctx.parquet_root)


def upsert_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: list[EmbeddedRow],
    mode: Literal["ndjson", "direct"] = "ndjson",
) -> None:
    """Insert or replace a batch of EmbeddedRow objects.

    Writes append-only parquet shards partitioned by model_version.
    Logical upsert semantics are preserved by the concept_embeddings view,
    which deduplicates by (concept_id, model_version) preferring the latest
    embedded_at value.
    """
    if mode not in {"ndjson", "direct"}:
        raise ValueError(f"Unsupported write mode: {mode}")

    _append_rows_as_parquet_shards(conn, rows)

    logger.info("Upserted %d rows in total (mode=%s)", len(rows), mode)


def count_rows(conn: duckdb.DuckDBPyConnection, model_version: str) -> int:
    """Return the number of stored embeddings for *model_version*."""
    result = conn.execute(
        "SELECT COUNT(*) FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchone()
    return int(result[0]) if result else 0
