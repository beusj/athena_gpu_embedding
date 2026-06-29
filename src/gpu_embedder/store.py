"""DuckDB-backed query layer over parquet-sharded embedding storage.

The DuckDB connection is opened once per CLI invocation and passed down.
Data is persisted in parquet shards partitioned by model_version.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from glob import glob
from math import ceil
from pathlib import Path
from typing import Literal

import duckdb

from gpu_embedder.models import EmbeddedRow

logger = logging.getLogger(__name__)
TARGET_ROWS_PER_SHARD = 250_000
NULL_VOCAB_PARTITION = "_null"
MODEL_REGISTRY_SUBDIR = Path("_meta") / "model_registry"


@dataclass(frozen=True)
class _StoreContext:
    parquet_root: Path
    legacy_db_path: Path | None = None


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_version: str
    model_id: str
    model_revision: str | None
    recorded_at: datetime


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


def _parquet_patterns(parquet_root: Path) -> list[str]:
    return [
        str((parquet_root / "model_version=*" / "*.parquet").as_posix()),
        str((parquet_root / "model_version=*" / "vocabulary_id=*" / "*.parquet").as_posix()),
    ]


def _registry_dir(parquet_root: Path) -> Path:
    return parquet_root / MODEL_REGISTRY_SUBDIR


def _registry_pattern(parquet_root: Path) -> str:
    return str((_registry_dir(parquet_root) / "*.parquet").as_posix())


def _registry_files(parquet_root: Path) -> list[Path]:
    return sorted(_registry_dir(parquet_root).glob("*.parquet"))


def _existing_parquet_patterns(parquet_root: Path) -> list[str]:
    existing: list[str] = []
    for pattern in _parquet_patterns(parquet_root):
        if glob(pattern):
            existing.append(pattern)
    return existing


def _has_parquet_data(parquet_root: Path) -> bool:
    return bool(_existing_parquet_patterns(parquet_root))


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
    patterns = _existing_parquet_patterns(parquet_root)
    if not patterns:
        _create_empty_view(conn)
        return

    parquet_sources = "\nUNION ALL\n".join(
        """
            SELECT *
            FROM read_parquet(
                '__PARQUET_GLOB__',
                hive_partitioning=true,
                union_by_name=true,
                filename=true
            )
        """.replace("__PARQUET_GLOB__", pattern.replace("'", "''"))
        for pattern in patterns
    )
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
            FROM (
__PARQUET_SOURCES__
            ) AS all_parquet
        ) dedup
        WHERE _rownum = 1
        """.replace("__PARQUET_SOURCES__", parquet_sources),
    )


def _copy_relation_to_partitioned_shards(
    conn: duckdb.DuckDBPyConnection,
    source_relation: str,
    parquet_root: Path,
    *,
    log_progress: bool = False,
) -> int:
    partitions = conn.execute(
        f"""
        SELECT
            model_version,
            COALESCE(vocabulary_id, ?) AS vocabulary_partition,
            vocabulary_id,
            COUNT(*) AS row_count
        FROM {source_relation}
        GROUP BY model_version, COALESCE(vocabulary_id, ?), vocabulary_id
        """,
        [NULL_VOCAB_PARTITION, NULL_VOCAB_PARTITION],
    ).fetchall()

    if not partitions:
        return 0

    total_rows = sum(int(row_count) for _, _, _, row_count in partitions)
    total_partitions = len(partitions)
    estimated_total_files = sum(ceil(int(row_count) / TARGET_ROWS_PER_SHARD) for *_, row_count in partitions)

    total_files = 0
    completed_rows = 0
    completed_partitions = 0
    started_at = time.perf_counter()
    next_log_at = started_at + 30.0

    if log_progress:
        logger.info(
            (
                "Migration planning: source=%s, partitions=%d, rows=%d, "
                "target_rows_per_shard=%d, estimated_files=%d"
            ),
            source_relation,
            total_partitions,
            total_rows,
            TARGET_ROWS_PER_SHARD,
            estimated_total_files,
        )

    for model_version, vocabulary_partition, vocabulary_id, row_count in partitions:
        partition_row_count = int(row_count)
        partition_dir = (
            parquet_root
            / f"model_version={model_version}"
            / f"vocabulary_id={vocabulary_partition}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        shard_count = ceil(partition_row_count / TARGET_ROWS_PER_SHARD)
        conn.execute("DROP TABLE IF EXISTS temp_partition_ranked")
        conn.execute(
            f"""
            CREATE TEMP TABLE temp_partition_ranked AS
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
                embedded_at,
                ROW_NUMBER() OVER (ORDER BY concept_id) AS rn
            FROM {source_relation}
            WHERE model_version = ?
              AND (
                    (? = ? AND vocabulary_id IS NULL)
                    OR vocabulary_id = ?
                  )
            """,
            [
                model_version,
                vocabulary_partition,
                NULL_VOCAB_PARTITION,
                vocabulary_id,
            ],
        )

        try:
            for shard_idx in range(shard_count):
                start_rn = shard_idx * TARGET_ROWS_PER_SHARD + 1
                end_rn = min((shard_idx + 1) * TARGET_ROWS_PER_SHARD, partition_row_count)
                shard_path = partition_dir / f"part-{shard_idx:05d}-{uuid.uuid4().hex}.parquet"
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
                        FROM temp_partition_ranked
                        WHERE rn BETWEEN ? AND ?
                    ) TO '{escaped_shard}'
                    (FORMAT PARQUET, COMPRESSION SNAPPY)
                    """,
                    [start_rn, end_rn],
                )
                total_files += 1
        finally:
            conn.execute("DROP TABLE IF EXISTS temp_partition_ranked")

        completed_rows += partition_row_count
        completed_partitions += 1

        if log_progress:
            now = time.perf_counter()
            if (
                completed_partitions == 1
                or completed_partitions == total_partitions
                or now >= next_log_at
            ):
                elapsed = max(now - started_at, 1e-9)
                rows_per_sec = completed_rows / elapsed
                remaining_rows = max(total_rows - completed_rows, 0)
                eta_minutes = (remaining_rows / rows_per_sec / 60.0) if rows_per_sec > 0 else 0.0
                pct = (completed_rows / total_rows * 100.0) if total_rows else 100.0
                logger.info(
                    (
                        "Migration progress: %d/%d partitions (%.1f%%), "
                        "%d/%d rows, files=%d, rows_per_sec=%.0f, eta_minutes=%.1f"
                    ),
                    completed_partitions,
                    total_partitions,
                    pct,
                    completed_rows,
                    total_rows,
                    total_files,
                    rows_per_sec,
                    eta_minutes,
                )
                next_log_at = now + 30.0

    return total_files


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

        row_count = conn.execute("SELECT COUNT(*) FROM legacy.concept_embeddings").fetchone()
        if not row_count or int(row_count[0]) == 0:
            logger.info("Legacy concept_embeddings is empty; skipping migration")
            return

        migrated_files = _copy_relation_to_partitioned_shards(
            conn,
            "legacy.concept_embeddings",
            ctx.parquet_root,
            log_progress=True,
        )

        logger.info("Migrated legacy DuckDB into %d parquet shard(s)", migrated_files)
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


def _append_rows_as_parquet_shards(
    conn: duckdb.DuckDBPyConnection,
    rows: list[EmbeddedRow],
    *,
    refresh_view: bool,
) -> None:
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

    _copy_relation_to_partitioned_shards(
        conn,
        "temp_embeddings",
        ctx.parquet_root,
        log_progress=False,
    )

    conn.execute("DROP TABLE IF EXISTS temp_embeddings")
    if refresh_view:
        _refresh_view(conn, ctx.parquet_root)


def upsert_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: list[EmbeddedRow],
    mode: Literal["ndjson", "direct"] = "ndjson",
    *,
    refresh_view: bool = True,
) -> None:
    """Insert or replace a batch of EmbeddedRow objects.

    Writes append-only parquet shards partitioned by model_version.
    Logical upsert semantics are preserved by the concept_embeddings view,
    which deduplicates by (concept_id, model_version) preferring the latest
    embedded_at value.
    """
    if mode not in {"ndjson", "direct"}:
        raise ValueError(f"Unsupported write mode: {mode}")

    _append_rows_as_parquet_shards(conn, rows, refresh_view=refresh_view)

    logger.info("Upserted %d rows in total (mode=%s)", len(rows), mode)


def count_rows(conn: duckdb.DuckDBPyConnection, model_version: str) -> int:
    """Return the number of stored embeddings for *model_version*."""
    result = conn.execute(
        "SELECT COUNT(*) FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchone()
    return int(result[0]) if result else 0


def upsert_model_registry(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    model_id: str,
    model_revision: str | None,
) -> None:
    """Upsert model provenance metadata alongside parquet embeddings.

    Registry rows are stored in ``<parquet_root>/_meta/model_registry/*.parquet``
    and deduplicated by ``model_version``.
    """
    ctx = _get_context(conn)
    registry_dir = _registry_dir(ctx.parquet_root)
    registry_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=UTC)
    conn.execute("DROP TABLE IF EXISTS temp_model_registry_new")
    conn.execute(
        """
        CREATE TEMP TABLE temp_model_registry_new (
            model_version VARCHAR,
            model_id VARCHAR,
            model_revision VARCHAR,
            recorded_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO temp_model_registry_new (
            model_version,
            model_id,
            model_revision,
            recorded_at
        ) VALUES (?, ?, ?, ?)
        """,
        [model_version, model_id, model_revision, now],
    )

    registry_files = _registry_files(ctx.parquet_root)
    if registry_files:
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE temp_model_registry_existing AS
            SELECT
                model_version,
                model_id,
                model_revision,
                CAST(recorded_at AS TIMESTAMP) AS recorded_at
            FROM read_parquet(
                '__REGISTRY_PATTERN__',
                union_by_name=true
            )
            """.replace(
                "__REGISTRY_PATTERN__",
                _registry_pattern(ctx.parquet_root).replace("'", "''"),
            )
        )
    else:
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE temp_model_registry_existing AS
            SELECT
                CAST(NULL AS VARCHAR) AS model_version,
                CAST(NULL AS VARCHAR) AS model_id,
                CAST(NULL AS VARCHAR) AS model_revision,
                CAST(NULL AS TIMESTAMP) AS recorded_at
            WHERE FALSE
            """
        )

    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE temp_model_registry_merged AS
        SELECT
            model_version,
            model_id,
            model_revision,
            recorded_at
        FROM (
            SELECT
                model_version,
                model_id,
                model_revision,
                recorded_at,
                ROW_NUMBER() OVER (
                    PARTITION BY model_version
                    ORDER BY recorded_at DESC
                ) AS _rownum
            FROM (
                SELECT * FROM temp_model_registry_existing
                UNION ALL
                SELECT * FROM temp_model_registry_new
            ) all_rows
        ) ranked
        WHERE _rownum = 1
        """
    )

    for file_path in registry_files:
        file_path.unlink()

    registry_path = registry_dir / f"part-{uuid.uuid4().hex}.parquet"
    conn.execute(
        """
        COPY (
            SELECT
                model_version,
                model_id,
                model_revision,
                recorded_at
            FROM temp_model_registry_merged
            ORDER BY model_version
        ) TO '__REGISTRY_PATH__'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
        """.replace("__REGISTRY_PATH__", registry_path.as_posix().replace("'", "''"))
    )

    conn.execute("DROP TABLE IF EXISTS temp_model_registry_existing")
    conn.execute("DROP TABLE IF EXISTS temp_model_registry_new")
    conn.execute("DROP TABLE IF EXISTS temp_model_registry_merged")


def list_model_registry(conn: duckdb.DuckDBPyConnection) -> list[ModelRegistryEntry]:
    """Return model registry entries stored alongside parquet shards."""
    ctx = _get_context(conn)
    if not _registry_files(ctx.parquet_root):
        return []

    rows = conn.execute(
        """
        SELECT
            model_version,
            model_id,
            model_revision,
            CAST(recorded_at AS TIMESTAMP) AS recorded_at
        FROM (
            SELECT
                model_version,
                model_id,
                model_revision,
                recorded_at,
                ROW_NUMBER() OVER (
                    PARTITION BY model_version
                    ORDER BY CAST(recorded_at AS TIMESTAMP) DESC
                ) AS _rownum
            FROM read_parquet(
                ?,
                union_by_name=true
            )
        ) ranked
        WHERE _rownum = 1
        ORDER BY CAST(recorded_at AS TIMESTAMP) DESC, model_version
        """,
        [_registry_pattern(ctx.parquet_root)],
    ).fetchall()

    return [
        ModelRegistryEntry(
            model_version=str(row[0]),
            model_id=str(row[1]),
            model_revision=str(row[2]) if row[2] is not None else None,
            recorded_at=row[3],
        )
        for row in rows
    ]
