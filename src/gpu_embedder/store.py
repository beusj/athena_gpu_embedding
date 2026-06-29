"""Storage layer for local DuckDB tables and parquet-sharded stores.

The DuckDB connection is opened once per CLI invocation and passed down.
`*.duckdb` paths use native DuckDB tables (fast local writes), while directory
paths use parquet shards partitioned by model_version and vocabulary_id.
"""

from __future__ import annotations

import logging
import time
import uuid
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from glob import glob
from math import ceil
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa

from gpu_embedder.models import (
    CSV_FINGERPRINTS_DDL,
    MODEL_VERSION_CACHE_DDL,
    SCHEMA_DDL,
    ConceptRow,
    EmbeddedRow,
)

logger = logging.getLogger(__name__)
TARGET_ROWS_PER_SHARD = 250_000
NULL_VOCAB_PARTITION = "_null"
MODEL_REGISTRY_SUBDIR = Path("_meta") / "model_registry"
EMBEDDING_DIM = 768

# Column order shared by the concept_embeddings table, the parquet shards, and
# the Arrow tables we build for bulk loading.  Keep these in lockstep.
_EMBEDDING_COLUMNS = (
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "invalid_reason",
    "embedding",
    "embed_text",
    "model_version",
    "embedded_at",
)


def _embedded_rows_to_arrow(rows: list[EmbeddedRow]) -> pa.Table:
    """Build a columnar Arrow table from EmbeddedRow objects for bulk loading.

    DuckDB ingests Arrow tables natively (zero-copy, fully vectorised), which is
    dramatically faster than ``executemany`` for the wide ``FLOAT[768]`` column
    — the per-row Python→DuckDB binding is the dominant cost otherwise.  The
    embedding is encoded as a fixed-size list of float32 so it maps directly to
    the ``FLOAT[768]`` column type.
    """
    embeddings = np.asarray([r.embedding for r in rows], dtype=np.float32)
    if embeddings.shape != (len(rows), EMBEDDING_DIM):
        raise ValueError(
            f"Expected embeddings of shape ({len(rows)}, {EMBEDDING_DIM}), "
            f"got {embeddings.shape}"
        )
    embedding_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(embeddings.reshape(-1)), EMBEDDING_DIM
    )
    # DuckDB TIMESTAMP is timezone-naive; drop tzinfo (values are UTC) so the
    # Arrow timestamp maps cleanly without an implicit conversion.
    embedded_at = [
        r.embedded_at.replace(tzinfo=None) if r.embedded_at.tzinfo else r.embedded_at
        for r in rows
    ]
    return pa.table(
        {
            "concept_id": pa.array([r.concept.concept_id for r in rows], type=pa.int64()),
            "concept_name": pa.array([r.concept.concept_name for r in rows], type=pa.string()),
            "domain_id": pa.array([r.concept.domain_id for r in rows], type=pa.string()),
            "vocabulary_id": pa.array([r.concept.vocabulary_id for r in rows], type=pa.string()),
            "concept_class_id": pa.array(
                [r.concept.concept_class_id for r in rows], type=pa.string()
            ),
            "standard_concept": pa.array(
                [r.concept.standard_concept for r in rows], type=pa.string()
            ),
            "concept_code": pa.array([r.concept.concept_code for r in rows], type=pa.string()),
            "invalid_reason": pa.array(
                [r.concept.invalid_reason for r in rows], type=pa.string()
            ),
            "embedding": embedding_arr,
            "embed_text": pa.array([r.embed_text for r in rows], type=pa.string()),
            "model_version": pa.array([r.model_version for r in rows], type=pa.string()),
            "embedded_at": pa.array(embedded_at, type=pa.timestamp("us")),
        }
    )


@dataclass(frozen=True)
class _StoreContext:
    backend: Literal["duckdb", "parquet"]
    db_path: Path | None = None
    parquet_root: Path | None = None


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_version: str
    model_id: str
    model_revision: str | None
    precision: str
    quantization_scheme: str
    recorded_at: datetime


# Keyed by the connection object itself (not id(conn)) via weak references so
# entries are reclaimed automatically when a connection is garbage collected.
# A plain dict keyed by id(conn) leaks entries and can alias a stale, closed
# connection because CPython recycles id() values after collection.
_CONTEXTS: "weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, _StoreContext]" = (
    weakref.WeakKeyDictionary()
)


def _resolve_paths(path: Path) -> _StoreContext:
    if path.suffix.lower() == ".duckdb":
        return _StoreContext(backend="duckdb", db_path=path)

    if path.exists() and path.is_file():
        raise ValueError(f"Expected directory path for parquet store, found file: {path}")

    return _StoreContext(backend="parquet", parquet_root=path)


def _get_context(conn: duckdb.DuckDBPyConnection) -> _StoreContext:
    ctx = _CONTEXTS.get(conn)
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

    # Zero-padded nanosecond stamp used as the leading filename component so a
    # lexical ``filename DESC`` sort is chronological.  The dedup view breaks
    # equal-``embedded_at`` ties by ``filename DESC``; a random UUID would pick
    # an arbitrary (possibly stale) shard, whereas this guarantees the most
    # recently written shard for a concept wins.
    write_seq = f"{time.time_ns():020d}"

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
                shard_path = (
                    partition_dir
                    / f"part-{write_seq}-{shard_idx:05d}-{uuid.uuid4().hex}.parquet"
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
    if ctx.parquet_root is None:
        return

    legacy_db_path = ctx.parquet_root.with_suffix(".duckdb")
    if not legacy_db_path.exists() or not legacy_db_path.is_file() or _has_parquet_data(ctx.parquet_root):
        return

    logger.info("Migrating legacy DuckDB store %s -> %s", legacy_db_path, ctx.parquet_root)
    escaped_legacy = str(legacy_db_path).replace("'", "''")
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
    """Open storage backend for *path*.

    - ``*.duckdb`` paths use native DuckDB table-backed storage (fast local writes).
    - Directory paths use parquet-sharded storage with DuckDB as query layer.
    """
    ctx = _resolve_paths(path)

    if ctx.backend == "duckdb":
        assert ctx.db_path is not None
        ctx.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(ctx.db_path))
        logger.info("Opened duckdb-backed store at %s", ctx.db_path)
    else:
        assert ctx.parquet_root is not None
        ctx.parquet_root.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(":memory:")
        logger.info("Opened parquet-backed store at %s", ctx.parquet_root)

    _CONTEXTS[conn] = ctx
    return conn


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure schema for active backend (idempotent)."""
    ctx = _get_context(conn)

    if ctx.backend == "duckdb":
        conn.execute(SCHEMA_DDL)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_version VARCHAR PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                model_revision VARCHAR,
                precision VARCHAR NOT NULL,
                quantization_scheme VARCHAR NOT NULL,
                recorded_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            ALTER TABLE model_registry
            ADD COLUMN IF NOT EXISTS precision VARCHAR DEFAULT 'fp32'
            """
        )
        conn.execute(
            """
            ALTER TABLE model_registry
            ADD COLUMN IF NOT EXISTS quantization_scheme VARCHAR DEFAULT 'none'
            """
        )
        conn.execute(
            """
            UPDATE model_registry
            SET
                precision = COALESCE(precision, 'fp32'),
                quantization_scheme = COALESCE(quantization_scheme, 'none')
            WHERE precision IS NULL OR quantization_scheme IS NULL
            """
        )
        conn.execute(CSV_FINGERPRINTS_DDL)
        conn.execute(MODEL_VERSION_CACHE_DDL)
        logger.debug("DuckDB-backed schema ensured")
        return

    assert ctx.parquet_root is not None
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


def filter_unembedded_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: list[ConceptRow],
    model_version: str,
) -> list[ConceptRow]:
    """Return only the rows whose concept_id has no embedding for *model_version*.

    Performs the anti-join inside DuckDB rather than materialising all existing
    IDs into a Python set.  This is significantly faster when the store already
    contains millions of rows and only a small fraction remain to embed.
    """
    if not rows:
        return []

    candidate_ids = [r.concept_id for r in rows]
    conn.execute("DROP TABLE IF EXISTS _candidate_ids")
    conn.execute("CREATE TEMP TABLE _candidate_ids (concept_id BIGINT)")
    conn.executemany("INSERT INTO _candidate_ids VALUES (?)", [(i,) for i in candidate_ids])
    result = conn.execute(
        """
        SELECT concept_id
        FROM _candidate_ids
        WHERE concept_id NOT IN (
            SELECT concept_id
            FROM concept_embeddings
            WHERE model_version = ?
        )
        """,
        [model_version],
    ).fetchall()
    conn.execute("DROP TABLE IF EXISTS _candidate_ids")
    unembedded = {r[0] for r in result}
    logger.info(
        "filter_unembedded_rows: %d candidates, %d already embedded, %d to embed "
        "(model_version=%s)",
        len(candidate_ids),
        len(candidate_ids) - len(unembedded),
        len(unembedded),
        model_version[:8],
    )
    return [r for r in rows if r.concept_id in unembedded]


def classify_rows_requiring_embedding(
    conn: duckdb.DuckDBPyConnection,
    rows: list[ConceptRow],
    model_version: str,
    candidate_texts: dict[int, str],
) -> tuple[list[ConceptRow], int, int, int]:
    """Return rows requiring embedding plus (new_count, changed_count, unchanged_count).

    A row requires embedding when either:
    - `(concept_id, model_version)` has no stored embedding, or
    - stored `embed_text` differs from the current candidate text.
    """
    if not rows:
        return ([], 0, 0, 0)

    concept_ids = [row.concept_id for row in rows]
    embed_texts = [candidate_texts[row.concept_id] for row in rows]
    conn.execute("DROP TABLE IF EXISTS _candidate_embed_texts")
    conn.execute(
        """
        CREATE TEMP TABLE _candidate_embed_texts (
            concept_id BIGINT,
            embed_text VARCHAR
        )
        """
    )
    # Use array unnest instead of executemany to avoid per-row Python→DuckDB
    # overhead, which is severe for millions of concepts.
    conn.execute(
        "INSERT INTO _candidate_embed_texts"
        " SELECT unnest(?::BIGINT[]), unnest(?::VARCHAR[])",
        [concept_ids, embed_texts],
    )
    result = conn.execute(
        """
        SELECT
            c.concept_id,
            CASE
                WHEN e.concept_id IS NULL THEN 'new'
                WHEN e.embed_text IS DISTINCT FROM c.embed_text THEN 'changed'
                ELSE 'unchanged'
            END AS status
        FROM _candidate_embed_texts c
        LEFT JOIN concept_embeddings e
          ON e.concept_id = c.concept_id
         AND e.model_version = ?
        """,
        [model_version],
    ).fetchall()
    conn.execute("DROP TABLE IF EXISTS _candidate_embed_texts")

    status_by_id = {int(row[0]): str(row[1]) for row in result}
    need_embed = {
        concept_id
        for concept_id, status in status_by_id.items()
        if status in {"new", "changed"}
    }
    new_count = sum(1 for status in status_by_id.values() if status == "new")
    changed_count = sum(1 for status in status_by_id.values() if status == "changed")
    unchanged_count = sum(1 for status in status_by_id.values() if status == "unchanged")
    logger.info(
        "classify_rows_requiring_embedding: %d candidates, %d new, %d changed, %d unchanged "
        "(model_version=%s)",
        len(rows),
        new_count,
        changed_count,
        unchanged_count,
        model_version[:8],
    )
    return ([row for row in rows if row.concept_id in need_embed], new_count, changed_count, unchanged_count)


def filter_rows_requiring_embedding(
    conn: duckdb.DuckDBPyConnection,
    rows: list[ConceptRow],
    model_version: str,
    candidate_texts: dict[int, str],
) -> list[ConceptRow]:
    """Backward-compatible wrapper returning only rows requiring embedding."""
    to_embed, _, _, _ = classify_rows_requiring_embedding(
        conn,
        rows,
        model_version,
        candidate_texts,
    )
    return to_embed


def _append_rows_as_parquet_shards(
    conn: duckdb.DuckDBPyConnection,
    rows: list[EmbeddedRow],
    *,
    refresh_view: bool,
) -> None:
    if not rows:
        return

    ctx = _get_context(conn)
    if ctx.parquet_root is None:
        raise RuntimeError("Parquet root not available for parquet append")

    # Register the batch as an Arrow table and let DuckDB read it directly,
    # avoiding per-row executemany binding of the FLOAT[768] embedding column.
    arrow_batch = _embedded_rows_to_arrow(rows)
    conn.register("temp_embeddings", arrow_batch)
    try:
        _copy_relation_to_partitioned_shards(
            conn,
            "temp_embeddings",
            ctx.parquet_root,
            log_progress=False,
        )
    finally:
        conn.unregister("temp_embeddings")

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

    if not rows:
        logger.info("Upserted 0 rows in total (mode=%s)", mode)
        return

    ctx = _get_context(conn)

    if ctx.backend == "duckdb":
        # Register the batch as an Arrow table and merge in one SQL statement.
        # DuckDB reads Arrow natively, so this avoids both per-row executemany
        # binding (catastrophically slow for the FLOAT[768] column) and the
        # primary-key index maintenance of a row-by-row insert.
        arrow_batch = _embedded_rows_to_arrow(rows)
        columns = ", ".join(_EMBEDDING_COLUMNS)
        conn.register("_upsert_batch", arrow_batch)
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO concept_embeddings ({columns})
                SELECT {columns} FROM _upsert_batch
                """
            )
        finally:
            conn.unregister("_upsert_batch")
        logger.info("Upserted %d rows in total (mode=%s)", len(rows), mode)
        return

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
    precision: str = "fp32",
    quantization_scheme: str = "none",
) -> None:
    """Upsert model provenance metadata alongside parquet embeddings.

    Registry rows are stored in ``<parquet_root>/_meta/model_registry/*.parquet``
    and deduplicated by ``model_version``.
    """
    ctx = _get_context(conn)

    if ctx.backend == "duckdb":
        conn.execute(
            """
            INSERT OR REPLACE INTO model_registry (
                model_version,
                model_id,
                model_revision,
                precision,
                quantization_scheme,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                model_version,
                model_id,
                model_revision,
                precision,
                quantization_scheme,
                datetime.now(tz=UTC),
            ],
        )
        return

    assert ctx.parquet_root is not None
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
            precision VARCHAR,
            quantization_scheme VARCHAR,
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
            precision,
            quantization_scheme,
            recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [model_version, model_id, model_revision, precision, quantization_scheme, now],
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
                COALESCE(precision, 'fp32') AS precision,
                COALESCE(quantization_scheme, 'none') AS quantization_scheme,
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
                CAST(NULL AS VARCHAR) AS precision,
                CAST(NULL AS VARCHAR) AS quantization_scheme,
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
            precision,
            quantization_scheme,
            recorded_at
        FROM (
            SELECT
                model_version,
                model_id,
                model_revision,
                precision,
                quantization_scheme,
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
                precision,
                quantization_scheme,
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

    if ctx.backend == "duckdb":
        rows = conn.execute(
            """
            SELECT
                model_version,
                model_id,
                model_revision,
                COALESCE(precision, 'fp32') AS precision,
                COALESCE(quantization_scheme, 'none') AS quantization_scheme,
                CAST(recorded_at AS TIMESTAMP) AS recorded_at
            FROM model_registry
            ORDER BY CAST(recorded_at AS TIMESTAMP) DESC, model_version
            """
        ).fetchall()
        return [
            ModelRegistryEntry(
                model_version=str(row[0]),
                model_id=str(row[1]),
                model_revision=str(row[2]) if row[2] is not None else None,
                precision=str(row[3]),
                quantization_scheme=str(row[4]),
                recorded_at=row[5],
            )
            for row in rows
        ]

    assert ctx.parquet_root is not None
    if not _registry_files(ctx.parquet_root):
        return []

    rows = conn.execute(
        """
        SELECT
            model_version,
            model_id,
            model_revision,
            COALESCE(precision, 'fp32') AS precision,
            COALESCE(quantization_scheme, 'none') AS quantization_scheme,
            CAST(recorded_at AS TIMESTAMP) AS recorded_at
        FROM (
            SELECT
                model_version,
                model_id,
                model_revision,
                precision,
                quantization_scheme,
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
            precision=str(row[3]),
            quantization_scheme=str(row[4]),
            recorded_at=row[5],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# CSV fingerprinting
# ---------------------------------------------------------------------------

def get_csv_fingerprint(
    conn: duckdb.DuckDBPyConnection,
    csv_path: str,
    model_version: str,
    filter_hash: str,
) -> dict[str, object] | None:
    """Return the stored fingerprint for *(csv_path, model_version, filter_hash)*, or None."""
    ctx = _get_context(conn)
    if ctx.backend != "duckdb":
        return None  # parquet backend has no fingerprint table
    row = conn.execute(
        """
        SELECT size_bytes, mtime_ns, sha256, row_count, completed_at
        FROM csv_fingerprints
        WHERE csv_path = ? AND model_version = ? AND filter_hash = ?
        """,
        [csv_path, model_version, filter_hash],
    ).fetchone()
    if row is None:
        return None
    return {
        "size_bytes": row[0],
        "mtime_ns": row[1],
        "sha256": row[2],
        "row_count": row[3],
        "completed_at": row[4],
    }


def upsert_csv_fingerprint(
    conn: duckdb.DuckDBPyConnection,
    *,
    csv_path: str,
    model_version: str,
    filter_hash: str,
    size_bytes: int,
    mtime_ns: int,
    sha256: str,
    row_count: int,
) -> None:
    """Record a successful ingest fingerprint for *(csv_path, model_version, filter_hash)*."""
    ctx = _get_context(conn)
    if ctx.backend != "duckdb":
        return  # no-op for parquet backend
    conn.execute(
        """
        INSERT OR REPLACE INTO csv_fingerprints
            (csv_path, model_version, filter_hash, size_bytes, mtime_ns, sha256, row_count, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            csv_path,
            model_version,
            filter_hash,
            size_bytes,
            mtime_ns,
            sha256,
            row_count,
            datetime.now(tz=UTC),
        ],
    )
    logger.info(
        "Upserted csv_fingerprint: path=%s model=%s filter=%s rows=%d",
        csv_path,
        model_version[:12],
        filter_hash[:12],
        row_count,
    )


# ---------------------------------------------------------------------------
# Model version cache
# ---------------------------------------------------------------------------

def get_cached_model_version(
    conn: duckdb.DuckDBPyConnection,
    model_id: str,
    revision: str | None,
) -> str | None:
    """Return a previously stored SHA-256 for *(model_id, revision)*, or None.

    Avoids re-hashing the ~440 MB weights file on every startup when the
    model has not changed.
    """
    ctx = _get_context(conn)
    if ctx.backend != "duckdb":
        return None
    row = conn.execute(
        "SELECT sha256 FROM model_version_cache WHERE model_id = ? AND revision = ?",
        [model_id, revision or ""],
    ).fetchone()
    return str(row[0]) if row is not None else None


def upsert_model_version_cache(
    conn: duckdb.DuckDBPyConnection,
    model_id: str,
    revision: str | None,
    sha256: str,
) -> None:
    """Persist the weights SHA-256 for *(model_id, revision)*."""
    ctx = _get_context(conn)
    if ctx.backend != "duckdb":
        return
    conn.execute(
        """
        INSERT OR REPLACE INTO model_version_cache (model_id, revision, sha256)
        VALUES (?, ?, ?)
        """,
        [model_id, revision or "", sha256],
    )
    logger.info(
        "Upserted model_version_cache: model=%s revision=%s sha256=%s…",
        model_id,
        revision or "default",
        sha256[:12],
    )
