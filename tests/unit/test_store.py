"""Unit tests for store.py — parquet-backed store behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from gpu_embedder.models import SCHEMA_DDL, ConceptRow, EmbeddedRow
from gpu_embedder.store import (
    count_rows,
    ensure_schema,
    get_existing_ids,
    list_model_registry,
    open_db,
    upsert_model_registry,
    upsert_rows,
)


def _make_row(concept_id: int = 1, vocabulary_id: str = "SNOMED") -> EmbeddedRow:
    return EmbeddedRow(
        concept=ConceptRow(
            concept_id=concept_id,
            concept_name=f"Concept {concept_id}",
            domain_id="Condition",
            vocabulary_id=vocabulary_id,
            concept_class_id="Clinical Finding",
            standard_concept="S",
            concept_code=str(concept_id),
            invalid_reason=None,
        ),
        embedding=[0.0] * 768,
        embed_text=f"Concept {concept_id}",
        model_version="v1",
        embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------

class TestOpenDb:
    def test_creates_parquet_root_for_directory_path(self, tmp_path: Path) -> None:
        store_path = tmp_path / "embeddings"
        conn = open_db(store_path)
        ensure_schema(conn)
        assert store_path.exists()
        assert store_path.is_dir()
        conn.close()

    def test_duckdb_suffix_uses_native_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "store.duckdb"
        conn = open_db(db_path)
        ensure_schema(conn)
        assert db_path.exists()
        assert db_path.is_file()
        conn.close()


class TestEnsureSchema:
    def test_creates_view(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        result = conn.execute(
            "SELECT table_name FROM information_schema.views "
            "WHERE table_name = 'concept_embeddings'"
        ).fetchall()
        assert len(result) == 1

    def test_idempotent(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        ensure_schema(conn)  # should not raise

    def test_migrates_legacy_duckdb_table_when_opening_parquet_root(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "legacy.duckdb"
        legacy = duckdb.connect(str(legacy_path))
        legacy.execute(SCHEMA_DDL)
        row = _make_row(concept_id=101)
        legacy.execute(
            """
            INSERT INTO concept_embeddings (
                concept_id, concept_name, domain_id, vocabulary_id,
                concept_class_id, standard_concept, concept_code,
                invalid_reason, embedding, embed_text, model_version, embedded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row.concept.concept_id,
                row.concept.concept_name,
                row.concept.domain_id,
                row.concept.vocabulary_id,
                row.concept.concept_class_id,
                row.concept.standard_concept,
                row.concept.concept_code,
                row.concept.invalid_reason,
                row.embedding,
                row.embed_text,
                row.model_version,
                row.embedded_at,
            ],
        )
        legacy.close()

        conn = open_db(legacy_path.with_suffix(""))
        ensure_schema(conn)
        assert get_existing_ids(conn, "v1") == {101}
        migrated_files = list((legacy_path.with_suffix("")).glob("model_version=*/**/*.parquet"))
        assert migrated_files
        conn.close()


# ---------------------------------------------------------------------------
# upsert_rows + get_existing_ids
# ---------------------------------------------------------------------------

class TestUpsertAndExistence:
    def test_upsert_then_get_ids(self, tmp_path: Path) -> None:
        store_root = tmp_path / "embeddings"
        conn = open_db(store_root)
        ensure_schema(conn)
        rows = [_make_row(i) for i in range(1, 4)]
        upsert_rows(conn, rows)
        ids = get_existing_ids(conn, "v1")
        assert ids == {1, 2, 3}
        shards = list(store_root.glob("model_version=*/vocabulary_id=*/*.parquet"))
        assert shards

    def test_get_ids_scoped_to_model_version(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        row_v1 = _make_row(concept_id=10)
        row_v2 = EmbeddedRow(
            concept=row_v1.concept,
            embedding=row_v1.embedding,
            embed_text=row_v1.embed_text,
            model_version="v2",
            embedded_at=row_v1.embedded_at,
        )
        upsert_rows(conn, [row_v1])
        upsert_rows(conn, [row_v2])
        assert get_existing_ids(conn, "v1") == {10}
        assert get_existing_ids(conn, "v2") == {10}
        assert get_existing_ids(conn, "v3") == set()

    def test_upsert_replaces_existing_row(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        row = _make_row(concept_id=5)
        upsert_rows(conn, [row])
        # Change the embed_text and upsert again
        updated = EmbeddedRow(
            concept=row.concept,
            embedding=[1.0] * 768,
            embed_text="updated text",
            model_version="v1",
            embedded_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        upsert_rows(conn, [updated])
        result = conn.execute(
            "SELECT embed_text FROM concept_embeddings WHERE concept_id = 5"
        ).fetchone()
        assert result is not None
        assert result[0] == "updated text"
        # Should still be only 1 row (not 2)
        count = conn.execute(
            "SELECT COUNT(*) FROM concept_embeddings WHERE concept_id = 5"
        ).fetchone()
        assert count[0] == 1

    def test_empty_upsert_does_not_raise(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        upsert_rows(conn, [])  # should be a no-op

    def test_upsert_ndjson_preserves_numeric_like_concept_code_as_text(
        self,
        tmp_path: Path,
    ) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        row = _make_row(concept_id=42)
        row.concept.concept_code = "2764601000001104"

        upsert_rows(conn, [row], mode="ndjson")

        result = conn.execute(
            "SELECT concept_code FROM concept_embeddings WHERE concept_id = 42"
        ).fetchone()
        assert result is not None
        assert result[0] == "2764601000001104"


# ---------------------------------------------------------------------------
# count_rows
# ---------------------------------------------------------------------------

class TestCountRows:
    def test_zero_before_insert(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        assert count_rows(conn, "v1") == 0

    def test_count_after_insert(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        upsert_rows(conn, [_make_row(i) for i in range(1, 6)])
        assert count_rows(conn, "v1") == 5

    def test_count_scoped_to_model_version(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        rows_v1 = [_make_row(i) for i in range(1, 4)]
        rows_v2 = [
            EmbeddedRow(
                concept=r.concept,
                embedding=r.embedding,
                embed_text=r.embed_text,
                model_version="v2",
                embedded_at=r.embedded_at,
            )
            for r in rows_v1[:2]
        ]
        upsert_rows(conn, rows_v1)
        upsert_rows(conn, rows_v2)
        assert count_rows(conn, "v1") == 3
        assert count_rows(conn, "v2") == 2


class TestModelRegistry:
    def test_writes_registry_row_to_meta_parquet(self, tmp_path: Path) -> None:
        store_root = tmp_path / "embeddings"
        conn = open_db(store_root)
        ensure_schema(conn)

        upsert_model_registry(
            conn,
            model_version="abc123",
            model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            model_revision="090663c3",
        )

        rows = conn.execute(
            """
            SELECT model_version, model_id, model_revision, precision, quantization_scheme
            FROM read_parquet(?, union_by_name=true)
            """,
            [str((store_root / "_meta" / "model_registry" / "*.parquet").as_posix())],
        ).fetchall()

        assert rows == [
            (
                "abc123",
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                "090663c3",
                "fp32",
                "none",
            )
        ]

    def test_upsert_replaces_existing_model_version_row(self, tmp_path: Path) -> None:
        store_root = tmp_path / "embeddings"
        conn = open_db(store_root)
        ensure_schema(conn)

        upsert_model_registry(
            conn,
            model_version="abc123",
            model_id="model/one",
            model_revision="rev1",
        )
        upsert_model_registry(
            conn,
            model_version="abc123",
            model_id="model/two",
            model_revision="rev2",
        )

        rows = conn.execute(
            """
            SELECT model_version, model_id, model_revision, precision, quantization_scheme
            FROM read_parquet(?, union_by_name=true)
            """,
            [str((store_root / "_meta" / "model_registry" / "*.parquet").as_posix())],
        ).fetchall()

        assert rows == [("abc123", "model/two", "rev2", "fp32", "none")]

    def test_list_model_registry_returns_latest_rows(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)

        upsert_model_registry(
            conn,
            model_version="v1",
            model_id="model/one",
            model_revision="rev1",
        )
        upsert_model_registry(
            conn,
            model_version="v2",
            model_id="model/two",
            model_revision=None,
        )

        rows = list_model_registry(conn)

        assert len(rows) == 2
        versions = {r.model_version for r in rows}
        assert versions == {"v1", "v2"}
        v2_row = next(r for r in rows if r.model_version == "v2")
        assert v2_row.model_id == "model/two"
        assert v2_row.model_revision is None
        assert v2_row.precision == "fp32"
        assert v2_row.quantization_scheme == "none"
