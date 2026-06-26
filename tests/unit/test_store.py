"""Unit tests for store.py — uses an in-memory DuckDB connection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from gpu_embedder.models import ConceptRow, EmbeddedRow
from gpu_embedder.store import (
    count_rows,
    ensure_schema,
    get_existing_ids,
    open_db,
    upsert_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mem_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


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
    def test_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        conn = open_db(db_path)
        assert db_path.exists()
        conn.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "dir" / "test.duckdb"
        conn = open_db(db_path)
        assert db_path.exists()
        conn.close()


class TestEnsureSchema:
    def test_creates_table(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'concept_embeddings'"
        ).fetchall()
        assert len(result) == 1

    def test_idempotent(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        ensure_schema(conn)  # should not raise


# ---------------------------------------------------------------------------
# upsert_rows + get_existing_ids
# ---------------------------------------------------------------------------

class TestUpsertAndExistence:
    def test_upsert_then_get_ids(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        rows = [_make_row(i) for i in range(1, 4)]
        upsert_rows(conn, rows)
        ids = get_existing_ids(conn, "v1")
        assert ids == {1, 2, 3}

    def test_get_ids_scoped_to_model_version(self) -> None:
        conn = _mem_conn()
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

    def test_upsert_replaces_existing_row(self) -> None:
        conn = _mem_conn()
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

    def test_empty_upsert_does_not_raise(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        upsert_rows(conn, [])  # should be a no-op


# ---------------------------------------------------------------------------
# count_rows
# ---------------------------------------------------------------------------

class TestCountRows:
    def test_zero_before_insert(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        assert count_rows(conn, "v1") == 0

    def test_count_after_insert(self) -> None:
        conn = _mem_conn()
        ensure_schema(conn)
        upsert_rows(conn, [_make_row(i) for i in range(1, 6)])
        assert count_rows(conn, "v1") == 5

    def test_count_scoped_to_model_version(self) -> None:
        conn = _mem_conn()
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
