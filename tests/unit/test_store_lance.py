"""Unit tests for the Lance store backend (store.py).

Skipped entirely when the optional ``pylance`` dependency is not installed.
Uses synthetic vectors only — no model, no GPU, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

pytest.importorskip("lance")

from gpu_embedder.models import SCHEMA_DDL, ConceptRow, EmbeddedRow  # noqa: E402
from gpu_embedder.store import (  # noqa: E402
    LANCE_DATASET_SUBDIR,
    classify_rows_requiring_embedding,
    compact,
    count_embeddings,
    count_rows,
    delete_csv_fingerprints,
    delete_embeddings,
    delete_model_metadata,
    ensure_schema,
    get_cached_model_version,
    get_csv_fingerprint,
    list_model_registry,
    list_vocabulary_counts,
    migrate_duckdb_to_lance,
    open_db,
    refresh_view,
    upsert_csv_fingerprint,
    upsert_model_registry,
    upsert_model_version_cache,
    upsert_rows,
)


def _make_row(
    concept_id: int = 1,
    *,
    model_version: str = "v1",
    vocabulary_id: str | None = "SNOMED",
    embed_text: str | None = None,
    namespace: str = "athena",
) -> EmbeddedRow:
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
            namespace=namespace,
        ),
        embedding=[float(concept_id % 5)] * 768,
        embed_text=embed_text or f"Concept {concept_id}",
        model_version=model_version,
        embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _open(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = open_db(tmp_path / "embeddings.lance")
    ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# open / schema
# ---------------------------------------------------------------------------


class TestOpenAndSchema:
    def test_lance_suffix_creates_container_dir(self, tmp_path: Path) -> None:
        store = tmp_path / "embeddings.lance"
        conn = open_db(store)
        ensure_schema(conn)
        assert store.exists() and store.is_dir()
        conn.close()

    def test_lance_path_that_is_a_file_is_rejected(self, tmp_path: Path) -> None:
        bogus = tmp_path / "oops.lance"
        bogus.write_text("not a directory")
        with pytest.raises(ValueError, match="found file"):
            open_db(bogus)

    def test_empty_store_has_queryable_view(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        # The view exists and is empty before any dataset is written.
        assert count_rows(conn, "v1") == 0
        assert list_model_registry(conn) == []
        conn.close()

    def test_idempotent_ensure_schema(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        ensure_schema(conn)  # second call must not raise
        conn.close()


# ---------------------------------------------------------------------------
# upsert / merge_insert
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_first_upsert_creates_dataset(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1), _make_row(2)])
        assert count_rows(conn, "v1") == 2
        assert (tmp_path / "embeddings.lance" / LANCE_DATASET_SUBDIR).exists()
        conn.close()

    def test_checkpoint_pattern_refresh_false_then_refresh(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1)], refresh_view=False)
        upsert_rows(conn, [_make_row(2)], refresh_view=False)
        refresh_view(conn)
        assert count_rows(conn, "v1") == 2
        conn.close()

    def test_merge_insert_updates_in_place_no_duplicates(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(i) for i in range(1, 6)])
        # Re-embed concept 3 with changed text + add a new concept 6.
        upsert_rows(conn, [_make_row(3, embed_text="CHANGED"), _make_row(6)])
        assert count_rows(conn, "v1") == 6  # 5 + 1 new, concept 3 replaced not duplicated
        text = conn.execute(
            "SELECT embed_text FROM concept_embeddings "
            "WHERE concept_id = 3 AND model_version = 'v1'"
        ).fetchall()
        assert text == [("CHANGED",)]
        conn.close()

    def test_empty_upsert_is_noop(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [])
        assert count_rows(conn, "v1") == 0
        conn.close()

    def test_namespaces_do_not_collide(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(
            conn,
            [_make_row(1, namespace="athena"), _make_row(1, namespace="source")],
        )
        assert count_rows(conn, "v1") == 2
        assert count_rows(conn, "v1", namespace="athena") == 1
        assert count_rows(conn, "v1", namespace="source") == 1
        conn.close()


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_new_changed_unchanged(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1, embed_text="orig"), _make_row(2, embed_text="keep")])
        rows = [
            ConceptRow(concept_id=1, concept_name="c1"),  # changed text
            ConceptRow(concept_id=2, concept_name="c2"),  # unchanged
            ConceptRow(concept_id=3, concept_name="c3"),  # new
        ]
        candidate_texts = {1: "updated", 2: "keep", 3: "brand-new"}
        need, new, changed, unchanged = classify_rows_requiring_embedding(
            conn, rows, "v1", candidate_texts
        )
        assert (new, changed, unchanged) == (1, 1, 1)
        assert {r.concept_id for r in need} == {1, 3}
        conn.close()


# ---------------------------------------------------------------------------
# counts / vocabularies
# ---------------------------------------------------------------------------


class TestCounts:
    def test_count_embeddings_and_vocab_counts(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(
            conn,
            [
                _make_row(1, vocabulary_id="SNOMED"),
                _make_row(2, vocabulary_id="SNOMED"),
                _make_row(3, vocabulary_id="LOINC"),
            ],
        )
        assert count_embeddings(conn, "v1") == 3
        assert count_embeddings(conn, "v1", ["LOINC"]) == 1
        assert dict(list_vocabulary_counts(conn, "v1")) == {"SNOMED": 2, "LOINC": 1}
        conn.close()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_vocabulary(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(
            conn,
            [_make_row(1, vocabulary_id="SNOMED"), _make_row(2, vocabulary_id="LOINC")],
        )
        removed = delete_embeddings(conn, model_version="v1", vocabulary_ids=["LOINC"])
        assert removed == 1
        assert count_rows(conn, "v1") == 1
        assert dict(list_vocabulary_counts(conn, "v1")) == {"SNOMED": 1}
        conn.close()

    def test_delete_whole_model_version(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1), _make_row(2)])
        removed = delete_embeddings(conn, model_version="v1")
        assert removed == 2
        assert count_rows(conn, "v1") == 0
        conn.close()

    def test_delete_when_nothing_matches(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1)])
        assert delete_embeddings(conn, model_version="other") == 0
        conn.close()


# ---------------------------------------------------------------------------
# model registry + no-op meta tables
# ---------------------------------------------------------------------------


class TestRegistryAndMeta:
    def test_registry_roundtrip_stored_under_meta(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_model_registry(
            conn,
            model_version="v1",
            model_id="cambridgeltl/SapBERT",
            model_revision=None,
            pooling="cls",
        )
        upsert_model_registry(
            conn,
            model_version="v2",
            model_id="other/model",
            model_revision="abc",
            pooling="mean",
        )
        entries = {e.model_version: e for e in list_model_registry(conn)}
        assert set(entries) == {"v1", "v2"}
        assert entries["v2"].pooling == "mean"
        assert (tmp_path / "embeddings.lance" / "_meta" / "model_registry").exists()
        conn.close()

    def test_delete_model_metadata_removes_registry_entry(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_model_registry(
            conn, model_version="v1", model_id="m1", model_revision=None
        )
        upsert_model_registry(
            conn, model_version="v2", model_id="m2", model_revision=None
        )
        delete_model_metadata(conn, "v1")
        assert {e.model_version for e in list_model_registry(conn)} == {"v2"}
        conn.close()

    def test_fingerprint_and_version_cache_are_noops(self, tmp_path: Path) -> None:
        # Lance, like parquet, has no fingerprint/version-cache tables; these are
        # safe no-ops (force re-hash / re-read rather than returning stale data).
        conn = _open(tmp_path)
        upsert_csv_fingerprint(
            conn,
            csv_path="/x.csv",
            model_version="v1",
            filter_hash="h",
            size_bytes=1,
            mtime_ns=1,
            sha256="s",
            row_count=1,
        )
        assert get_csv_fingerprint(conn, "/x.csv", "v1", "h") is None
        assert delete_csv_fingerprints(conn, "v1") == 0
        upsert_model_version_cache(conn, "m", None, "cls", "sha")
        assert get_cached_model_version(conn, "m", None, "cls") is None
        conn.close()


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


class TestCompact:
    def test_compact_consolidates_fragments_and_preserves_rows(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        # Several separate writes create several fragments.
        for i in range(4):
            upsert_rows(conn, [_make_row(i)], refresh_view=False)
        refresh_view(conn)
        metrics = compact(conn, cleanup_older_than_days=0)
        assert metrics["fragments_removed"] >= metrics["fragments_added"]
        assert count_rows(conn, "v1") == 4
        conn.close()

    def test_compact_no_cleanup_keeps_versions(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(1)])
        metrics = compact(conn, cleanup_older_than_days=None)
        assert metrics["versions_removed"] == 0
        conn.close()

    def test_compact_empty_store_is_noop(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        metrics = compact(conn)
        assert metrics["fragments_removed"] == 0
        conn.close()

    def test_compact_rejects_non_lance_backend(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "store.duckdb")
        ensure_schema(conn)
        with pytest.raises(ValueError, match="lance backend"):
            compact(conn)
        conn.close()


# ---------------------------------------------------------------------------
# export (lance -> parquet) via the generic view
# ---------------------------------------------------------------------------


class TestExport:
    def test_copy_view_to_parquet(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        upsert_rows(conn, [_make_row(i) for i in range(1, 4)])
        out = tmp_path / "export.parquet"
        conn.execute(
            f"COPY (SELECT * FROM concept_embeddings) TO '{out.as_posix()}' (FORMAT PARQUET)"
        )
        n = (
            duckdb.connect(":memory:")
            .execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')")
            .fetchone()
        )
        assert n is not None and n[0] == 3
        conn.close()


# ---------------------------------------------------------------------------
# migration (duckdb -> lance)
# ---------------------------------------------------------------------------


def _make_legacy_duckdb(path: Path, n: int, *, with_namespace: bool = True) -> None:
    legacy = duckdb.connect(str(path))
    legacy.execute(SCHEMA_DDL)
    from gpu_embedder import store as st

    arrow = st._embedded_rows_to_arrow([_make_row(100 + i) for i in range(n)])
    legacy.register("_b", arrow)
    cols = ", ".join(st._EMBEDDING_COLUMNS)
    legacy.execute(f"INSERT INTO concept_embeddings ({cols}) SELECT {cols} FROM _b")
    legacy.close()


class TestMigrate:
    def test_streaming_migration_and_idempotency(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.duckdb"
        _make_legacy_duckdb(legacy, 1500)
        conn = _open(tmp_path)
        migrated = migrate_duckdb_to_lance(conn, legacy, batch_rows=500)
        assert migrated == 1500
        assert count_rows(conn, "v1") == 1500
        # Re-running is a no-op (target already populated).
        assert migrate_duckdb_to_lance(conn, legacy) == 0
        conn.close()

    def test_post_migration_embed_merges_without_schema_clash(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.duckdb"
        _make_legacy_duckdb(legacy, 10)
        conn = _open(tmp_path)
        migrate_duckdb_to_lance(conn, legacy)
        # The canonical schema must match so a subsequent embed merge_insert works.
        upsert_rows(conn, [_make_row(100, embed_text="POST")])
        assert count_rows(conn, "v1") == 10
        text = conn.execute(
            "SELECT embed_text FROM concept_embeddings WHERE concept_id = 100"
        ).fetchone()
        assert text == ("POST",)
        conn.close()

    def test_migration_missing_legacy_file_raises(self, tmp_path: Path) -> None:
        conn = _open(tmp_path)
        with pytest.raises(FileNotFoundError):
            migrate_duckdb_to_lance(conn, tmp_path / "nope.duckdb")
        conn.close()

    def test_migration_empty_legacy_table(self, tmp_path: Path) -> None:
        legacy = tmp_path / "empty.duckdb"
        lcon = duckdb.connect(str(legacy))
        lcon.execute(SCHEMA_DDL)
        lcon.close()
        conn = _open(tmp_path)
        assert migrate_duckdb_to_lance(conn, legacy) == 0
        assert count_rows(conn, "v1") == 0
        conn.close()

    def test_migration_legacy_without_concept_table(self, tmp_path: Path) -> None:
        legacy = tmp_path / "other.duckdb"
        lcon = duckdb.connect(str(legacy))
        lcon.execute("CREATE TABLE unrelated (x INTEGER)")
        lcon.close()
        conn = _open(tmp_path)
        assert migrate_duckdb_to_lance(conn, legacy) == 0
        assert count_rows(conn, "v1") == 0
        conn.close()

    def test_migration_rejects_non_lance_target(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.duckdb"
        _make_legacy_duckdb(legacy, 1)
        conn = open_db(tmp_path / "store.duckdb")
        ensure_schema(conn)
        with pytest.raises(ValueError, match="lance store path"):
            migrate_duckdb_to_lance(conn, legacy)
        conn.close()
