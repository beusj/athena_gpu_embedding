"""Unit tests for store.py — parquet-backed store behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from gpu_embedder.models import SCHEMA_DDL, ConceptRow, EmbeddedRow
from gpu_embedder.store import (
    classify_rows_requiring_embedding,
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
    open_db,
    upsert_csv_fingerprint,
    upsert_model_registry,
    upsert_model_version_cache,
    upsert_rows,
)


def _embedded_ids(conn: duckdb.DuckDBPyConnection, model_version: str) -> set[int]:
    """Test helper: concept_ids stored for *model_version* (via the public view)."""
    rows = conn.execute(
        "SELECT concept_id FROM concept_embeddings WHERE model_version = ?",
        [model_version],
    ).fetchall()
    return {r[0] for r in rows}


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


def _make_concept_row(concept_id: int = 1, vocabulary_id: str = "SNOMED") -> ConceptRow:
    return ConceptRow(
        concept_id=concept_id,
        concept_name=f"Concept {concept_id}",
        domain_id="Condition",
        vocabulary_id=vocabulary_id,
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code=str(concept_id),
        invalid_reason=None,
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
        assert _embedded_ids(conn, "v1") == {101}
        migrated_files = list((legacy_path.with_suffix("")).glob("model_version=*/**/*.parquet"))
        assert migrated_files
        conn.close()


# ---------------------------------------------------------------------------
# upsert_rows + existence
# ---------------------------------------------------------------------------

class TestUpsertAndExistence:
    def test_upsert_then_get_ids(self, tmp_path: Path) -> None:
        store_root = tmp_path / "embeddings"
        conn = open_db(store_root)
        ensure_schema(conn)
        rows = [_make_row(i) for i in range(1, 4)]
        upsert_rows(conn, rows)
        assert _embedded_ids(conn, "v1") == {1, 2, 3}
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
        assert _embedded_ids(conn, "v1") == {10}
        assert _embedded_ids(conn, "v2") == {10}
        assert _embedded_ids(conn, "v3") == set()

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
# classify_rows_requiring_embedding
# ---------------------------------------------------------------------------

class TestClassifyRowsRequiringEmbedding:
    def test_new_rows_require_embedding(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        rows = [_make_concept_row(1), _make_concept_row(2)]
        candidate_texts = {1: "Concept 1", 2: "Concept 2"}

        to_embed, new, changed, unchanged = classify_rows_requiring_embedding(
            conn, rows, "v1", candidate_texts
        )

        assert {r.concept_id for r in to_embed} == {1, 2}
        assert (new, changed, unchanged) == (2, 0, 0)

    def test_unchanged_rows_are_skipped(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        upsert_rows(conn, [_make_row(concept_id=7)])
        rows = [_make_concept_row(7)]
        candidate_texts = {7: "Concept 7"}

        to_embed, new, changed, unchanged = classify_rows_requiring_embedding(
            conn, rows, "v1", candidate_texts
        )

        assert to_embed == []
        assert (new, changed, unchanged) == (0, 0, 1)

    def test_changed_embed_text_requires_reembed(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        upsert_rows(conn, [_make_row(concept_id=9)])
        rows = [_make_concept_row(9)]
        candidate_texts = {9: "Concept 9 (updated)"}

        to_embed, new, changed, unchanged = classify_rows_requiring_embedding(
            conn, rows, "v1", candidate_texts
        )

        assert [r.concept_id for r in to_embed] == [9]
        assert (new, changed, unchanged) == (0, 1, 0)


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


class TestCsvFingerprints:
    def test_get_returns_none_when_missing(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)

        result = get_csv_fingerprint(conn, "C:/tmp/CONCEPT.csv", "v1", "f1")

        assert result is None

    def test_upsert_then_get_round_trip(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)

        upsert_csv_fingerprint(
            conn,
            csv_path="C:/tmp/CONCEPT.csv",
            model_version="v1",
            filter_hash="f1",
            size_bytes=123,
            mtime_ns=456,
            sha256="abc",
            row_count=789,
        )

        result = get_csv_fingerprint(conn, "C:/tmp/CONCEPT.csv", "v1", "f1")

        assert result is not None
        assert result["size_bytes"] == 123
        assert result["mtime_ns"] == 456
        assert result["sha256"] == "abc"
        assert result["row_count"] == 789

    def test_upsert_replaces_same_pk(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)

        upsert_csv_fingerprint(
            conn,
            csv_path="C:/tmp/CONCEPT.csv",
            model_version="v1",
            filter_hash="f1",
            size_bytes=100,
            mtime_ns=200,
            sha256="old",
            row_count=300,
        )
        upsert_csv_fingerprint(
            conn,
            csv_path="C:/tmp/CONCEPT.csv",
            model_version="v1",
            filter_hash="f1",
            size_bytes=101,
            mtime_ns=201,
            sha256="new",
            row_count=301,
        )

        result = get_csv_fingerprint(conn, "C:/tmp/CONCEPT.csv", "v1", "f1")

        assert result is not None
        assert result["size_bytes"] == 101
        assert result["mtime_ns"] == 201
        assert result["sha256"] == "new"
        assert result["row_count"] == 301


# ---------------------------------------------------------------------------
# cleanup / deletion
# ---------------------------------------------------------------------------


def _make_row_mv(
    concept_id: int,
    vocabulary_id: str,
    model_version: str,
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
        ),
        embedding=[0.0] * 768,
        embed_text=f"Concept {concept_id}",
        model_version=model_version,
        embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _seed_multi(conn: duckdb.DuckDBPyConnection) -> None:
    upsert_rows(
        conn,
        [
            _make_row_mv(1, "SNOMED", "mvA"),
            _make_row_mv(2, "SNOMED", "mvA"),
            _make_row_mv(3, "LOINC", "mvA"),
            _make_row_mv(4, "RxNorm", "mvA"),
            _make_row_mv(5, "SNOMED", "mvB"),
        ],
    )


class TestListVocabularyCounts:
    def test_counts_per_vocabulary(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        counts = dict(list_vocabulary_counts(conn, "mvA"))

        assert counts == {"SNOMED": 2, "LOINC": 1, "RxNorm": 1}
        conn.close()

    def test_empty_for_unknown_model(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        assert list_vocabulary_counts(conn, "does-not-exist") == []
        conn.close()


class TestCountEmbeddings:
    def test_counts_whole_model(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        assert count_embeddings(conn, "mvA") == 4
        conn.close()

    def test_counts_selected_vocabularies(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        assert count_embeddings(conn, "mvA", ["SNOMED", "LOINC"]) == 3
        conn.close()


class TestDeleteEmbeddings:
    def test_delete_selected_vocabularies_only(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        deleted = delete_embeddings(conn, model_version="mvA", vocabulary_ids=["SNOMED"])

        assert deleted == 2
        assert dict(list_vocabulary_counts(conn, "mvA")) == {"LOINC": 1, "RxNorm": 1}
        # Other model versions are untouched.
        assert count_rows(conn, "mvB") == 1
        conn.close()

    def test_delete_whole_model(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        deleted = delete_embeddings(conn, model_version="mvA", vocabulary_ids=None)

        assert deleted == 4
        assert count_rows(conn, "mvA") == 0
        assert count_rows(conn, "mvB") == 1
        conn.close()

    def test_delete_no_match_returns_zero(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        _seed_multi(conn)

        assert delete_embeddings(conn, model_version="mvA", vocabulary_ids=["ICD10CM"]) == 0
        assert count_rows(conn, "mvA") == 4
        conn.close()

    def test_delete_whole_model_parquet_backend(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        _seed_multi(conn)

        deleted = delete_embeddings(conn, model_version="mvA", vocabulary_ids=None)

        assert deleted == 4
        assert count_rows(conn, "mvA") == 0
        assert count_rows(conn, "mvB") == 1
        conn.close()

    def test_delete_vocabularies_parquet_backend(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings")
        ensure_schema(conn)
        _seed_multi(conn)

        deleted = delete_embeddings(conn, model_version="mvA", vocabulary_ids=["SNOMED"])

        assert deleted == 2
        assert dict(list_vocabulary_counts(conn, "mvA")) == {"LOINC": 1, "RxNorm": 1}
        assert count_rows(conn, "mvB") == 1
        conn.close()


class TestDeleteCsvFingerprints:
    def test_removes_only_matching_model(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        upsert_csv_fingerprint(
            conn,
            csv_path="/tmp/CONCEPT.csv",
            model_version="mvA",
            filter_hash="f1",
            size_bytes=1,
            mtime_ns=1,
            sha256="a",
            row_count=1,
        )
        upsert_csv_fingerprint(
            conn,
            csv_path="/tmp/CONCEPT.csv",
            model_version="mvB",
            filter_hash="f1",
            size_bytes=1,
            mtime_ns=1,
            sha256="b",
            row_count=1,
        )

        removed = delete_csv_fingerprints(conn, "mvA")

        assert removed == 1
        assert get_csv_fingerprint(conn, "/tmp/CONCEPT.csv", "mvA", "f1") is None
        assert get_csv_fingerprint(conn, "/tmp/CONCEPT.csv", "mvB", "f1") is not None
        conn.close()


class TestDeleteModelMetadata:
    def test_removes_registry_and_cache(self, tmp_path: Path) -> None:
        conn = open_db(tmp_path / "embeddings.duckdb")
        ensure_schema(conn)
        upsert_model_registry(
            conn,
            model_version="mvA",
            model_id="FremyCompany/BioLORD-2023",
            model_revision=None,
        )
        upsert_model_version_cache(conn, "FremyCompany/BioLORD-2023", None, "cls", "mvA")

        delete_model_metadata(conn, "mvA")

        assert all(e.model_version != "mvA" for e in list_model_registry(conn))
        assert get_cached_model_version(conn, "FremyCompany/BioLORD-2023", None) is None
        conn.close()
