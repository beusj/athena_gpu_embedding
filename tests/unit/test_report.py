"""Unit tests for report.py — uses in-memory DuckDB and the fixture TSV."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from gpu_embedder.models import SCHEMA_DDL, ConceptRow, EmbeddedRow
from gpu_embedder.report import (
    ModelVersionInfo,
    VocabCoverage,
    coverage_report,
    embedded_summary,
    list_model_versions,
)

FIXTURE_TSV = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(SCHEMA_DDL)
    return conn


def _insert_embedded(
    conn: duckdb.DuckDBPyConnection,
    concept_id: int,
    vocabulary_id: str,
    domain_id: str,
    model_version: str = "abc123",
    embedded_at: datetime | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO concept_embeddings (
            concept_id, concept_name, domain_id, vocabulary_id,
            concept_class_id, standard_concept, concept_code,
            invalid_reason, embedding, embed_text, model_version, embedded_at
        ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
        """,
        [
            concept_id,
            f"Concept {concept_id}",
            domain_id,
            vocabulary_id,
            [0.0] * 768,
            f"Concept {concept_id}",
            model_version,
            embedded_at or datetime(2026, 1, 1, tzinfo=UTC),
        ],
    )


# ---------------------------------------------------------------------------
# VocabCoverage properties
# ---------------------------------------------------------------------------


class TestVocabCoverage:
    def test_gap(self) -> None:
        r = VocabCoverage(vocabulary_id="LOINC", domain_id="Measurement", total=100, embedded=70)
        assert r.gap == 30

    def test_pct(self) -> None:
        r = VocabCoverage(vocabulary_id="LOINC", domain_id="Measurement", total=200, embedded=50)
        assert r.pct == pytest.approx(25.0)

    def test_pct_zero_total(self) -> None:
        r = VocabCoverage(vocabulary_id="LOINC", domain_id="Measurement", total=0, embedded=0)
        assert r.pct == 0.0

    def test_fully_embedded_gap_is_zero(self) -> None:
        r = VocabCoverage(vocabulary_id="LOINC", domain_id="Measurement", total=5, embedded=5)
        assert r.gap == 0


# ---------------------------------------------------------------------------
# list_model_versions
# ---------------------------------------------------------------------------


class TestListModelVersions:
    def test_empty_when_no_table(self) -> None:
        conn = duckdb.connect(":memory:")
        result = list_model_versions(conn)
        assert result == []

    def test_empty_when_table_exists_but_empty(self) -> None:
        conn = _mem_conn()
        result = list_model_versions(conn)
        assert result == []

    def test_single_version(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "LOINC", "Measurement", model_version="aaa")
        _insert_embedded(conn, 2, "SNOMED", "Condition", model_version="aaa")
        result = list_model_versions(conn)
        assert len(result) == 1
        assert result[0].model_version == "aaa"
        assert result[0].count == 2

    def test_multiple_versions_sorted_most_recent_first(self) -> None:
        conn = _mem_conn()
        _insert_embedded(
            conn, 1, "LOINC", "Measurement",
            model_version="old", embedded_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
        _insert_embedded(
            conn, 2, "SNOMED", "Condition",
            model_version="new", embedded_at=datetime(2026, 6, 1, tzinfo=UTC)
        )
        result = list_model_versions(conn)
        assert [r.model_version for r in result] == ["new", "old"]

    def test_short_hash_is_first_16_chars(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "LOINC", "Measurement", model_version="abcdef1234567890xyz")
        result = list_model_versions(conn)
        assert result[0].short_hash == "abcdef1234567890"


# ---------------------------------------------------------------------------
# embedded_summary
# ---------------------------------------------------------------------------


class TestEmbeddedSummary:
    def test_empty_when_no_table(self) -> None:
        conn = duckdb.connect(":memory:")
        assert embedded_summary(conn) == []

    def test_empty_when_no_rows(self) -> None:
        conn = _mem_conn()
        assert embedded_summary(conn) == []

    def test_groups_by_vocabulary_and_domain(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "LOINC", "Measurement")
        _insert_embedded(conn, 2, "LOINC", "Measurement")
        _insert_embedded(conn, 3, "SNOMED", "Condition")
        rows = embedded_summary(conn)
        assert len(rows) == 2
        loinc = next(r for r in rows if r.vocabulary_id == "LOINC")
        assert loinc.embedded == 2
        assert loinc.total == 2  # total == embedded in DB-only mode

    def test_filters_by_model_version(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "LOINC", "Measurement", model_version="v1")
        _insert_embedded(conn, 2, "SNOMED", "Condition", model_version="v2")
        rows_v1 = embedded_summary(conn, model_version="v1")
        assert len(rows_v1) == 1
        assert rows_v1[0].vocabulary_id == "LOINC"

    def test_all_versions_when_no_filter(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "LOINC", "Measurement", model_version="v1")
        _insert_embedded(conn, 2, "LOINC", "Measurement", model_version="v2")
        rows = embedded_summary(conn)
        # concept_id 1 and 2 are in same vocab/domain but different versions
        assert len(rows) == 1
        assert rows[0].embedded == 2

    def test_sorted_alphabetically(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 1, "SNOMED", "Condition")
        _insert_embedded(conn, 2, "LOINC", "Measurement")
        rows = embedded_summary(conn)
        assert [r.vocabulary_id for r in rows] == ["LOINC", "SNOMED"]


# ---------------------------------------------------------------------------
# coverage_report
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def test_all_gaps_when_no_table(self) -> None:
        conn = duckdb.connect(":memory:")  # no schema created
        rows = coverage_report(conn, FIXTURE_TSV)
        # All concepts are gaps
        assert all(r.embedded == 0 for r in rows)
        assert sum(r.total for r in rows) > 0

    def test_all_gaps_when_table_empty(self) -> None:
        conn = _mem_conn()
        rows = coverage_report(conn, FIXTURE_TSV)
        assert all(r.embedded == 0 for r in rows)

    def test_partial_coverage(self) -> None:
        conn = _mem_conn()
        # Embed only the two LOINC concepts from the fixture
        # concept_ids: 40481088, 3004249 (both LOINC / Measurement)
        _insert_embedded(conn, 40481088, "LOINC", "Measurement")
        _insert_embedded(conn, 3004249, "LOINC", "Measurement")

        rows = coverage_report(conn, FIXTURE_TSV)
        loinc = next((r for r in rows if r.vocabulary_id == "LOINC"), None)
        assert loinc is not None
        assert loinc.embedded == 2
        # Fixture has 3 LOINC rows (including one invalid)
        assert loinc.total == 3
        assert loinc.gap == 1

    def test_full_coverage_group(self) -> None:
        conn = _mem_conn()
        # Embed all UCUM-style: fixture only has 1 CPT4 row
        _insert_embedded(conn, 999002, "CPT4", "Procedure")
        rows = coverage_report(conn, FIXTURE_TSV)
        cpt4 = next((r for r in rows if r.vocabulary_id == "CPT4"), None)
        assert cpt4 is not None
        assert cpt4.gap == 0
        assert cpt4.pct == pytest.approx(100.0)

    def test_model_version_filter(self) -> None:
        conn = _mem_conn()
        _insert_embedded(conn, 40481088, "LOINC", "Measurement", model_version="v1")
        _insert_embedded(conn, 3004249, "LOINC", "Measurement", model_version="v2")

        rows_v1 = coverage_report(conn, FIXTURE_TSV, model_version="v1")
        loinc = next((r for r in rows_v1 if r.vocabulary_id == "LOINC"), None)
        assert loinc is not None
        assert loinc.embedded == 1  # only v1 concept counted

    def test_groups_sorted_alphabetically(self) -> None:
        conn = _mem_conn()
        rows = coverage_report(conn, FIXTURE_TSV)
        vocab_ids = [r.vocabulary_id for r in rows]
        assert vocab_ids == sorted(vocab_ids)

    def test_total_count_matches_csv(self) -> None:
        conn = _mem_conn()
        rows = coverage_report(conn, FIXTURE_TSV)
        # Fixture has 10 data rows
        assert sum(r.total for r in rows) == 10


# ---------------------------------------------------------------------------
# ModelVersionInfo
# ---------------------------------------------------------------------------


class TestModelVersionInfo:
    def test_short_hash(self) -> None:
        info = ModelVersionInfo(
            model_version="1234567890abcdef" + "x" * 48,
            count=1,
            first_embedded_at=datetime(2026, 1, 1),
            last_embedded_at=datetime(2026, 1, 1),
        )
        assert info.short_hash == "1234567890abcdef"
