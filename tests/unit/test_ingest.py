"""Unit tests for ingest.py — pure CSV parsing and filtering."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gpu_embedder.ingest import (
    compute_csv_fingerprint,
    count_csv_rows,
    filter_rows,
    filter_spec_hash,
    read_csv,
    read_source_parquet,
)
from gpu_embedder.models import ConceptRow, FilterSpec

FIXTURE = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------

class TestReadCsv:
    def test_count_csv_rows(self) -> None:
        assert count_csv_rows(FIXTURE) == 10

    def test_loads_all_rows(self) -> None:
        rows = read_csv(FIXTURE)
        assert len(rows) == 10

    def test_read_csv_with_duckdb_filter_pushdown(self) -> None:
        rows = read_csv(FIXTURE, spec=FilterSpec(vocabulary_ids=["LOINC"]))
        assert len(rows) == 3
        assert all(row.vocabulary_id == "LOINC" for row in rows)

    def test_python_engine_loads_all_rows(self) -> None:
        rows = read_csv(FIXTURE, engine="python")
        assert len(rows) == 10

    def test_python_engine_filter_matches_duckdb(self) -> None:
        duckdb_rows = read_csv(FIXTURE, spec=FilterSpec(vocabulary_ids=["LOINC"]))
        python_rows = read_csv(
            FIXTURE,
            spec=FilterSpec(vocabulary_ids=["LOINC"]),
            engine="python",
        )
        assert python_rows == duckdb_rows

    def test_returns_concept_rows(self) -> None:
        rows = read_csv(FIXTURE)
        assert all(isinstance(r, ConceptRow) for r in rows)

    def test_concept_id_is_int(self) -> None:
        rows = read_csv(FIXTURE)
        assert all(isinstance(r.concept_id, int) for r in rows)

    def test_empty_standard_concept_is_none(self) -> None:
        rows = read_csv(FIXTURE)
        # Rows 6-7 in the fixture have empty standard_concept
        non_standard = [r for r in rows if r.standard_concept is None]
        assert len(non_standard) == 2

    def test_invalid_reason_d_preserved(self) -> None:
        rows = read_csv(FIXTURE)
        invalid_rows = [r for r in rows if r.invalid_reason == "D"]
        assert len(invalid_rows) == 2

    def test_null_invalid_reason_is_none(self) -> None:
        rows = read_csv(FIXTURE)
        valid_rows = [r for r in rows if r.invalid_reason is None]
        # 8 rows have empty invalid_reason in the fixture
        assert len(valid_rows) == 8


# ---------------------------------------------------------------------------
# filter_rows — empty spec (accept all)
# ---------------------------------------------------------------------------

class TestFilterRowsNoFilter:
    def test_empty_spec_returns_all(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec())
        assert len(result) == len(rows)


# ---------------------------------------------------------------------------
# filter_rows — vocabulary_id (OR within column)
# ---------------------------------------------------------------------------

class TestFilterRowsVocabulary:
    def test_single_vocabulary(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(vocabulary_ids=["LOINC"]))
        assert all(r.vocabulary_id == "LOINC" for r in result)
        assert len(result) == 3

    def test_multiple_vocabularies_or(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(vocabulary_ids=["LOINC", "SNOMED"]))
        assert all(r.vocabulary_id in ("LOINC", "SNOMED") for r in result)
        # fixture: 3 LOINC rows + 3 SNOMED rows = 6
        assert len(result) == 6

    def test_nonexistent_vocabulary_returns_empty(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(vocabulary_ids=["ICD10CM"]))
        assert result == []


# ---------------------------------------------------------------------------
# filter_rows — AND across columns
# ---------------------------------------------------------------------------

class TestFilterRowsAndLogic:
    def test_vocabulary_and_domain(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(
            rows,
            FilterSpec(vocabulary_ids=["LOINC"], domain_ids=["Measurement"]),
        )
        assert all(r.vocabulary_id == "LOINC" and r.domain_id == "Measurement" for r in result)
        assert len(result) == 3

    def test_vocabulary_and_standard_concept(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(
            rows,
            FilterSpec(vocabulary_ids=["SNOMED"], standard_concepts=["S"]),
        )
        assert all(r.vocabulary_id == "SNOMED" and r.standard_concept == "S" for r in result)

    def test_all_filters_combined(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(
            rows,
            FilterSpec(
                vocabulary_ids=["LOINC"],
                domain_ids=["Measurement"],
                standard_concepts=["S"],
                invalid_reasons=["valid"],
            ),
        )
        # Only the 2 valid standard LOINC rows
        assert len(result) == 2
        assert all(r.invalid_reason is None for r in result)


# ---------------------------------------------------------------------------
# filter_rows — invalid_reason "valid" shorthand
# ---------------------------------------------------------------------------

class TestFilterRowsInvalidReason:
    def test_valid_shorthand_keeps_null_rows(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(invalid_reasons=["valid"]))
        assert all(r.invalid_reason is None for r in result)
        assert len(result) == 8

    def test_explicit_d_keeps_d_rows(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(invalid_reasons=["D"]))
        assert all(r.invalid_reason == "D" for r in result)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFilterRowsEdgeCases:
    def test_empty_result_does_not_raise(self) -> None:
        rows = read_csv(FIXTURE)
        result = filter_rows(rows, FilterSpec(vocabulary_ids=["DOES_NOT_EXIST"]))
        assert result == []

    def test_does_not_mutate_input(self) -> None:
        rows = read_csv(FIXTURE)
        original_count = len(rows)
        filter_rows(rows, FilterSpec(vocabulary_ids=["LOINC"]))
        assert len(rows) == original_count

    def test_malformed_row_skipped_not_raised(self, tmp_path: Path) -> None:
        """A row with an unparseable concept_id should be skipped, not crash."""
        bad_tsv = tmp_path / "BAD.csv"
        bad_tsv.write_text(
            "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id"
            "\tstandard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason\n"
            "not_an_int\tSome Concept\tCondition\tSNOMED\t"
            "Clinical Finding\tS\t123\t19700101\t20991231\t\n"
            "999\tGood Concept\tCondition\tSNOMED\t"
            "Clinical Finding\tS\t456\t19700101\t20991231\t\n",
            encoding="utf-8",
        )
        rows = read_csv(bad_tsv)
        assert len(rows) == 1
        assert rows[0].concept_id == 999


class TestCsvFingerprintHelpers:
    def test_compute_csv_fingerprint_changes_when_file_changes(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "CONCEPT.csv"
        csv_path.write_text("a\tb\n1\t2\n", encoding="utf-8")

        fp1 = compute_csv_fingerprint(csv_path)
        csv_path.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")
        fp2 = compute_csv_fingerprint(csv_path)

        assert fp1["sha256"] != fp2["sha256"]

    def test_filter_spec_hash_is_order_insensitive_within_columns(self) -> None:
        spec_a = FilterSpec(
            vocabulary_ids=["SNOMED", "LOINC"],
            domain_ids=["Condition", "Measurement"],
        )
        spec_b = FilterSpec(
            vocabulary_ids=["LOINC", "SNOMED"],
            domain_ids=["Measurement", "Condition"],
        )

        assert filter_spec_hash(spec_a) == filter_spec_hash(spec_b)

    def test_filter_spec_hash_changes_when_filter_changes(self) -> None:
        spec_a = FilterSpec(vocabulary_ids=["LOINC"])
        spec_b = FilterSpec(vocabulary_ids=["LOINC", "SNOMED"])

        assert filter_spec_hash(spec_a) != filter_spec_hash(spec_b)

    def test_filter_spec_hash_changes_when_extra_changes(self) -> None:
        spec = FilterSpec(vocabulary_ids=["LOINC"])

        assert filter_spec_hash(spec, extra={"text_fields": ["concept_name"]}) != filter_spec_hash(
            spec,
            extra={"text_fields": ["concept_code", "concept_name"]},
        )


class TestReadSourceParquet:
    def test_reads_source_parquet_and_builds_embed_text(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "source.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": ["SRC_1", "SRC_2"],
                    "source_name": ["Glucose", "mg/dL"],
                    "source_description": ["Blood glucose", "milligram per deciliter"],
                    "source_domain": ["lab_component", "unit"],
                    "ehr_codes": [
                        '[{"system":"LOINC","code":"2345-7"}]',
                        '[{"system":"UCUM","code":"mg/dL"}]',
                    ],
                    "sample_units": ["mg/dL", ""],
                    "sample_values": ["80,95", ""],
                    "data_type": ["numeric", "text"],
                }
            ),
            parquet_path,
        )

        result = read_source_parquet(
            parquet_path,
            namespace="stcm_source",
            text_fields=["concept_name", "source_description", "ehr_codes"],
            separator=" | ",
        )

        assert len(result.rows) == 2
        assert all(isinstance(row, ConceptRow) for row in result.rows)
        first = result.rows[0]
        assert first.namespace == "stcm_source"
        assert first.concept_name == "Glucose"
        assert first.domain_id == "lab_component"
        assert first.vocabulary_id == "LOINC"
        assert first.concept_code == "2345-7"
        assert result.embed_texts[first.concept_id] == "Glucose | Blood glucose | LOINC:2345-7"

    def test_blank_source_name_falls_back_to_source_id(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "source.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": ["SRC_1"],
                    "source_name": [""],
                    "source_description": [None],
                    "source_domain": ["unknown"],
                    "ehr_codes": ["[]"],
                    "sample_units": [None],
                    "sample_values": [None],
                    "data_type": [None],
                }
            ),
            parquet_path,
        )

        result = read_source_parquet(parquet_path, namespace="stcm_source")

        assert len(result.rows) == 1
        row = result.rows[0]
        assert row.concept_name == "SRC_1"
        assert result.embed_texts[row.concept_id] == "SRC_1"
