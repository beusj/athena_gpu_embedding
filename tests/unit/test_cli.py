"""Unit tests for cli.py command behavior."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from gpu_embedder.cli import app
from gpu_embedder.ingest import SourceParquetRows
from gpu_embedder.models import DEFAULT_VOCABULARY_IDS, ConceptRow, EmbeddedRow
from gpu_embedder.store import (
    ensure_schema,
    list_model_registry,
    open_db,
    upsert_model_registry,
    upsert_rows,
)


def _seed_embeddings_db(db_path: Path) -> None:
    conn = open_db(db_path)
    ensure_schema(conn)
    upsert_rows(
        conn,
        [
            EmbeddedRow(
                concept=ConceptRow(
                    concept_id=999002,
                    concept_name="CPT4 test concept",
                    domain_id="Procedure",
                    vocabulary_id="CPT4",
                ),
                embedding=[0.0] * 768,
                embed_text="CPT4 test concept",
                model_version="v1",
                embedded_at=datetime.now(tz=UTC),
            )
        ],
    )
    conn.close()


def test_cpt4_loads_api_key_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    vocab_dir = tmp_path / "athena_vocab"
    vocab_dir.mkdir()
    jar_path = vocab_dir / "cpt4.jar"
    jar_path.write_bytes(b"fake jar")
    (tmp_path / ".env").write_text(
        "UMLS_API_KEY=test-key\n"
        "GPU_EMBED_VOCAB_DIR=athena_vocab\n"
        "CPT4_JAR=athena_vocab/cpt4.jar\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_which(cmd: str) -> str:
        assert cmd == "java"
        return "java"

    def fake_run(cmd: list[str], check: bool, cwd: Path) -> None:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    monkeypatch.delenv("CPT4_JAR", raising=False)
    monkeypatch.delenv("GPU_EMBED_VOCAB_DIR", raising=False)
    monkeypatch.setattr("gpu_embedder.cli.shutil.which", fake_which)
    monkeypatch.setattr("gpu_embedder.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["cpt4"])

    assert result.exit_code == 0
    assert captured["check"] is True
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == ["java", "-Dumls-apikey=test-key", "-jar"]
    assert Path(cmd[3]).name == "cpt4.jar"
    assert cmd[4] == "5"
    assert captured["cwd"] == vocab_dir.resolve()


def test_cpt4_uses_java_home_when_java_not_on_path(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    vocab_dir = tmp_path / "athena_vocab"
    vocab_dir.mkdir()
    jar_path = vocab_dir / "cpt4.jar"
    jar_path.write_bytes(b"fake jar")
    java_home = tmp_path / "jdk"
    java_bin = java_home / "bin"
    java_bin.mkdir(parents=True)
    java_exe = java_bin / ("java.exe" if __import__("os").name == "nt" else "java")
    java_exe.write_bytes(b"fake java")
    (tmp_path / ".env").write_text(
        "UMLS_API_KEY=test-key\n"
        "GPU_EMBED_VOCAB_DIR=athena_vocab\n"
        "CPT4_JAR=athena_vocab/cpt4.jar\n"
        f"JAVA_HOME={java_home}\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_which(cmd: str) -> str | None:
        assert cmd == "java"
        return None

    def fake_run(cmd: list[str], check: bool, cwd: Path) -> None:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    monkeypatch.delenv("CPT4_JAR", raising=False)
    monkeypatch.delenv("GPU_EMBED_VOCAB_DIR", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("gpu_embedder.cli.shutil.which", fake_which)
    monkeypatch.setattr("gpu_embedder.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["cpt4"])

    assert result.exit_code == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert Path(cmd[0]).name in {"java", "java.exe"}
    assert Path(cmd[0]).resolve() == java_exe.resolve()


def test_coverage_shows_complete_section_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings.duckdb"
    _seed_embeddings_db(db_path)

    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"
    result = runner.invoke(app, ["coverage", str(fixture), "--db", str(db_path)])

    assert result.exit_code == 0
    assert "Groups With Gaps" in result.output
    assert "Fully Embedded Groups" in result.output
    assert "CPT4" in result.output


def test_coverage_gaps_only_hides_complete_section(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings.duckdb"
    _seed_embeddings_db(db_path)

    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"
    result = runner.invoke(app, ["coverage", str(fixture), "--db", str(db_path), "--gaps-only"])

    assert result.exit_code == 0
    assert "Groups With Gaps" in result.output
    assert "Fully Embedded Groups" not in result.output


def test_coverage_writes_csv_output(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings.duckdb"
    _seed_embeddings_db(db_path)
    csv_path = tmp_path / "coverage" / "report.csv"

    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"
    result = runner.invoke(
        app,
        ["coverage", str(fixture), "--db", str(db_path), "--csv", str(csv_path)],
    )

    assert result.exit_code == 0
    assert csv_path.exists()
    assert "Coverage CSV written" in result.output

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "vocabulary_id",
        "domain_id",
        "source_count",
        "embedded_count",
        "gap_count",
        "coverage_pct",
    ]
    assert any(row[0] == "CPT4" for row in rows[1:])


def test_embed_upserts_every_n_batches(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    rows = [
        ConceptRow(
            concept_id=i,
            concept_name=f"Concept {i}",
            domain_id="Drug",
            vocabulary_id="NDC",
            concept_class_id="Drug",
            standard_concept="S",
            concept_code=str(i),
            invalid_reason=None,
        )
        for i in range(1, 6)
    ]

    upsert_sizes: list[int] = []

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        return rows

    def fake_load_model(model_id: str, device: str, revision: str | None = None):
        return object(), object()

    def fake_compute_model_version(model_id: str, revision: str | None = None) -> str:
        return "test-model-version"

    def fake_embed_all(
        chunk_rows,
        model,
        tokenizer,
        device,
        batch_size,
        max_length,
        text_fields,
        separator,
        model_version,
        *,
        precomputed_texts=None,
    ):
        return [
            EmbeddedRow(
                concept=r,
                embedding=[0.0] * 768,
                embed_text=r.concept_name,
                model_version=model_version,
                embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for r in chunk_rows
        ]

    def fake_open_db(path: Path):
        return _FakeConn()

    def fake_ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
        return None

    def fake_classify_rows_requiring_embedding(  # type: ignore[no-untyped-def]
        conn,
        rows,
        model_version: str,
        candidate_texts,
    ):
        return rows, len(rows), 0, 0  # all rows are new in this test

    def fake_upsert_model_registry(  # type: ignore[no-untyped-def]
        conn,
        *,
        model_version: str,
        model_id: str,
        model_revision: str | None,
        precision: str = "fp32",
        quantization_scheme: str = "none",
    ) -> None:
        return None

    def fake_upsert_rows(  # type: ignore[no-untyped-def]
        conn,
        embedded_rows,
        mode: str = "ndjson",
        *,
        refresh_view: bool = True,
    ) -> None:
        upsert_sizes.append(len(embedded_rows))

    def fake_count_rows(conn, model_version: str, namespace: str | None = None) -> int:  # type: ignore[no-untyped-def]
        return sum(upsert_sizes)

    def fake_list_model_registry(conn):  # type: ignore[no-untyped-def]
        return []

    def fake_get_csv_fingerprint(conn, csv_path: str, model_version: str, filter_hash: str):  # type: ignore[no-untyped-def]
        return None

    def fake_upsert_csv_fingerprint(  # type: ignore[no-untyped-def]
        conn,
        *,
        csv_path: str,
        model_version: str,
        filter_hash: str,
        size_bytes: int,
        mtime_ns: int,
        sha256: str,
        row_count: int,
    ) -> None:
        return None

    def fake_get_cached_model_version(conn, model_id: str, revision) -> None:  # type: ignore[no-untyped-def]
        return None  # always miss — compute_model_version (also mocked) is called

    def fake_upsert_model_version_cache(conn, model_id: str, revision, sha256: str) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)
    monkeypatch.setattr("gpu_embedder.embed.load_model", fake_load_model)
    monkeypatch.setattr("gpu_embedder.embed.compute_model_version", fake_compute_model_version)
    monkeypatch.setattr("gpu_embedder.embed.embed_all", fake_embed_all)
    monkeypatch.setattr("gpu_embedder.store.open_db", fake_open_db)
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", fake_ensure_schema)
    monkeypatch.setattr(
        "gpu_embedder.store.classify_rows_requiring_embedding",
        fake_classify_rows_requiring_embedding,
    )
    monkeypatch.setattr("gpu_embedder.store.upsert_model_registry", fake_upsert_model_registry)
    monkeypatch.setattr("gpu_embedder.store.upsert_rows", fake_upsert_rows)
    monkeypatch.setattr("gpu_embedder.store.count_rows", fake_count_rows)
    monkeypatch.setattr("gpu_embedder.store.list_model_registry", fake_list_model_registry)
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", fake_get_csv_fingerprint)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", fake_upsert_csv_fingerprint)
    monkeypatch.setattr("gpu_embedder.store.get_cached_model_version", fake_get_cached_model_version)
    monkeypatch.setattr("gpu_embedder.store.upsert_model_version_cache", fake_upsert_model_version_cache)
    monkeypatch.setattr("gpu_embedder.store.refresh_view", lambda conn: None)

    result = runner.invoke(
        app,
        [
            "embed",
            str(fixture),
            "--batch-size",
            "2",
            "--upsert-every-batches",
            "2",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert upsert_sizes == [4, 1]


def test_embed_persists_fingerprint_when_nothing_new_to_embed(monkeypatch) -> None:
    """Regression: a CSV that was read but yields no rows to embed must still

    have its fingerprint recorded, otherwise the file is re-read and re-hashed
    on every subsequent run (the diff-detection optimization never converges).
    """
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    rows = [
        ConceptRow(
            concept_id=i,
            concept_name=f"Concept {i}",
            domain_id="Drug",
            vocabulary_id="NDC",
        )
        for i in range(1, 4)
    ]

    recorded_fingerprints: list[str] = []

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        return rows

    def fake_load_model(model_id: str, device: str, revision: str | None = None):
        return object(), object()

    def fake_compute_model_version(model_id: str, revision: str | None = None) -> str:
        return "test-model-version"

    def fake_open_db(path: Path):
        return _FakeConn()

    def fake_ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
        return None

    def fake_classify_rows_requiring_embedding(  # type: ignore[no-untyped-def]
        conn,
        rows,
        model_version: str,
        candidate_texts,
    ):
        # Everything is already embedded with the current text -> nothing to do.
        return [], 0, 0, len(rows)

    def fake_upsert_model_registry(conn, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def fake_get_csv_fingerprint(conn, csv_path: str, model_version: str, filter_hash: str):  # type: ignore[no-untyped-def]
        return None  # no stored fingerprint -> file is (re-)read this run

    def fake_upsert_csv_fingerprint(conn, *, csv_path: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        recorded_fingerprints.append(csv_path)

    def fake_get_cached_model_version(conn, model_id: str, revision):  # type: ignore[no-untyped-def]
        return "test-model-version"

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)
    monkeypatch.setattr("gpu_embedder.embed.load_model", fake_load_model)
    monkeypatch.setattr("gpu_embedder.embed.compute_model_version", fake_compute_model_version)
    monkeypatch.setattr("gpu_embedder.store.open_db", fake_open_db)
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", fake_ensure_schema)
    monkeypatch.setattr(
        "gpu_embedder.store.classify_rows_requiring_embedding",
        fake_classify_rows_requiring_embedding,
    )
    monkeypatch.setattr("gpu_embedder.store.upsert_model_registry", fake_upsert_model_registry)
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", fake_get_csv_fingerprint)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", fake_upsert_csv_fingerprint)
    monkeypatch.setattr("gpu_embedder.store.get_cached_model_version", fake_get_cached_model_version)
    monkeypatch.setattr("gpu_embedder.store.count_rows", lambda conn, model_version, namespace=None: 0)
    monkeypatch.setattr("gpu_embedder.store.list_model_registry", lambda conn: [])

    result = runner.invoke(app, ["embed", str(fixture), "--device", "cpu"])

    assert result.exit_code == 0
    assert "Nothing new to embed" in result.output
    assert recorded_fingerprints == [str(fixture.resolve())]


def test_embed_accepts_comma_delimited_vocabulary_id(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    captured: dict[str, object] = {}

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["vocabulary_ids"] = spec.vocabulary_ids
        captured["engine"] = engine
        return []

    monkeypatch.setattr("gpu_embedder.store.open_db", lambda path: _FakeConn())
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "gpu_embedder.store.get_cached_model_version",
        lambda conn, model_id, revision: "test-model-version",
    )
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", lambda conn, csv_path, model_version, filter_hash: None)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", lambda conn, **kwargs: None)

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)

    result = runner.invoke(
        app,
        ["embed", str(fixture), "--vocabulary-id", "LOINC,SNOMED"],
    )

    assert result.exit_code == 0
    assert captured["vocabulary_ids"] == ["LOINC", "SNOMED"]
    assert "Nothing to embed." in result.output


def test_embed_accepts_mixed_repeat_and_comma_vocabulary_id(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    captured: dict[str, object] = {}

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        captured["vocabulary_ids"] = spec.vocabulary_ids
        return []

    monkeypatch.setattr("gpu_embedder.store.open_db", lambda path: _FakeConn())
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "gpu_embedder.store.get_cached_model_version",
        lambda conn, model_id, revision: "test-model-version",
    )
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", lambda conn, csv_path, model_version, filter_hash: None)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", lambda conn, **kwargs: None)

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)

    result = runner.invoke(
        app,
        [
            "embed",
            str(fixture),
            "--vocabulary-id",
            "LOINC,SNOMED",
            "--vocabulary-id",
            "RxNorm",
        ],
    )

    assert result.exit_code == 0
    assert captured["vocabulary_ids"] == ["LOINC", "SNOMED", "RxNorm"]


def test_embed_defaults_to_highest_yield_vocabularies(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    captured: dict[str, object] = {}

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        captured["vocabulary_ids"] = spec.vocabulary_ids
        return []

    monkeypatch.setattr("gpu_embedder.store.open_db", lambda path: _FakeConn())
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "gpu_embedder.store.get_cached_model_version",
        lambda conn, model_id, revision: "test-model-version",
    )
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", lambda conn, csv_path, model_version, filter_hash: None)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", lambda conn, **kwargs: None)

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)

    result = runner.invoke(app, ["embed", str(fixture)])

    assert result.exit_code == 0
    assert captured["vocabulary_ids"] == list(DEFAULT_VOCABULARY_IDS)
    assert "defaulting to highest-yield vocabularies" in result.output


def test_embed_all_sentinel_embeds_every_vocabulary(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    captured: dict[str, object] = {}

    class _FakeConn:
        pass

    def fake_read_csv(path: Path, spec, engine: str, namespace: str = "athena"):  # type: ignore[no-untyped-def]
        captured["vocabulary_ids"] = spec.vocabulary_ids
        return []

    monkeypatch.setattr("gpu_embedder.store.open_db", lambda path: _FakeConn())
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "gpu_embedder.store.get_cached_model_version",
        lambda conn, model_id, revision: "test-model-version",
    )
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", lambda conn, csv_path, model_version, filter_hash: None)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", lambda conn, **kwargs: None)

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)

    result = runner.invoke(app, ["embed", str(fixture), "--vocabulary-id", "all"])

    assert result.exit_code == 0
    assert captured["vocabulary_ids"] == []
    assert "Embedding all vocabularies" in result.output


def test_embed_source_parquet_uses_source_adapter_defaults(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    source_dir = tmp_path / "source_concepts"
    source_dir.mkdir()
    parquet_path = source_dir / "wave.parquet"
    parquet_path.write_bytes(b"not-real-parquet")

    captured: dict[str, object] = {}

    class _FakeConn:
        pass

    def fake_open_db(path: Path):
        return _FakeConn()

    def fake_ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
        return None

    def fake_get_cached_model_version(conn, model_id: str, revision):  # type: ignore[no-untyped-def]
        return "test-model-version"

    def fake_get_csv_fingerprint(conn, csv_path: str, model_version: str, filter_hash: str):  # type: ignore[no-untyped-def]
        return None

    def fake_read_source_parquet(path: Path, *, namespace: str, text_fields, separator: str):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["namespace"] = namespace
        captured["text_fields"] = list(text_fields)
        captured["separator"] = separator
        return SourceParquetRows(rows=[], embed_texts={})

    monkeypatch.setattr("gpu_embedder.store.open_db", fake_open_db)
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", fake_ensure_schema)
    monkeypatch.setattr("gpu_embedder.store.get_cached_model_version", fake_get_cached_model_version)
    monkeypatch.setattr("gpu_embedder.store.get_csv_fingerprint", fake_get_csv_fingerprint)
    monkeypatch.setattr("gpu_embedder.store.upsert_csv_fingerprint", lambda conn, **kwargs: None)
    monkeypatch.setattr("gpu_embedder.cli.read_source_parquet", fake_read_source_parquet)

    result = runner.invoke(
        app,
        [
            "embed",
            "--source-parquet",
            str(source_dir),
            "--source-namespace",
            "stcm_source",
            "--source-text-field",
            "concept_name",
            "--source-text-field",
            "source_description",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == parquet_path
    assert captured["namespace"] == "stcm_source"
    assert captured["text_fields"] == ["concept_name", "source_description"]
    assert "Loaded 0 rows from source parquet input." in result.output
    assert "Nothing to embed." in result.output


def test_embed_source_parquet_rejects_athena_filters(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    parquet_path = tmp_path / "wave.parquet"
    parquet_path.write_bytes(b"not-real-parquet")

    result = runner.invoke(
        app,
        [
            "embed",
            "--source-parquet",
            str(parquet_path),
            "--vocabulary-id",
            "LOINC",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 2
    assert "Athena filter options are not supported with --source-parquet" in result.output


def test_migrate_store_invokes_store_initialization(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class _FakeConn:
        pass

    def fake_open_db(path: Path):
        calls.append(f"open:{path}")
        return _FakeConn()

    def fake_ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
        calls.append("ensure")

    monkeypatch.setattr("gpu_embedder.store.open_db", fake_open_db)
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", fake_ensure_schema)

    result = runner.invoke(app, ["migrate-store", "--db", str(tmp_path / "legacy.duckdb")])

    assert result.exit_code == 0
    assert calls[0] == f"open:{tmp_path / 'legacy'}"
    assert "ensure" in calls


def test_export_writes_sharded_parquet_by_vocabulary(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings.duckdb"
    out_dir = tmp_path / "parquet"

    conn = open_db(db_path)
    ensure_schema(conn)

    now = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        EmbeddedRow(
            concept=ConceptRow(
                concept_id=1,
                concept_name="SNOMED A",
                domain_id="Condition",
                vocabulary_id="SNOMED",
            ),
            embedding=[0.1] * 768,
            embed_text="SNOMED A",
            model_version="v1",
            embedded_at=now,
        ),
        EmbeddedRow(
            concept=ConceptRow(
                concept_id=2,
                concept_name="SNOMED B",
                domain_id="Condition",
                vocabulary_id="SNOMED",
            ),
            embedding=[0.2] * 768,
            embed_text="SNOMED B",
            model_version="v1",
            embedded_at=now,
        ),
        EmbeddedRow(
            concept=ConceptRow(
                concept_id=3,
                concept_name="SNOMED C",
                domain_id="Condition",
                vocabulary_id="SNOMED",
            ),
            embedding=[0.3] * 768,
            embed_text="SNOMED C",
            model_version="v1",
            embedded_at=now,
        ),
        EmbeddedRow(
            concept=ConceptRow(
                concept_id=4,
                concept_name="LOINC A",
                domain_id="Measurement",
                vocabulary_id="LOINC",
            ),
            embedding=[0.4] * 768,
            embed_text="LOINC A",
            model_version="v1",
            embedded_at=now,
        ),
    ]

    upsert_rows(conn, rows)
    conn.close()

    result = runner.invoke(
        app,
        [
            "export",
            str(out_dir),
            "--db",
            str(db_path),
            "--model-version",
            "v1",
            "--shard-rows",
            "2",
        ],
    )

    assert result.exit_code == 0

    snomed_files = sorted((out_dir / "SNOMED").glob("*.parquet"))
    loinc_files = sorted((out_dir / "LOINC").glob("*.parquet"))

    assert len(snomed_files) == 2
    assert len(loinc_files) == 1

    from duckdb import connect

    verify_conn = connect()
    snomed_count = verify_conn.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(out_dir / "SNOMED" / "*.parquet")],
    ).fetchone()
    loinc_count = verify_conn.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(out_dir / "LOINC" / "*.parquet")],
    ).fetchone()
    verify_conn.close()

    assert snomed_count is not None
    assert loinc_count is not None
    assert snomed_count[0] == 3
    assert loinc_count[0] == 1


def test_model_registry_lists_entries(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings"

    conn = open_db(db_path)
    ensure_schema(conn)
    upsert_model_registry(
        conn,
        model_version="a" * 64,
        model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        model_revision="090663c3",
    )
    conn.close()

    result = runner.invoke(app, ["model-registry", "--db", str(db_path)])

    assert result.exit_code == 0
    assert "MODEL VERSION" in result.output
    assert "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" in result.output
    assert "090663c3" in result.output


def test_model_registry_backfills_from_logs(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "gpu-embed-2026-06-28.log").write_text(
        "2026-06-28 09:00:00 INFO gpu_embedder.embed: "
        "Loading model from FremyCompany/BioLORD-2023 → device=cuda "
        "(FP32, revision=167aab527b238a50ca65224e6319215d2ff4fc9f, source=cached)\n",
        encoding="utf-8",
    )

    def fake_compute_model_version(model_id: str, revision: str | None = None) -> str:
        assert model_id == "FremyCompany/BioLORD-2023"
        assert revision == "167aab527b238a50ca65224e6319215d2ff4fc9f"
        return "f8c969586cc6b0fd393faa7d879de93d6cc532123956041fefd3474194322050"

    monkeypatch.setattr("gpu_embedder.embed.compute_model_version", fake_compute_model_version)

    result = runner.invoke(
        app,
        [
            "model-registry",
            "--db",
            str(db_path),
            "--backfill-from-logs",
            "--log-dir",
            str(log_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Backfill complete from logs" in result.output
    assert "FremyCompany/BioLORD-2023" in result.output
    assert "f8c969586cc6b0fd" in result.output


# ---------------------------------------------------------------------------
# cleanup subcommand
# ---------------------------------------------------------------------------


def _seed_cleanup_db(db_path: Path) -> None:
    conn = open_db(db_path)
    ensure_schema(conn)
    rows = []
    for cid, vocab in [(1, "SNOMED"), (2, "SNOMED"), (3, "LOINC"), (4, "RxNorm")]:
        rows.append(
            EmbeddedRow(
                concept=ConceptRow(
                    concept_id=cid,
                    concept_name=f"Concept {cid}",
                    domain_id="Condition",
                    vocabulary_id=vocab,
                ),
                embedding=[0.0] * 768,
                embed_text=f"Concept {cid}",
                model_version="abc123def456",
                embedded_at=datetime.now(tz=UTC),
            )
        )
    upsert_rows(conn, rows)
    upsert_model_registry(
        conn,
        model_version="abc123def456",
        model_id="FremyCompany/BioLORD-2023",
        model_revision=None,
    )
    conn.close()


def test_cleanup_deletes_selected_vocabulary_with_yes(tmp_path: Path) -> None:
    db_path = tmp_path / "embeddings.duckdb"
    _seed_cleanup_db(db_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cleanup",
            "--db",
            str(db_path),
            "--model-version",
            "abc123",
            "--vocabulary-id",
            "SNOMED",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Deleted 2 embedding(s)." in result.output

    conn = open_db(db_path)
    ensure_schema(conn)
    remaining = conn.execute(
        "SELECT COUNT(*) FROM concept_embeddings WHERE model_version = ?",
        ["abc123def456"],
    ).fetchone()
    assert remaining[0] == 2
    conn.close()


def test_cleanup_dry_run_changes_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "embeddings.duckdb"
    _seed_cleanup_db(db_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cleanup",
            "--db",
            str(db_path),
            "--model-version",
            "abc123",
            "--all-vocabularies",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output

    conn = open_db(db_path)
    ensure_schema(conn)
    count = conn.execute("SELECT COUNT(*) FROM concept_embeddings").fetchone()
    assert count[0] == 4
    conn.close()


def test_cleanup_whole_model_requires_hash_confirmation(tmp_path: Path) -> None:
    db_path = tmp_path / "embeddings.duckdb"
    _seed_cleanup_db(db_path)
    runner = CliRunner()

    # Wrong confirmation aborts and deletes nothing.
    bad = runner.invoke(
        app,
        ["cleanup", "--db", str(db_path), "--model-version", "abc123", "--all-vocabularies"],
        input="nope\n",
    )
    assert bad.exit_code == 1
    conn = open_db(db_path)
    ensure_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM concept_embeddings").fetchone()[0] == 4
    conn.close()

    # Correct hash prefix proceeds and removes the whole model + registry entry.
    ok = runner.invoke(
        app,
        ["cleanup", "--db", str(db_path), "--model-version", "abc123", "--all-vocabularies"],
        input="abc123def456\n",
    )
    assert ok.exit_code == 0, ok.output
    assert "Deleted 4 embedding(s)." in ok.output
    conn = open_db(db_path)
    ensure_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM concept_embeddings").fetchone()[0] == 0
    assert list_model_registry(conn) == []
    conn.close()


def test_cleanup_unknown_vocabulary_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "embeddings.duckdb"
    _seed_cleanup_db(db_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cleanup",
            "--db",
            str(db_path),
            "--model-version",
            "abc123",
            "--vocabulary-id",
            "NOTREAL",
            "--yes",
        ],
    )

    assert result.exit_code == 1
    assert "not present" in result.output


def test_cleanup_interactive_selection(tmp_path: Path) -> None:
    db_path = tmp_path / "embeddings.duckdb"
    _seed_cleanup_db(db_path)
    runner = CliRunner()

    # One model version (select 1), then pick the LOINC row, then confirm y.
    # SNOMED(2) sorts first by count, LOINC and RxNorm follow.
    result = runner.invoke(
        app,
        ["cleanup", "--db", str(db_path)],
        input="1\n2\ny\n",
    )

    assert result.exit_code == 0, result.output
    conn = open_db(db_path)
    ensure_schema(conn)
    remaining = conn.execute("SELECT COUNT(*) FROM concept_embeddings").fetchone()[0]
    assert remaining == 3
    conn.close()


def test_cleanup_no_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    conn = open_db(db_path)
    ensure_schema(conn)
    conn.close()
    runner = CliRunner()

    result = runner.invoke(app, ["cleanup", "--db", str(db_path), "--yes"])

    assert result.exit_code == 0
    assert "nothing to clean up" in result.output
