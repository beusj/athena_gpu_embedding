"""Unit tests for cli.py command behavior."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from typer.testing import CliRunner

from gpu_embedder.cli import app
from gpu_embedder.models import SCHEMA_DDL, ConceptRow, EmbeddedRow


def _seed_embeddings_db(db_path: Path) -> None:
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_DDL)
    conn.execute(
        """
        INSERT INTO concept_embeddings (
            concept_id, concept_name, domain_id, vocabulary_id,
            concept_class_id, standard_concept, concept_code,
            invalid_reason, embedding, embed_text, model_version, embedded_at
        ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, NOW())
        """,
        [999002, "CPT4 test concept", "Procedure", "CPT4", [0.0] * 768, "CPT4 test concept", "v1"],
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

    def fake_read_csv(path: Path, spec, engine: str):  # type: ignore[no-untyped-def]
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

    def fake_get_existing_ids(conn, model_version: str) -> set[int]:  # type: ignore[no-untyped-def]
        return set()

    def fake_upsert_rows(conn, embedded_rows, mode: str = "ndjson") -> None:  # type: ignore[no-untyped-def]
        upsert_sizes.append(len(embedded_rows))

    def fake_count_rows(conn, model_version: str) -> int:  # type: ignore[no-untyped-def]
        return sum(upsert_sizes)

    monkeypatch.setattr("gpu_embedder.cli.read_csv", fake_read_csv)
    monkeypatch.setattr("gpu_embedder.embed.load_model", fake_load_model)
    monkeypatch.setattr("gpu_embedder.embed.compute_model_version", fake_compute_model_version)
    monkeypatch.setattr("gpu_embedder.embed.embed_all", fake_embed_all)
    monkeypatch.setattr("gpu_embedder.store.open_db", fake_open_db)
    monkeypatch.setattr("gpu_embedder.store.ensure_schema", fake_ensure_schema)
    monkeypatch.setattr("gpu_embedder.store.get_existing_ids", fake_get_existing_ids)
    monkeypatch.setattr("gpu_embedder.store.upsert_rows", fake_upsert_rows)
    monkeypatch.setattr("gpu_embedder.store.count_rows", fake_count_rows)

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


def test_embed_accepts_comma_delimited_vocabulary_id(monkeypatch) -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent.parent / "fixtures" / "CONCEPT_mini.tsv"

    captured: dict[str, object] = {}

    def fake_read_csv(path: Path, spec, engine: str):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["vocabulary_ids"] = spec.vocabulary_ids
        captured["engine"] = engine
        return []

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

    def fake_read_csv(path: Path, spec, engine: str):  # type: ignore[no-untyped-def]
        captured["vocabulary_ids"] = spec.vocabulary_ids
        return []

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


def test_export_writes_sharded_parquet_by_vocabulary(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "embeddings.duckdb"
    out_dir = tmp_path / "parquet"

    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_DDL)

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

    conn.executemany(
        """
        INSERT INTO concept_embeddings (
            concept_id, concept_name, domain_id, vocabulary_id,
            concept_class_id, standard_concept, concept_code,
            invalid_reason, embedding, embed_text, model_version, embedded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
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
            )
            for row in rows
        ],
    )
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

    verify_conn = duckdb.connect()
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
