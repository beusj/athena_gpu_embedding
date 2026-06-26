"""Unit tests for cli.py command behavior."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
from typer.testing import CliRunner

from gpu_embedder.cli import app
from gpu_embedder.models import SCHEMA_DDL


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
