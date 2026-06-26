"""Unit tests for cli.py command behavior."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gpu_embedder.cli import app


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

    def fake_run(cmd: list[str], check: bool) -> None:
        captured["cmd"] = cmd
        captured["check"] = check
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
    assert Path(cmd[4]).name == "athena_vocab"
