"""Unit tests for config.py — EmbedConfig parsing and validation."""

from __future__ import annotations


class TestEmbedConfigDefaults:
    def test_defaults(self, monkeypatch) -> None:
        from pathlib import Path

        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.vocab_dir == Path("athena_vocab")
        assert cfg.db == Path("embeddings.duckdb")
        assert cfg.log_dir == Path("logs")
        assert cfg.log_max_bytes == 2 * 1024 * 1024
        assert cfg.log_max_files == 5
        assert cfg.model == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        assert cfg.model_revision is None
        assert cfg.batch_size == 256
        assert cfg.max_length == 128
        assert cfg.pooling == "cls"
        assert cfg.upsert_every_batches == 250
        assert cfg.ingest_engine == "duckdb"
        assert cfg.write_mode == "ndjson"
        assert cfg.text_fields == ["concept_name"]
        assert cfg.separator == " "
        assert cfg.force is False

    def test_device_auto_resolves_to_string(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.device in ("cuda", "mps", "cpu")

    def test_device_explicit_preserved(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(device="cpu", _env_file=None)  # type: ignore[call-arg]
        assert cfg.device == "cpu"


class TestEmbedConfigTextFields:
    def test_list_input(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(text_fields=["concept_code", "concept_name"], _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]

    def test_comma_separated_string_parsed(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(text_fields="concept_code,concept_name", _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]

    def test_comma_separated_with_spaces(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(text_fields=" concept_code , concept_name ", _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]


class TestEmbedConfigRevision:
    def test_revision_none_by_default(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.model_revision is None

    def test_revision_set(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        monkeypatch.delenv("GPU_EMBED_INGEST_ENGINE", raising=False)
        cfg = EmbedConfig(model_revision="abc123def456", _env_file=None)  # type: ignore[call-arg]
        assert cfg.model_revision == "abc123def456"


class TestEmbedConfigWriteMode:
    def test_write_mode_env_override(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.setenv("GPU_EMBED_WRITE_MODE", "direct")
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.write_mode == "direct"


class TestEmbedConfigCheckpointing:
    def test_upsert_every_batches_env_override(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.setenv("GPU_EMBED_UPSERT_EVERY_BATCHES", "7")
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.upsert_every_batches == 7

    def test_upsert_every_batches_must_be_positive(self, monkeypatch) -> None:
        import pytest

        from gpu_embedder.config import EmbedConfig

        monkeypatch.setenv("GPU_EMBED_UPSERT_EVERY_BATCHES", "0")
        with pytest.raises(ValueError, match="must be greater than 0"):
            EmbedConfig(_env_file=None)  # type: ignore[call-arg]
