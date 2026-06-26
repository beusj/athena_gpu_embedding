"""Unit tests for config.py — EmbedConfig parsing and validation."""

from __future__ import annotations


class TestEmbedConfigDefaults:
    def test_defaults(self, monkeypatch) -> None:
        from pathlib import Path

        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.vocab_dir == Path("athena_vocab")
        assert cfg.db == Path("embeddings.duckdb")
        assert cfg.model == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        assert cfg.model_revision is None
        assert cfg.batch_size == 256
        assert cfg.max_length == 128
        assert cfg.text_fields == ["concept_name"]
        assert cfg.separator == " "
        assert cfg.force is False

    def test_device_auto_resolves_to_string(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.device in ("cuda", "mps", "cpu")

    def test_device_explicit_preserved(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(device="cpu", _env_file=None)  # type: ignore[call-arg]
        assert cfg.device == "cpu"


class TestEmbedConfigTextFields:
    def test_list_input(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(text_fields=["concept_code", "concept_name"], _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]

    def test_comma_separated_string_parsed(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(text_fields="concept_code,concept_name", _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]

    def test_comma_separated_with_spaces(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(text_fields=" concept_code , concept_name ", _env_file=None)  # type: ignore[call-arg]
        assert cfg.text_fields == ["concept_code", "concept_name"]


class TestEmbedConfigRevision:
    def test_revision_none_by_default(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.model_revision is None

    def test_revision_set(self, monkeypatch) -> None:
        from gpu_embedder.config import EmbedConfig

        monkeypatch.delenv("GPU_EMBED_MODEL_REVISION", raising=False)
        monkeypatch.delenv("GPU_EMBED_TEXT_FIELDS", raising=False)
        cfg = EmbedConfig(model_revision="abc123def456", _env_file=None)  # type: ignore[call-arg]
        assert cfg.model_revision == "abc123def456"
