"""Unit tests for embed.py — uses a FakeModel/FakeTokenizer; no GPU or network needed."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpu_embedder.embed import (
    build_embed_text,
    compute_model_version,
    embed_all,
    embed_batch,
)
from gpu_embedder.models import ConceptRow

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_row(**kwargs: Any) -> ConceptRow:
    defaults = {
        "concept_id": 1,
        "concept_name": "test concept",
        "domain_id": "Condition",
        "vocabulary_id": "SNOMED",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
        "concept_code": "123456",
        "invalid_reason": None,
    }
    defaults.update(kwargs)
    return ConceptRow(**defaults)


def _fake_model_output(batch_size: int, hidden: int = 768) -> MagicMock:
    """Return a mock model output with random last_hidden_state."""
    import torch

    out = MagicMock()
    out.last_hidden_state = torch.randn(batch_size, 10, hidden)
    return out


def _fake_model(batch_size: int | None = None) -> MagicMock:
    model = MagicMock()
    if batch_size is not None:
        model.return_value = _fake_model_output(batch_size)
    else:
        # dynamically match batch size from input_ids
        def _call(**kwargs: Any) -> MagicMock:
            n = kwargs["input_ids"].shape[0]
            return _fake_model_output(n)
        model.side_effect = _call
    return model


def _fake_tokenizer(batch_size: int | None = None) -> MagicMock:
    import torch

    tok = MagicMock()
    def _call(texts: list[str], **kwargs: Any) -> dict[str, Any]:
        n = len(texts)
        return {"input_ids": torch.zeros(n, 10, dtype=torch.long)}
    tok.side_effect = _call
    return tok


# ---------------------------------------------------------------------------
# compute_model_version
# ---------------------------------------------------------------------------

class TestComputeModelVersion:
    def test_returns_64_char_hex_string(self, tmp_path: Path) -> None:
        # Create a fake safetensors file
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"fake weights data")
        version = compute_model_version(tmp_path)
        assert len(version) == 64
        assert all(c in "0123456789abcdef" for c in version)

    def test_same_content_same_digest(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stable weights")
        v1 = compute_model_version(tmp_path)
        v2 = compute_model_version(tmp_path)
        assert v1 == v2

    def test_different_content_different_digest(self, tmp_path: Path) -> None:
        d1 = tmp_path / "m1"
        d2 = tmp_path / "m2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "model.safetensors").write_bytes(b"content A")
        (d2 / "model.safetensors").write_bytes(b"content B")
        assert compute_model_version(d1) != compute_model_version(d2)

    def test_fallback_to_string_hash_when_no_file(self) -> None:
        # Non-existent path and cannot resolve HF cache → hash of string
        version = compute_model_version("some-nonexistent-model-id-xyz")
        assert len(version) == 64

    def test_pytorch_bin_fallback(self, tmp_path: Path) -> None:
        weights = tmp_path / "pytorch_model.bin"
        weights.write_bytes(b"pytorch weights")
        version = compute_model_version(tmp_path)
        expected = hashlib.sha256(b"pytorch weights").hexdigest()
        assert version == expected

    def test_fp32_none_keeps_bare_weights_digest(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"weights")
        bare = hashlib.sha256(b"weights").hexdigest()
        # Default and explicit fp32/none must equal the unsuffixed weights hash
        # so existing stores keep their model_version after this change.
        assert compute_model_version(tmp_path) == bare
        assert (
            compute_model_version(tmp_path, precision="fp32", quantization_scheme="none")
            == bare
        )

    def test_quantization_yields_distinct_version(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"weights")
        bare = compute_model_version(tmp_path)
        int8 = compute_model_version(tmp_path, quantization_scheme="int8")
        fp16 = compute_model_version(tmp_path, precision="fp16")
        assert len({bare, int8, fp16}) == 3
        assert all(len(v) == 64 for v in (int8, fp16))


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_model — revision forwarding
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_revision_forwarded_to_from_pretrained(self) -> None:
        """load_model must pass the revision kwarg to both from_pretrained calls."""
        fake_model = MagicMock()
        fake_model.float.return_value = fake_model
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        with (
            patch("transformers.AutoModel.from_pretrained", return_value=fake_model) as m_model,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()) as m_tok,
        ):
            from gpu_embedder.embed import load_model

            load_model("some/model", "cpu", revision="abc123def456")
            _, model_kwargs = m_model.call_args
            _, tok_kwargs = m_tok.call_args
            assert model_kwargs.get("revision") == "abc123def456"
            assert tok_kwargs.get("revision") == "abc123def456"

    def test_none_revision_still_calls_from_pretrained(self) -> None:
        fake_model = MagicMock()
        fake_model.float.return_value = fake_model
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        with (
            patch("transformers.AutoModel.from_pretrained", return_value=fake_model) as m_model,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        ):
            from gpu_embedder.embed import load_model

            load_model("some/model", "cpu", revision=None)
            _, model_kwargs = m_model.call_args
            assert model_kwargs.get("revision") is None


class TestEmbedBatch:
    def test_output_shape(self) -> None:

        texts = ["concept one", "concept two", "concept three"]
        model = _fake_model()
        tokenizer = _fake_tokenizer()
        result = embed_batch(texts, model, tokenizer, "cpu", max_length=128)
        assert result.shape == (3, 768)

    def test_output_dtype_float32(self) -> None:
        texts = ["hello"]
        result = embed_batch(texts, _fake_model(), _fake_tokenizer(), "cpu")
        assert result.dtype == np.float32

    def test_output_is_l2_normalised(self) -> None:

        texts = ["a", "b", "c"]
        result = embed_batch(texts, _fake_model(), _fake_tokenizer(), "cpu")
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_returns_numpy_not_tensor(self) -> None:
        result = embed_batch(["x"], _fake_model(), _fake_tokenizer(), "cpu")
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# build_embed_text
# ---------------------------------------------------------------------------

class TestBuildEmbedText:
    def test_single_field(self) -> None:
        row = _make_row(concept_name="Type 2 DM")
        assert build_embed_text(row, ["concept_name"], " ") == "Type 2 DM"

    def test_multiple_fields_with_separator(self) -> None:
        row = _make_row(concept_code="44054006", concept_name="Type 2 DM")
        result = build_embed_text(row, ["concept_code", "concept_name"], ": ")
        assert result == "44054006: Type 2 DM"

    def test_none_field_skipped(self) -> None:
        row = _make_row(concept_name="Test", standard_concept=None)
        result = build_embed_text(row, ["standard_concept", "concept_name"], " ")
        assert result == "Test"


# ---------------------------------------------------------------------------
# embed_all
# ---------------------------------------------------------------------------

class TestEmbedAll:
    def _run(self, rows: list[ConceptRow], **kwargs: Any) -> list[Any]:
        defaults = dict(
            model=_fake_model(),
            tokenizer=_fake_tokenizer(),
            device="cpu",
            batch_size=4,
            max_length=128,
            text_fields=["concept_name"],
            separator=" ",
            model_version="abc123",
        )
        defaults.update(kwargs)
        return embed_all(rows, **defaults)  # type: ignore[arg-type]

    def test_returns_embedded_rows(self) -> None:
        from gpu_embedder.models import EmbeddedRow

        rows = [_make_row(concept_id=i, concept_name=f"concept {i}") for i in range(5)]
        result = self._run(rows)
        assert len(result) == 5
        assert all(isinstance(r, EmbeddedRow) for r in result)

    def test_embedding_length_is_768(self) -> None:
        rows = [_make_row(concept_id=1, concept_name="test")]
        result = self._run(rows)
        assert len(result[0].embedding) == 768

    def test_embed_text_uses_text_fields(self) -> None:
        row = _make_row(concept_id=1, concept_name="Diabetes", concept_code="44054006")
        result = self._run([row], text_fields=["concept_code", "concept_name"], separator=": ")
        assert result[0].embed_text == "44054006: Diabetes"

    def test_model_version_stamped(self) -> None:
        rows = [_make_row()]
        result = self._run(rows, model_version="test_version_xyz")
        assert result[0].model_version == "test_version_xyz"

    def test_batching(self) -> None:
        rows = [_make_row(concept_id=i, concept_name=f"c{i}") for i in range(10)]
        result = self._run(rows, batch_size=3)
        assert len(result) == 10

    def test_reraises_on_batch_error(self) -> None:
        rows = [_make_row()]
        bad_model = MagicMock(side_effect=RuntimeError("GPU OOM"))
        with pytest.raises(RuntimeError, match="GPU OOM"):
            self._run(rows, model=bad_model)
