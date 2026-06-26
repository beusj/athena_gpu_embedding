"""SapBERT FP32 embedding: model loading, versioning, and batched inference.

Key invariants (see AGENTS.md):
- FP32 only — never call .half() or set torch_dtype.
- CLS-token pooling, L2-normalized output.
- model_version is a SHA-256 digest of the weights file on disk.
- Tensors are moved off GPU before collection (.cpu().numpy()).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from tqdm import tqdm

from gpu_embedder.models import ConceptRow, EmbeddedRow

if TYPE_CHECKING:
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for unit-test injection
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Minimal interface satisfied by (model, tokenizer) pairs and fakes."""

    def __call__(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        device: str,
    ) -> np.ndarray:
        """Return float32 array of shape (N, 768), L2-normalised."""
        ...


# ---------------------------------------------------------------------------
# Model version
# ---------------------------------------------------------------------------

def compute_model_version(model_id_or_path: str | Path) -> str:
    """Return a SHA-256 hex digest of the model weights file.

    Tries model.safetensors first, then pytorch_model.bin.  If neither is
    found at a local path, falls back to hashing the model ID string (used in
    tests with non-existent paths).
    """
    candidates: list[Path] = []
    base = Path(model_id_or_path)
    if base.is_dir():
        candidates = [
            base / "model.safetensors",
            base / "pytorch_model.bin",
        ]
    else:
        # Try HuggingFace cache layout
        import huggingface_hub  # type: ignore[import-untyped]

        try:
            cache_dir = Path(
                huggingface_hub.snapshot_download(
                    str(model_id_or_path),
                    local_files_only=True,
                )
            )
            candidates = [
                cache_dir / "model.safetensors",
                cache_dir / "pytorch_model.bin",
            ]
        except Exception:
            logger.debug("Could not resolve HF cache path for %s", model_id_or_path)

    for candidate in candidates:
        if candidate.exists():
            logger.info("Hashing weights file: %s", candidate)
            h = hashlib.sha256()
            with candidate.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

    # Fallback: hash the model ID string (deterministic for the same string)
    logger.warning(
        "No weights file found for %s; using hash of model ID string", model_id_or_path
    )
    return hashlib.sha256(str(model_id_or_path).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    device: str,
    revision: str | None = None,
) -> tuple[object, object]:
    """Load SapBERT tokenizer and model in FP32 eval mode.

    Returns (model, tokenizer).  The model is moved to *device* and set to
    eval mode.  fp16/bf16 is never used.

    *revision* pins a specific HuggingFace commit hash, branch, or tag so that
    downloads are reproducible.  None uses the upstream default branch.
    """
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    rev_label = revision or "default"
    logger.info("Loading tokenizer from %s (revision=%s)", model_id, rev_label)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    logger.info(
        "Loading model from %s → device=%s (FP32, revision=%s)", model_id, device, rev_label
    )
    model = AutoModel.from_pretrained(model_id, revision=revision)
    model = model.float()  # enforce FP32 — never call .half()
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def embed_batch(
    texts: list[str],
    model: object,
    tokenizer: object,
    device: str,
    max_length: int = 128,
) -> np.ndarray:
    """Embed a list of strings and return float32 (N, 768) array.

    CLS-token pooling + L2 normalisation.  Tensors are moved off GPU before
    returning so callers never accumulate GPU memory across batches.
    """
    enc = tokenizer(  # type: ignore[operator]
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)  # type: ignore[operator]

    # CLS pooling
    cls_vecs: torch.Tensor = out.last_hidden_state[:, 0, :]  # (N, 768)

    # L2 normalise
    norms = cls_vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cls_vecs = cls_vecs / norms

    return cls_vecs.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_embed_text(row: ConceptRow, text_fields: list[str], separator: str) -> str:
    """Concatenate the requested fields from a ConceptRow into the input string."""
    parts: list[str] = []
    for field_name in text_fields:
        val = getattr(row, field_name, None)
        if val is not None:
            parts.append(str(val))
    return separator.join(parts)


def embed_all(
    rows: list[ConceptRow],
    model: object,
    tokenizer: object,
    device: str,
    batch_size: int,
    max_length: int,
    text_fields: list[str],
    separator: str,
    model_version: str,
) -> list[EmbeddedRow]:
    """Embed all rows in batches, returning EmbeddedRow objects.

    Progress is shown via tqdm.  On any exception within a batch the error is
    logged and re-raised — no partial writes.
    """
    result: list[EmbeddedRow] = []
    now = datetime.now(tz=UTC)

    for start in tqdm(range(0, len(rows), batch_size), desc="Embedding", unit="batch"):
        batch = rows[start : start + batch_size]
        texts = [build_embed_text(r, text_fields, separator) for r in batch]
        try:
            vecs: np.ndarray = embed_batch(texts, model, tokenizer, device, max_length)
        except Exception:
            logger.error(
                "embed_batch failed on rows %d-%d", start, start + len(batch) - 1
            )
            raise

        for row, text, vec in zip(batch, texts, vecs, strict=True):
            result.append(
                EmbeddedRow(
                    concept=row,
                    embedding=vec.tolist(),
                    embed_text=text,
                    model_version=model_version,
                    embedded_at=now,
                )
            )

    return result
