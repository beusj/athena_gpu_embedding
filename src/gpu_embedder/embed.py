"""SapBERT FP32 embedding: model loading, versioning, and batched inference.

Key invariants (see AGENTS.md):
- FP32 only — never call .half() or set torch_dtype.
- Pooling is selectable (default ``cls``; mask-aware ``mean`` available),
  L2-normalized output. Non-default pooling is folded into model_version.
- model_version is a SHA-256 digest of the weights file on disk.
- Tensors are moved off GPU before collection (.cpu().numpy()).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from tqdm import tqdm

from gpu_embedder.models import EMBEDDING_DIM, ConceptRow, EmbeddedRow

if TYPE_CHECKING:
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401

logger = logging.getLogger(__name__)


def _resolve_cached_snapshot(model_id: str, revision: str | None) -> Path | None:
    """Return a local Hugging Face snapshot if one is already cached."""
    import huggingface_hub  # type: ignore[import-untyped]

    try:
        snapshot = huggingface_hub.snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        return None
    return Path(snapshot)


def _resolve_model_source(model_id: str, revision: str | None) -> tuple[Path | str, bool]:
    """Resolve a model path from cache first, then fall back to Hub download."""
    local_path = Path(model_id)
    if local_path.exists():
        return local_path, True

    cached = _resolve_cached_snapshot(model_id, revision)
    if cached is not None:
        return cached, True

    return model_id, False


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

def _apply_run_variant(
    weights_digest: str,
    precision: str,
    quantization_scheme: str,
    pooling: str = "cls",
) -> str:
    """Fold non-default run variants into the model_version digest.

    A "run variant" is something that changes the embeddings without changing
    the weights file — precision, quantization, or pooling strategy. The full
    default (``fp32`` + ``none`` + ``cls``) returns the bare weights digest
    unchanged, so stores hashed before this existed keep their ``model_version``
    (no mass re-embed). Any non-default variant yields a distinct digest, so it
    gets a separate ``(namespace, concept_id, model_version)`` identity instead
    of colliding with — and overwriting — the embeddings of the same weights
    pooled/quantized differently. Provenance stays human-readable in
    ``model_registry``.

    ``pooling="cls"`` adds no suffix, so the precision/quantization digests
    predating pooling support are preserved byte-for-byte.
    """
    if precision == "fp32" and quantization_scheme == "none" and pooling == "cls":
        return weights_digest
    suffix = f"|precision={precision}|quantization={quantization_scheme}"
    if pooling != "cls":
        suffix += f"|pooling={pooling}"
    return hashlib.sha256((weights_digest + suffix).encode()).hexdigest()


def compute_model_version(
    model_id_or_path: str | Path,
    revision: str | None = None,
    *,
    precision: str = "fp32",
    quantization_scheme: str = "none",
    pooling: str = "cls",
) -> str:
    """Return the model_version digest for a checkpoint.

    Base digest is the SHA-256 of the model weights file (model.safetensors,
    then pytorch_model.bin; falls back to hashing the model ID string when no
    weights file is found, as in tests with non-existent paths). When
    *precision*/*quantization_scheme*/*pooling* are non-default they are folded
    into the digest so variants of the same weights get distinct versions; the
    default fp32/none/cls returns the bare weights digest (stable across
    upgrades).
    """
    base = Path(model_id_or_path)
    if base.is_dir():
        candidates = [
            base / "model.safetensors",
            base / "pytorch_model.bin",
        ]
    else:
        cached = _resolve_cached_snapshot(str(model_id_or_path), revision)
        if cached is not None:
            candidates = [
                cached / "model.safetensors",
                cached / "pytorch_model.bin",
            ]
        else:
            try:
                import huggingface_hub  # type: ignore[import-untyped]

                cache_dir = Path(
                    huggingface_hub.snapshot_download(
                        repo_id=str(model_id_or_path),
                        revision=revision,
                        local_files_only=False,
                    )
                )
                candidates = [
                    cache_dir / "model.safetensors",
                    cache_dir / "pytorch_model.bin",
                ]
            except Exception:
                logger.debug("Could not resolve HF cache path for %s", model_id_or_path)
                candidates = []

    weights_digest: str | None = None
    for candidate in candidates:
        if candidate.exists():
            logger.info("Hashing weights file: %s", candidate)
            h = hashlib.sha256()
            with candidate.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            weights_digest = h.hexdigest()
            break

    if weights_digest is None:
        # Fallback: hash the model ID string (deterministic for the same string)
        logger.warning(
            "No weights file found for %s; using hash of model ID string", model_id_or_path
        )
        weights_digest = hashlib.sha256(str(model_id_or_path).encode()).hexdigest()

    return _apply_run_variant(weights_digest, precision, quantization_scheme, pooling)


# ---------------------------------------------------------------------------
# Shared cross-repo retrieval stamp (ALIGNMENT.md §4.2)
# ---------------------------------------------------------------------------

_RETRIEVAL_DIMENSION = EMBEDDING_DIM  # canonical dim (gpu_embedder.models, ALIGNMENT.md §7)


def retrieval_model_version(
    model_name: str,
    revision: str | None = None,
    *,
    pooling: str = "cls",
    precision: str = "fp32",
    normalize: bool = True,
    dimension: int = _RETRIEVAL_DIMENSION,
) -> str:
    """Config-derived ``embed_model_version`` for the warehouse handoff (ALIGNMENT.md §4.2).

    This is the *retrieval-facing* stamp concept-mapper's Stage 3 filters on
    (``concept_embeddings.embed_model_version`` and
    ``source_concepts.embed_model_version``). It is a pure function of the pinned
    vector-space attributes and deliberately **excludes the runtime engine** (CUDA
    here vs ONNX in concept-mapper) and throughput knobs, so one pinned artifact
    yields one stamp on both sides. The identity dict below MUST stay byte-identical
    to ``concept_mapper.embeddings.embedder.pinned_attributes`` or the two repos
    will compute different versions and semantic retrieval will silently miss.

    Distinct from :func:`compute_model_version`, which hashes the weights file and
    remains the local Lance/DuckDB store identity (provenance in ``model_registry``).
    Both target concepts (``concept_name``) and source concepts (``source_name``)
    are stamped with this same value — they share one vector space; only the
    embedded text differs.
    """
    identity = {
        "model_name": model_name,
        "revision": revision,
        "pooling": pooling,
        "precision": precision,
        "normalize": normalize,
        "dimension": dimension,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    return f"sapbert-{pooling}-{precision}-{digest}"


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
    source, cached = _resolve_model_source(model_id, revision)
    source_label = str(source)
    cache_state = "cached" if cached else "download"

    logger.info(
        "Loading tokenizer from %s (revision=%s, source=%s)",
        model_id,
        rev_label,
        cache_state,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=cached,
    )

    logger.info(
        "Loading model from %s → device=%s (FP32, revision=%s, source=%s)",
        model_id,
        device,
        rev_label,
        source_label,
    )
    model = AutoModel.from_pretrained(source, revision=revision, local_files_only=cached)
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
    pooling: str = "cls",
) -> np.ndarray:
    """Embed a list of strings and return float32 (N, 768) array.

    Pooling is ``cls`` (CLS token, SapBERT default) or ``mean`` (mask-aware
    average over tokens, for sentence-transformers models like BioLORD-2023),
    followed by L2 normalisation. Tensors are moved off GPU before returning so
    callers never accumulate GPU memory across batches.
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

    if pooling == "mean":
        # Mask-aware mean pooling: average only over real (non-padding) tokens
        # so the result is independent of right-padding length.
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (N, L, 1)
        summed = (out.last_hidden_state * mask).sum(dim=1)  # (N, 768)
        vecs: torch.Tensor = summed / mask.sum(dim=1).clamp(min=1e-9)
    else:
        # CLS pooling
        vecs = out.last_hidden_state[:, 0, :]  # (N, 768)

    # L2 normalise
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vecs = vecs / norms

    return vecs.cpu().float().numpy()


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
    *,
    pooling: str = "cls",
    precomputed_texts: dict[int, str] | None = None,
) -> list[EmbeddedRow]:
    """Embed all rows in batches, returning EmbeddedRow objects.

    Progress is shown via tqdm.  On any exception within a batch the error is
    logged and re-raised — no partial writes.

    *precomputed_texts* maps ``concept_id`` → embed text; when supplied the
    caller has already built the text (e.g. for change detection) and we reuse
    it instead of recomputing ``build_embed_text`` per row.
    """
    total_batches = max((len(rows) + batch_size - 1) // batch_size, 1)
    logger.info(
        "Embedding %d rows in %d batches of up to %d on %s",
        len(rows),
        total_batches,
        batch_size,
        device,
    )

    result: list[EmbeddedRow] = []
    now = datetime.now(tz=UTC)

    progress = tqdm(
        range(0, len(rows), batch_size),
        desc=f"Embedding ({device})",
        unit="batch",
        total=total_batches,
        leave=True,
    )
    for batch_index, start in enumerate(progress, start=1):
        batch = rows[start : start + batch_size]
        if precomputed_texts is not None:
            texts = [precomputed_texts[r.concept_id] for r in batch]
        else:
            texts = [build_embed_text(r, text_fields, separator) for r in batch]
        try:
            logger.info(
                "Embedding batch %d/%d (%d rows)",
                batch_index,
                total_batches,
                len(batch),
            )
            progress.set_postfix(rows=f"{start + 1}-{start + len(batch)}")
            vecs: np.ndarray = embed_batch(
                texts, model, tokenizer, device, max_length, pooling
            )
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
