"""Typer CLI entry point for gpu-embedder.

Subcommands:
  gpu-embed embed   [OPTIONS] [CSV_PATH...] — batch-embed concepts
  gpu-embed cpt4    [OPTIONS]               — populate CPT-4 names via Athena Java tool
  gpu-embed cleanup [OPTIONS]               — delete embeddings for a model/vocabularies

This module is intentionally thin: all logic lives in config, ingest, embed,
and store.  cli.py is excluded from coverage requirements.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import torch
import typer
from dotenv import load_dotenv
from typer.main import get_command

if TYPE_CHECKING:
    from gpu_embedder.report import ModelVersionInfo
    from gpu_embedder.store import ModelRegistryEntry

from gpu_embedder import __version__
from gpu_embedder.config import EmbedConfig
from gpu_embedder.ingest import (
    compute_csv_fingerprint,
    filter_spec_hash,
    read_csv,
    read_source_parquet,
)
from gpu_embedder.models import (
    ALL_VOCABULARIES_SENTINEL,
    DEFAULT_VOCABULARY_IDS,
    FilterSpec,
)

app = typer.Typer(
    name="gpu-embed",
    help="Batch-embed OHDSI Athena concepts with SapBERT into DuckDB.",
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    invoke_without_command=True,
)

logger = logging.getLogger(__name__)


_LOADING_MODEL_RE = re.compile(
    r"Loading model from (?P<model_id>.+?) → .*?\(FP32, revision=(?P<revision>[^,\)]+)",
)


def _split_multi_values(values: list[str] | None) -> list[str]:
    """Normalize repeatable option values with optional comma-delimited input.

    Example:
    - ["LOINC", "SNOMED,RxNorm"] -> ["LOINC", "SNOMED", "RxNorm"]
    """
    if not values:
        return []

    normalized: list[str] = []
    for value in values:
        normalized.extend(piece.strip() for piece in value.split(",") if piece.strip())
    return normalized


def _resolve_source_parquet_paths(path: Path) -> list[Path]:
    """Return parquet files under *path* (file or directory), sorted for stability."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.parquet") if p.is_file())
    raise ValueError(f"Unsupported source parquet path: {path}")


def _extract_model_pairs_from_logs(log_dir: Path) -> list[tuple[str, str | None]]:
    """Parse log files for model_id/revision pairs used by embed runs."""
    if not log_dir.exists() or not log_dir.is_dir():
        return []

    seen: set[tuple[str, str | None]] = set()
    ordered: list[tuple[str, str | None]] = []

    for log_path in sorted(log_dir.glob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _LOADING_MODEL_RE.search(line)
            if not match:
                continue
            model_id = match.group("model_id").strip()
            revision_text = match.group("revision").strip()
            revision = None if revision_text == "default" else revision_text
            key = (model_id, revision)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)

    return ordered


def _backfill_model_registry_from_logs(
    conn: Any,
    *,
    log_dir: Path,
) -> tuple[int, int] | None:
    """Backfill model-registry entries from embed logs.

    Returns ``(added, failed)`` when log candidates are found, otherwise
    ``None`` (and emits a "no candidates" message).
    """
    from gpu_embedder import store as st
    from gpu_embedder.embed import compute_model_version

    candidates = _extract_model_pairs_from_logs(log_dir)
    if not candidates:
        typer.echo(f"No log-derived model pairs found in {log_dir}.")
        return None

    added = 0
    failed = 0
    for model_id, revision in candidates:
        try:
            model_version = compute_model_version(model_id, revision=revision)
        except Exception as exc:  # pragma: no cover - defensive CLI handling
            failed += 1
            revision_label = revision or "default"
            typer.echo(
                (
                    f"Warning: could not hash model '{model_id}' "
                    f"(revision={revision_label}): {exc}"
                ),
                err=True,
            )
            continue

        st.upsert_model_registry(
            conn,
            model_version=model_version,
            model_id=model_id,
            model_revision=revision,
            precision="fp32",
            quantization_scheme="none",
        )
        added += 1

    return added, failed


def _resolve_java_executable() -> Path | None:
    """Return a usable Java executable from PATH, JAVA_HOME, or common installs."""
    on_path = shutil.which("java")
    if on_path:
        return Path(on_path)

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        suffix = "java.exe" if os.name == "nt" else "java"
        candidate = Path(java_home) / "bin" / suffix
        if candidate.exists():
            return candidate

    if os.name == "nt":
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        patterns = [
            "Eclipse Adoptium/*/bin/java.exe",
            "Java/*/bin/java.exe",
            "Microsoft/*/jdk/*/bin/java.exe",
            "JetBrains/*/jbr/bin/java.exe",
        ]
        for root in program_files:
            for pattern in patterns:
                matches = sorted(glob(str(Path(root) / pattern)), reverse=True)
                if matches:
                    return Path(matches[0])

    return None


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gpu-embedder {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging")] = False,
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)

    if ctx.invoked_subcommand is None:
        group = get_command(app)
        embed_command = group.commands["embed"]
        embed_command.main(
            args=list(ctx.args),
            prog_name=f"{ctx.info_name} embed",
            standalone_mode=False,
        )
        raise typer.Exit()


# ---------------------------------------------------------------------------
# embed subcommand
# ---------------------------------------------------------------------------

@app.command("embed")
def embed_cmd(
    csv_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Explicit CONCEPT.csv paths (defaults to <vocab-dir>/CONCEPT.csv)"),
    ] = None,
    vocab_dir: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_VOCAB_DIR", help="Directory containing CONCEPT.csv"),
    ] = None,
    source_parquet: Annotated[
        Path | None,
        typer.Option(
            "--source-parquet",
            envvar="GPU_EMBED_SOURCE_PARQUET",
            help="Source-concept parquet file or directory to embed",
        ),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option(
            envvar="GPU_EMBED_DB",
            help="Embedding store path: .lance (default), .duckdb, or a parquet directory",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(envvar="GPU_EMBED_MODEL", help="HuggingFace model ID or local path"),
    ] = None,
    model_revision: Annotated[
        str | None,
        typer.Option(
            "--model-revision",
            envvar="GPU_EMBED_MODEL_REVISION",
            help="HuggingFace commit hash / branch / tag to pin the model revision",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(envvar="GPU_EMBED_DEVICE", help="cuda | cpu | mps | auto"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(envvar="GPU_EMBED_BATCH_SIZE", help="Rows per GPU forward pass"),
    ] = None,
    max_length: Annotated[
        int | None,
        typer.Option(envvar="GPU_EMBED_MAX_LENGTH", help="Tokenizer max sequence length"),
    ] = None,
    pooling: Annotated[
        str | None,
        typer.Option(
            "--pooling",
            envvar="GPU_EMBED_POOLING",
            help="Token pooling: cls (SapBERT default) or mean (e.g. BioLORD-2023)",
        ),
    ] = None,
    upsert_every_batches: Annotated[
        int | None,
        typer.Option(
            "--upsert-every-batches",
            envvar="GPU_EMBED_UPSERT_EVERY_BATCHES",
            help="Checkpoint writes every N embedding batches",
        ),
    ] = None,
    ingest_engine: Annotated[
        str | None,
        typer.Option(
            "--ingest-engine",
            envvar="GPU_EMBED_INGEST_ENGINE",
            help="CSV ingest engine: duckdb (default) or python",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-embed already-stored concepts")
    ] = False,
    vocabulary_id: Annotated[
        list[str] | None,
        typer.Option(
            "--vocabulary-id",
            help="Filter: vocabulary IDs (repeatable or comma-delimited)",
        ),
    ] = None,
    domain_id: Annotated[
        list[str] | None,
        typer.Option("--domain-id", help="Filter: domain IDs to include (repeatable)"),
    ] = None,
    concept_class_id: Annotated[
        list[str] | None,
        typer.Option(
            "--concept-class-id",
            help="Filter: concept class IDs to include (repeatable)",
        ),
    ] = None,
    standard_concept: Annotated[
        list[str] | None,
        typer.Option(
            "--standard-concept",
            help="Filter: standard_concept values to include; S, C, or blank (repeatable)",
        ),
    ] = None,
    invalid_reason: Annotated[
        list[str] | None,
        typer.Option(
            "--invalid-reason",
            help='Filter: invalid_reason values; use "valid" for NULL/empty (repeatable)',
        ),
    ] = None,
    text_field: Annotated[
        list[str] | None,
        typer.Option(
            "--text-field",
            help="Concept columns to concatenate as embed input (repeatable)",
        ),
    ] = None,
    source_text_field: Annotated[
        list[str] | None,
        typer.Option(
            "--source-text-field",
            help="Source parquet columns to concatenate as embed input (repeatable)",
        ),
    ] = None,
    separator: Annotated[
        str | None,
        typer.Option(
            envvar="GPU_EMBED_SEPARATOR",
            help="Separator between concatenated text fields",
        ),
    ] = None,
    namespace: Annotated[
        str | None,
        typer.Option(
            "--namespace",
            envvar="GPU_EMBED_NAMESPACE",
            help=(
                "Identity namespace for these concepts (default 'athena'). Use a "
                "distinct value for source-concept datasets so their concept_ids "
                "do not collide with Athena standard concepts."
            ),
        ),
    ] = None,
    source_namespace: Annotated[
        str | None,
        typer.Option(
            "--source-namespace",
            envvar="GPU_EMBED_SOURCE_NAMESPACE",
            help="Default namespace to use for --source-parquet runs",
        ),
    ] = None,
) -> None:
    """Batch-embed Athena CSVs or source-concept parquet rows with SapBERT."""
    # Build config, allowing CLI overrides
    cfg_overrides: dict[str, Any] = {}
    if vocab_dir is not None:
        cfg_overrides["vocab_dir"] = vocab_dir
    if source_parquet is not None:
        cfg_overrides["source_parquet"] = source_parquet
    if db is not None:
        cfg_overrides["db"] = db
    if model is not None:
        cfg_overrides["model"] = model
    if model_revision is not None:
        cfg_overrides["model_revision"] = model_revision
    if device is not None:
        cfg_overrides["device"] = device
    if batch_size is not None:
        cfg_overrides["batch_size"] = batch_size
    if max_length is not None:
        cfg_overrides["max_length"] = max_length
    if pooling is not None:
        cfg_overrides["pooling"] = pooling
    if upsert_every_batches is not None:
        cfg_overrides["upsert_every_batches"] = upsert_every_batches
    if ingest_engine is not None:
        cfg_overrides["ingest_engine"] = ingest_engine
    if force:
        cfg_overrides["force"] = True
    if text_field:
        cfg_overrides["text_fields"] = text_field
    if source_text_field:
        cfg_overrides["source_text_fields"] = source_text_field
    if separator is not None:
        cfg_overrides["separator"] = separator
    if namespace is not None:
        cfg_overrides["namespace"] = namespace
    if source_namespace is not None:
        cfg_overrides["source_namespace"] = source_namespace

    cfg = EmbedConfig(**cfg_overrides)
    source_mode = cfg.source_parquet is not None
    effective_namespace = (
        cfg.namespace if not source_mode or namespace is not None else cfg.source_namespace
    )
    effective_source_text_fields = source_text_field or text_field or cfg.source_text_fields

    if device is None and cfg.device == "cpu":
        typer.secho(
            "WARNING: CUDA is not available in this Python environment; "
            "running embeddings on CPU. Install a CUDA-enabled PyTorch build "
            "and/or pass --device cuda on a GPU machine.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        logger.warning(
            "CPU fallback active: torch_version=%s, torch.version.cuda=%s, cuda_available=%s",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.is_available(),
        )

    if source_mode and csv_paths:
        typer.echo("ERROR: pass either CSV_PATH arguments or --source-parquet, not both.", err=True)
        raise typer.Exit(2)

    # Resolve input paths
    paths: list[Path]
    if source_mode:
        assert cfg.source_parquet is not None
        try:
            paths = _resolve_source_parquet_paths(cfg.source_parquet)
        except FileNotFoundError:
            typer.echo(f"ERROR: {cfg.source_parquet} not found.", err=True)
            raise typer.Exit(1) from None
        if not paths:
            typer.echo(f"ERROR: no parquet files found under {cfg.source_parquet}.", err=True)
            raise typer.Exit(1)
    elif csv_paths:
        paths = list(csv_paths)
    else:
        default = cfg.vocab_dir / "CONCEPT.csv"
        if not default.exists():
            typer.echo(
                f"ERROR: {default} not found. "
                "Use --vocab-dir or pass explicit CSV paths.",
                err=True,
            )
            raise typer.Exit(1)
        paths = [default]

    # Build filter spec
    normalized_vocabulary_ids = _split_multi_values(vocabulary_id)
    if source_mode:
        if any(
            option
            for option in (
                normalized_vocabulary_ids,
                domain_id,
                concept_class_id,
                standard_concept,
                invalid_reason,
            )
        ):
            typer.echo(
                "ERROR: Athena filter options are not supported with --source-parquet.",
                err=True,
            )
            raise typer.Exit(2)
        spec = None
    else:
        # Resolve the vocabulary filter default. With no --vocabulary-id, embedding
        # every vocabulary in CONCEPT.csv is rarely intended, so default to the
        # curated highest-yield set. The reserved sentinel "all" (case-insensitive)
        # is the explicit escape hatch back to "embed everything".
        if any(v.lower() == ALL_VOCABULARIES_SENTINEL for v in normalized_vocabulary_ids):
            normalized_vocabulary_ids = []
            typer.echo("Embedding all vocabularies (--vocabulary-id all).")
        elif not normalized_vocabulary_ids:
            normalized_vocabulary_ids = list(DEFAULT_VOCABULARY_IDS)
            typer.echo(
                "No --vocabulary-id given; defaulting to highest-yield vocabularies: "
                + ", ".join(normalized_vocabulary_ids)
                + ".\nPass --vocabulary-id all to embed every vocabulary instead."
            )
            if "CPT4" in normalized_vocabulary_ids:
                typer.echo(
                    "Note: CPT4 concept names are blank in the raw Athena download until "
                    "you run `gpu-embed cpt4` (requires a UMLS license)."
                )

        spec = FilterSpec(
            vocabulary_ids=normalized_vocabulary_ids,
            domain_ids=domain_id or [],
            concept_class_ids=concept_class_id or [],
            standard_concepts=(
                [None if v in ("", "null", "NULL") else v for v in standard_concept]
                if standard_concept
                else []
            ),
            invalid_reasons=cast(list[str | None], invalid_reason or []),
        )

    # Open store connection
    from gpu_embedder import store as st

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    # Resolve model version before CSV load so we can short-circuit unchanged inputs.
    from gpu_embedder.embed import build_embed_text, compute_model_version, embed_all, load_model

    _cached_mv = None if cfg.force else st.get_cached_model_version(
        conn, cfg.model, cfg.model_revision, cfg.pooling
    )
    if _cached_mv is not None:
        model_version = _cached_mv
        logger.info("model_version from cache: %s…", model_version[:16])
    else:
        typer.echo(
            f"Hashing model weights for {cfg.model} "
            f"(revision={cfg.model_revision or 'default'}, pooling={cfg.pooling}) …"
        )
        model_version = compute_model_version(
            cfg.model, revision=cfg.model_revision, pooling=cfg.pooling
        )
        st.upsert_model_version_cache(
            conn, cfg.model, cfg.model_revision, cfg.pooling, model_version
        )
    if source_mode:
        hash_extra = {
            "input_kind": "source_parquet",
            "text_fields": effective_source_text_fields,
            "separator": cfg.separator,
        }
    else:
        hash_extra = {
            "input_kind": "athena_csv",
            "text_fields": cfg.text_fields,
            "separator": cfg.separator,
        }
    filter_hash = filter_spec_hash(spec, effective_namespace, extra=hash_extra)

    # Load only CSVs that changed for this (model_version, filter_hash).
    # Each ingested_fingerprints entry is (path, fingerprint, row_count) where
    # row_count is this file's *post-filter, pre-dedup* row count. Cross-file
    # concept_id de-duplication happens later, so the summed skipped/loaded row
    # counts below are reporting figures only — they can exceed the number of
    # distinct concepts actually stored and must not be treated as authoritative.
    filtered = []
    precomputed_source_texts: dict[int, str] = {}
    ingested_fingerprints: list[tuple[Path, dict[str, Any], int]] = []
    skipped_unchanged = 0
    skipped_unchanged_rows = 0

    for p in paths:
        csv_path = str(p.resolve())
        stored = cast(dict[str, Any] | None, st.get_csv_fingerprint(conn, csv_path, model_version, filter_hash))

        stat = p.stat()
        size_bytes = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)

        if stored is not None and not cfg.force:
            stored_size = int(stored["size_bytes"])
            stored_mtime = int(stored["mtime_ns"])

            if stored_size == size_bytes and stored_mtime == mtime_ns:
                skipped_unchanged += 1
                skipped_unchanged_rows += int(stored["row_count"])
                logger.info(
                    "Skipping unchanged CSV by size/mtime: %s (rows=%d)",
                    csv_path,
                    int(stored["row_count"]),
                )
                continue

            current_fp = cast(dict[str, Any], compute_csv_fingerprint(p))
            if str(stored["sha256"]) == str(current_fp["sha256"]):
                skipped_unchanged += 1
                skipped_unchanged_rows += int(stored["row_count"])
                logger.info(
                    "Skipping unchanged CSV by SHA-256: %s (rows=%d)",
                    csv_path,
                    int(stored["row_count"]),
                )
                continue

            if source_mode:
                source_rows = read_source_parquet(
                    p,
                    namespace=effective_namespace,
                    text_fields=effective_source_text_fields,
                    separator=cfg.separator,
                )
                loaded = source_rows.rows
                for concept_id, embed_text in source_rows.embed_texts.items():
                    precomputed_source_texts.setdefault(concept_id, embed_text)
            else:
                loaded = read_csv(
                    p,
                    spec=spec,
                    engine=cfg.ingest_engine,
                    namespace=effective_namespace,
                )
            filtered.extend(loaded)
            ingested_fingerprints.append((p, current_fp, len(loaded)))
            continue

        # Read CSV first so the file is warm in the OS page cache, then hash
        # it.  Computing the SHA-256 on a cold file before read_csv caused two
        # sequential full scans of CONCEPT.csv (~500 MB) on every first run.
        if source_mode:
            source_rows = read_source_parquet(
                p,
                namespace=effective_namespace,
                text_fields=effective_source_text_fields,
                separator=cfg.separator,
            )
            loaded = source_rows.rows
            for concept_id, embed_text in source_rows.embed_texts.items():
                precomputed_source_texts.setdefault(concept_id, embed_text)
        else:
            loaded = read_csv(
                p,
                spec=spec,
                engine=cfg.ingest_engine,
                namespace=effective_namespace,
            )
        current_fp = cast(dict[str, Any], compute_csv_fingerprint(p))
        filtered.extend(loaded)
        ingested_fingerprints.append((p, current_fp, len(loaded)))

    if source_mode:
        typer.echo(f"Loaded {len(filtered)} rows from source parquet input.")
    else:
        typer.echo(f"Loaded {len(filtered)} rows after {cfg.ingest_engine} filtering.")
    if skipped_unchanged:
        typer.echo(
            f"Skipped {skipped_unchanged} unchanged CSV file(s) "
            f"({skipped_unchanged_rows} filtered rows)."
        )

    fingerprints_persisted = False

    def persist_ingested_fingerprints() -> None:
        """Record fingerprints for every CSV read this run (idempotent).

        Safe to call at any point where the concepts contributed by these CSVs
        are already fully represented in the store — i.e. the normal completion
        path *and* the early-exit paths where nothing remains to embed.  This
        ensures a CSV whose bytes changed but yields no new/changed embeddings
        (or zero rows after filtering) still gets its fingerprint updated, so we
        do not re-read and re-hash the same large file on every subsequent run.
        """
        nonlocal fingerprints_persisted
        if fingerprints_persisted:
            return
        for path_obj, fingerprint, row_count in ingested_fingerprints:
            st.upsert_csv_fingerprint(
                conn,
                csv_path=str(path_obj.resolve()),
                model_version=model_version,
                filter_hash=filter_hash,
                size_bytes=int(fingerprint["size_bytes"]),
                mtime_ns=int(fingerprint["mtime_ns"]),
                sha256=str(fingerprint["sha256"]),
                row_count=row_count,
            )
        fingerprints_persisted = True

    # Deduplicate by concept_id before model loading/embedding. Keep first-seen row.
    seen_concept_ids: set[int] = set()
    deduped: list = []
    duplicate_count = 0
    for row in filtered:
        if row.concept_id in seen_concept_ids:
            duplicate_count += 1
            continue
        seen_concept_ids.add(row.concept_id)
        deduped.append(row)

    if duplicate_count:
        typer.echo(
            f"Deduplicated {duplicate_count} duplicate input rows by concept_id; "
            f"{len(deduped)} unique rows remain."
        )
    filtered = deduped

    if not filtered:
        # CSVs read this run produced no rows to embed; persist their
        # fingerprints so they are not re-read and re-hashed next time.
        persist_ingested_fingerprints()
        typer.echo("Nothing to embed.")
        raise typer.Exit(0)

    if not cfg.force:
        existing_for_model = st.count_rows(conn, model_version, namespace=effective_namespace)
        registry_rows = st.list_model_registry(conn)
        has_other_model_versions = any(
            row.model_version != model_version for row in registry_rows
        )
        if existing_for_model == 0 and has_other_model_versions:
            latest_other = next(
                (row for row in registry_rows if row.model_version != model_version),
                None,
            )
            latest_other_label = (
                latest_other.model_version[:16] + "…"
                if latest_other is not None
                else "(unknown)"
            )
            typer.secho(
                "WARNING: current model_version has 0 existing embeddings while "
                "other model_version(s) exist in this DB. This run may embed most "
                "or all concepts under a new model hash. "
                f"Current={model_version[:16]}…, latest_other={latest_other_label}.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    rev_label = cfg.model_revision or "default"
    logger.info(
        "Device diagnostics: requested=%s, torch.cuda.is_available=%s, "
        "torch.version.cuda=%s, mps=%s",
        cfg.device,
        torch.cuda.is_available(),
        torch.version.cuda,
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    )
    typer.echo(f"Loading model {cfg.model} (revision={rev_label}) on {cfg.device} …")
    mdl, tok = load_model(cfg.model, cfg.device, revision=cfg.model_revision)
    typer.echo(f"model_version={model_version[:16]}…")
    st.upsert_model_registry(
        conn,
        model_version=model_version,
        model_id=cfg.model,
        model_revision=cfg.model_revision,
        precision="fp32",
        quantization_scheme="none",
        pooling=cfg.pooling,
    )

    # Texts to feed the tokenizer, keyed by concept_id; reused for change
    # detection and embedding so build_embed_text runs at most once per row.
    texts_for_embed: dict[int, str] | None = None
    if cfg.force:
        to_embed = filtered
        if source_mode:
            texts_for_embed = {
                row.concept_id: precomputed_source_texts[row.concept_id] for row in filtered
            }
    else:
        typer.echo(
            f"Detecting which of {len(filtered)} concept(s) need embedding "
            f"(namespace={effective_namespace}) …"
        )
        # Fast path for the common single-field default avoids the per-row
        # build_embed_text call overhead across millions of rows.
        if source_mode:
            candidate_texts = {row.concept_id: precomputed_source_texts[row.concept_id] for row in filtered}
        elif cfg.text_fields == ["concept_name"]:
            candidate_texts = {row.concept_id: row.concept_name for row in filtered}
        else:
            candidate_texts = {
                row.concept_id: build_embed_text(row, cfg.text_fields, cfg.separator)
                for row in filtered
            }
        to_embed, new_count, changed_count, unchanged_count = st.classify_rows_requiring_embedding(
            conn,
            filtered,
            model_version,
            candidate_texts,
        )
        typer.echo(
            "Embedding delta: "
            f"{new_count} new, {changed_count} changed-text, {unchanged_count} unchanged."
        )
        # Keep only the texts we will actually embed, then release the rest.
        texts_for_embed = {row.concept_id: candidate_texts[row.concept_id] for row in to_embed}
        del candidate_texts
    skipped = len(filtered) - len(to_embed)
    typer.echo(f"Skipping {skipped} already-embedded, embedding {len(to_embed)} …")

    if not to_embed:
        # Every concept from these CSVs is already embedded with current text,
        # so the store is complete for them; record the (possibly updated)
        # fingerprints to avoid re-reading unchanged-result files each run.
        persist_ingested_fingerprints()
        typer.echo("Nothing new to embed. Use --force to re-embed.")
        raise typer.Exit(0)

    checkpoint_size = cfg.batch_size * cfg.upsert_every_batches
    typer.echo(
        "Checkpointing writes every "
        f"{cfg.upsert_every_batches} batch(es) ({checkpoint_size} rows max per upsert)."
    )

    total_embedded = 0
    total_embed_seconds = 0.0
    total_write_seconds = 0.0
    total_checkpoints = 0

    for chunk_start in range(0, len(to_embed), checkpoint_size):
        chunk_rows = to_embed[chunk_start : chunk_start + checkpoint_size]
        chunk_end = chunk_start + len(chunk_rows)

        embed_started = time.perf_counter()
        embedded_chunk = embed_all(
            chunk_rows,
            mdl,
            tok,
            cfg.device,
            cfg.batch_size,
            cfg.max_length,
            cfg.text_fields,
            cfg.separator,
            model_version,
            pooling=cfg.pooling,
            precomputed_texts=texts_for_embed,
        )
        total_embed_seconds += time.perf_counter() - embed_started

        write_started = time.perf_counter()
        st.upsert_rows(
            conn,
            embedded_chunk,
            mode=cfg.write_mode,
            refresh_view=False,
        )
        total_write_seconds += time.perf_counter() - write_started

        total_embedded += len(embedded_chunk)
        total_checkpoints += 1
        typer.echo(
            "Checkpoint "
            f"{total_checkpoints}: wrote rows {chunk_start + 1}-{chunk_end} "
            f"({total_embedded}/{len(to_embed)} total embedded)."
        )

    typer.echo(f"Embedding phase: {total_embed_seconds:.2f}s for {total_embedded} rows.")
    typer.echo(f"Write phase: {total_write_seconds:.2f}s for {total_embedded} rows.")

    # Refresh logical view once after all checkpoint shards are written
    # (no-op for the duckdb backend).
    st.refresh_view(conn)

    total = st.count_rows(conn, model_version, namespace=effective_namespace)
    typer.echo(
        f"Done. Embedded {total_embedded} concepts. "
        f"Total stored for this model version (namespace={effective_namespace}): {total}."
    )

    persist_ingested_fingerprints()


# ---------------------------------------------------------------------------
# cpt4 subcommand
# ---------------------------------------------------------------------------

@app.command("cpt4")
def cpt4_cmd(
    vocab_dir: Annotated[
        Path | None,
        typer.Option(
            envvar="GPU_EMBED_VOCAB_DIR",
            help="Directory containing cpt4.jar and Athena CSVs",
        ),
    ] = None,
    jar: Annotated[
        Path | None,
        typer.Option(envvar="CPT4_JAR", help="Explicit path to cpt4.jar"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(envvar="UMLS_API_KEY", help="UMLS API key (prefer setting in .env)"),
    ] = None,
) -> None:
    """Populate CPT-4 concept names via the Athena-provided Java tool (requires UMLS license)."""
    load_dotenv(dotenv_path=Path(".env"), override=False)
    cdm_version = "5"

    # Resolve vocab dir
    effective_vocab_dir = (
        vocab_dir or Path(os.environ.get("GPU_EMBED_VOCAB_DIR", "athena_vocab"))
    ).resolve()

    # Resolve jar path
    effective_jar = (
        jar or Path(os.environ.get("CPT4_JAR", str(effective_vocab_dir / "cpt4.jar")))
    ).resolve()

    # Resolve API key (never log the full value)
    effective_key = api_key or os.environ.get("UMLS_API_KEY", "")
    if not effective_key:
        typer.echo(
            "ERROR: UMLS_API_KEY is not set. "
            "Set it in .env or pass --api-key.",
            err=True,
        )
        raise typer.Exit(1)

    java_executable = _resolve_java_executable()
    if java_executable is None:
        typer.echo(
            "ERROR: Java not found. Install JRE ≥ 11, or set JAVA_HOME, or add java to PATH.",
            err=True,
        )
        raise typer.Exit(1)

    # Guard: jar exists
    if not effective_jar.exists():
        typer.echo(
            f"ERROR: cpt4.jar not found at {effective_jar}. "
            "Check CPT4_JAR in .env or pass --jar.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Running CPT-4 population in {effective_vocab_dir} …")
    try:
        subprocess.run(
            [
                str(java_executable),
                f"-Dumls-apikey={effective_key}",
                "-jar",
                str(effective_jar),
                cdm_version,
            ],
            check=True,
            cwd=effective_vocab_dir,
        )
    except subprocess.CalledProcessError as exc:
        # Redact the API key from any exception message before printing
        msg = str(exc).replace(effective_key, "<REDACTED>")
        typer.echo(f"ERROR: CPT-4 Java process failed: {msg}", err=True)
        sys.exit(exc.returncode)

    typer.echo("CPT-4 population complete.")


# ---------------------------------------------------------------------------
# migrate-store subcommand
# ---------------------------------------------------------------------------


@app.command("migrate-store")
def migrate_store_cmd(
    db: Annotated[
        Path | None,
        typer.Option(
            envvar="GPU_EMBED_DB",
            help="Legacy .duckdb path (or store path) to migrate/initialize",
        ),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help="Move existing parquet store directory aside before migration",
        ),
    ] = False,
) -> None:
    """Migrate or initialize the parquet embeddings store without heavy summaries."""
    from gpu_embedder import store as st

    cfg = EmbedConfig(**cast(dict[str, Any], {"db": db} if db is not None else {}))

    store_root = cfg.db.with_suffix("") if cfg.db.suffix.lower() == ".duckdb" else cfg.db
    if reset and store_root.exists():
        backup_dir = store_root.parent / f"{store_root.name}_backup_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(store_root), str(backup_dir))
        typer.echo(f"Moved existing store to backup: {backup_dir}")

    conn = st.open_db(store_root)
    st.ensure_schema(conn)

    if cfg.db.suffix.lower() == ".duckdb":
        typer.echo(
            "Migration/initialization complete for legacy source: "
            f"{cfg.db} -> {store_root}"
        )
    else:
        typer.echo(f"Store initialization complete: {store_root}")


# ---------------------------------------------------------------------------
# migrate-lance subcommand
# ---------------------------------------------------------------------------


@app.command("migrate-lance")
def migrate_lance_cmd(
    db: Annotated[
        Path | None,
        typer.Option(
            envvar="GPU_EMBED_DB",
            help="Destination Lance store path (must end in .lance)",
        ),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option(
            "--from",
            help="Legacy .duckdb file to migrate from (default: <db>.duckdb sibling)",
        ),
    ] = None,
    batch_rows: Annotated[
        int,
        typer.Option("--batch-rows", min=1, help="Rows per streamed Arrow batch"),
    ] = 25_000,
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help="Move an existing Lance store aside before migrating",
        ),
    ] = False,
) -> None:
    """Migrate a legacy .duckdb embeddings table into a Lance store.

    Streaming and re-runnable: ATTACHes the legacy DuckDB read-only and streams
    Arrow batches into the Lance dataset. If the target already holds rows it is
    left untouched (pass --reset to move it aside and re-migrate).
    """
    from gpu_embedder import store as st

    cfg = EmbedConfig(**cast(dict[str, Any], {"db": db} if db is not None else {}))

    if cfg.db.suffix.lower() != ".lance":
        typer.echo(
            f"ERROR: --db must be a Lance store path ending in .lance; got {cfg.db}.",
            err=True,
        )
        raise typer.Exit(2)

    legacy = source if source is not None else cfg.db.with_suffix(".duckdb")
    if not legacy.exists() or not legacy.is_file():
        typer.echo(
            f"ERROR: legacy DuckDB store not found: {legacy}. Pass --from <path.duckdb>.",
            err=True,
        )
        raise typer.Exit(1)

    if reset and cfg.db.exists():
        backup_dir = (
            cfg.db.parent
            / f"{cfg.db.name}_backup_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.move(str(cfg.db), str(backup_dir))
        typer.echo(f"Moved existing Lance store to backup: {backup_dir}")

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    typer.echo(f"Migrating {legacy} -> {cfg.db} (streaming {batch_rows:,} rows/batch) …")
    migrated = st.migrate_duckdb_to_lance(conn, legacy, batch_rows=batch_rows)
    if migrated == 0:
        typer.echo(
            "Nothing migrated: the target already holds rows, or the legacy store "
            "is empty. Use --reset to re-migrate from scratch."
        )
    else:
        typer.echo(f"Migrated {migrated:,} embedding(s) into {cfg.db}.")


# ---------------------------------------------------------------------------
# compact subcommand
# ---------------------------------------------------------------------------


@app.command("compact")
def compact_cmd(
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Lance store path to compact (.lance)"),
    ] = None,
    cleanup_older_than_days: Annotated[
        float,
        typer.Option(
            "--cleanup-older-than-days",
            min=0.0,
            help="Prune Lance versions older than this many days (retention window)",
        ),
    ] = 7.0,
    no_cleanup: Annotated[
        bool,
        typer.Option("--no-cleanup", help="Compact only; keep all old versions for time-travel"),
    ] = False,
) -> None:
    """Compact a Lance store: bin-pack fragments and prune old versions.

    Reads stay correct without compaction (deletion vectors already dedupe), so
    this is optional maintenance. It is a *writer* — run it only when `embed` is
    idle, never concurrently with a live embed run.
    """
    from gpu_embedder import store as st

    cfg = EmbedConfig(**cast(dict[str, Any], {"db": db} if db is not None else {}))

    if cfg.db.suffix.lower() != ".lance":
        typer.echo(
            f"ERROR: compact applies only to a Lance store (a .lance path); got {cfg.db}.",
            err=True,
        )
        raise typer.Exit(2)

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    typer.echo(f"Compacting Lance store: {cfg.db} …")
    metrics = st.compact(
        conn,
        cleanup_older_than_days=None if no_cleanup else cleanup_older_than_days,
    )
    typer.echo("Compaction complete.")
    typer.echo(
        f"  Fragments removed/added: {metrics['fragments_removed']}/{metrics['fragments_added']}"
    )
    if no_cleanup:
        typer.echo("  Old versions retained (--no-cleanup).")
    else:
        typer.echo(
            f"  Old versions removed: {metrics['versions_removed']} "
            f"({metrics['bytes_removed']:,} bytes reclaimed)"
        )


# ---------------------------------------------------------------------------
# export subcommand
# ---------------------------------------------------------------------------


@app.command("export")
def export_cmd(
    output_dir: Annotated[
        Path,
        typer.Argument(help="Destination root directory for parquet export"),
    ],
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Embedding store path to export from"),
    ] = None,
    model_version_prefix: Annotated[
        str | None,
        typer.Option(
            "--model-version",
            help=(
                "Export only rows for model versions starting with this prefix "
                "(default: most recent)"
            ),
        ),
    ] = None,
    pooling: Annotated[
        str | None,
        typer.Option(
            "--pooling",
            help=(
                "Disambiguate by pooling strategy: cls or mean. Required when the "
                "selection matches both a cls and a mean version of the same weights"
            ),
        ),
    ] = None,
    vocabulary_id: Annotated[
        list[str] | None,
        typer.Option(
            "--vocabulary-id",
            help="Limit export to these vocabulary IDs (repeatable or comma-delimited)",
        ),
    ] = None,
    namespace: Annotated[
        str | None,
        typer.Option(
            "--namespace",
            help="Export only this identity namespace (default: all namespaces)",
        ),
    ] = None,
    shard_rows: Annotated[
        int,
        typer.Option(
            "--shard-rows",
            min=1,
            help="Maximum rows per parquet file shard",
        ),
    ] = 50_000,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing parquet shard files if present",
        ),
    ] = False,
    compression: Annotated[
        str,
        typer.Option(
            "--compression",
            help="Parquet compression codec (for example zstd or snappy)",
        ),
    ] = "snappy",
) -> None:
    """Export embeddings to Hive-partitioned parquet by model_version and vocabulary."""
    from gpu_embedder import store as st
    from gpu_embedder.report import list_model_versions

    cfg = EmbedConfig(**cast(dict[str, Any], {"db": db} if db is not None else {}))
    allowed_compressions = {"zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"}
    normalized_compression = compression.lower()
    if normalized_compression not in allowed_compressions:
        typer.echo(
            "ERROR: unsupported compression codec. "
            f"Choose one of: {', '.join(sorted(allowed_compressions))}",
            err=True,
        )
        raise typer.Exit(1)

    allowed_poolings = {"cls", "mean"}
    requested_pooling = pooling.lower() if pooling is not None else None
    if requested_pooling is not None and requested_pooling not in allowed_poolings:
        typer.echo(
            "ERROR: unsupported pooling. "
            f"Choose one of: {', '.join(sorted(allowed_poolings))}",
            err=True,
        )
        raise typer.Exit(1)

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    versions = list_model_versions(conn)
    if not versions:
        typer.echo("No embeddings found in the database.")
        raise typer.Exit(0)

    # Map each stored model_version to its pooling + model_id via the registry.
    # Versions absent from the registry (legacy stores) are treated as cls, matching
    # the registry's COALESCE(pooling, 'cls') default.
    registry = {e.model_version: e for e in st.list_model_registry(conn)}

    def _pooling_of(model_version: str) -> str:
        entry = registry.get(model_version)
        return entry.pooling if entry is not None else "cls"

    # Candidate versions, most-recent-first, narrowed by --model-version and --pooling.
    candidates = list(versions)
    if model_version_prefix:
        candidates = [
            v for v in candidates if v.model_version.startswith(model_version_prefix)
        ]
        if not candidates:
            typer.echo(
                f"No model version starting with '{model_version_prefix}' found.",
                err=True,
            )
            raise typer.Exit(1)
    if requested_pooling is not None:
        candidates = [v for v in candidates if _pooling_of(v.model_version) == requested_pooling]
        if not candidates:
            typer.echo(
                f"No model version with pooling='{requested_pooling}' matches the "
                "given filters.",
                err=True,
            )
            raise typer.Exit(1)

    distinct_poolings = {_pooling_of(v.model_version) for v in candidates}
    if requested_pooling is None and len(distinct_poolings) > 1:
        typer.echo(
            "ERROR: selection is ambiguous across pooling strategies. Re-run with "
            "--pooling {cls|mean}. Candidates:",
            err=True,
        )
        for v in candidates:
            entry = registry.get(v.model_version)
            model_id_label = entry.model_id if entry is not None else "(unregistered)"
            typer.echo(
                f"  {v.short_hash}…  pooling={_pooling_of(v.model_version):4}  "
                f"{v.count:,} row(s)  {model_id_label}",
                err=True,
            )
        raise typer.Exit(1)

    selected_model_version = candidates[0].model_version

    normalized_vocabulary_ids = _split_multi_values(vocabulary_id)

    vocab_sql = """
        SELECT DISTINCT vocabulary_id
        FROM concept_embeddings
        WHERE model_version = ?
        ORDER BY vocabulary_id NULLS LAST
    """
    vocab_rows = conn.execute(vocab_sql, [selected_model_version]).fetchall()
    available_vocabularies = [row[0] for row in vocab_rows]

    if normalized_vocabulary_ids:
        requested_vocab_set = set(normalized_vocabulary_ids)
        available_vocab_set = {v for v in available_vocabularies if v is not None}
        missing = sorted(requested_vocab_set - available_vocab_set)
        if missing:
            typer.echo(
                "Requested vocabulary_id values not found for this model version: "
                + ", ".join(missing),
                err=True,
            )
            raise typer.Exit(1)
        vocabularies_to_export: list[str | None] = [
            v for v in available_vocabularies if v is not None and v in requested_vocab_set
        ]
    else:
        vocabularies_to_export = available_vocabularies

    if not vocabularies_to_export:
        typer.echo("No rows match the requested export filters.")
        raise typer.Exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Hive-style partition root mirroring the parquet store / `migrate-store`
    # layout (`model_version=<digest>/vocabulary_id=<value>/`). Keeping the two
    # layouts identical means a single uniform stage layout in S3/Snowflake and
    # lets exports of different model versions coexist under one OUTPUT_DIR
    # without colliding on `part-*.parquet` filenames. Pooling is already folded
    # into the digest, so cls and mean land under distinct model_version dirs.
    model_dir = output_dir / f"model_version={selected_model_version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    selected_pooling = _pooling_of(selected_model_version)
    typer.echo(f"Export root: {output_dir}")
    typer.echo(f"model_version={selected_model_version[:16]}…  pooling={selected_pooling}")
    typer.echo(
        f"Sharding by up to {shard_rows:,} row(s) per file with {normalized_compression} "
        "compression."
    )

    total_rows_exported = 0
    total_files_written = 0
    total_files_skipped = 0

    # Optional namespace filter, applied to both the count and the COPY.
    ns_predicate = "" if namespace is None else " AND namespace = ?"
    ns_param: list[object] = [] if namespace is None else [namespace]

    count_sql = f"""
        SELECT COUNT(*)
        FROM concept_embeddings
        WHERE model_version = ?
          AND ((? IS NULL AND vocabulary_id IS NULL) OR vocabulary_id = ?)
          {ns_predicate}
    """

    for vocab in vocabularies_to_export:
        vocab_value = vocab
        vocab_label = vocab_value if vocab_value is not None else st.NULL_VOCAB_PARTITION
        vocab_dir = model_dir / f"vocabulary_id={vocab_label}"
        vocab_dir.mkdir(parents=True, exist_ok=True)

        count_row = conn.execute(
            count_sql,
            [selected_model_version, vocab_value, vocab_value, *ns_param],
        ).fetchone()
        vocab_count = int(count_row[0]) if count_row else 0
        if vocab_count == 0:
            continue

        file_count = (vocab_count + shard_rows - 1) // shard_rows
        typer.echo(
            f"Exporting vocabulary_id={vocab_value or '(null)'}: "
            f"{vocab_count:,} row(s) -> {file_count} file(s)"
        )

        for shard_index in range(file_count):
            start_rn = shard_index * shard_rows + 1
            end_rn = min((shard_index + 1) * shard_rows, vocab_count)
            shard_path = vocab_dir / f"part-{shard_index:05d}.parquet"

            if shard_path.exists() and not overwrite:
                total_files_skipped += 1
                continue

            # Write to a sibling .tmp file first, then atomically rename to the
            # final path on success.  This ensures a re-run never silently skips
            # a shard that was left incomplete by an interrupted previous export.
            tmp_path = shard_path.with_suffix(".parquet.tmp")
            escaped_tmp_path = tmp_path.as_posix().replace("'", "''")
            export_sql = f"""
                COPY (
                    SELECT
                        namespace,
                        concept_id,
                        concept_name,
                        domain_id,
                        vocabulary_id,
                        concept_class_id,
                        standard_concept,
                        concept_code,
                        invalid_reason,
                        embedding,
                        embed_text,
                        model_version,
                        embedded_at,
                        source_id,
                        mapping_wave
                    FROM (
                        SELECT
                            namespace,
                            concept_id,
                            concept_name,
                            domain_id,
                            vocabulary_id,
                            concept_class_id,
                            standard_concept,
                            concept_code,
                            invalid_reason,
                            embedding,
                            embed_text,
                            model_version,
                            embedded_at,
                            source_id,
                            mapping_wave,
                            row_number() OVER (ORDER BY concept_id) AS rn
                        FROM concept_embeddings
                        WHERE model_version = ?
                          AND ((? IS NULL AND vocabulary_id IS NULL) OR vocabulary_id = ?)
                          {ns_predicate}
                    ) ranked
                    WHERE rn BETWEEN ? AND ?
                ) TO '{escaped_tmp_path}'
                (FORMAT PARQUET, COMPRESSION {normalized_compression})
            """
            try:
                conn.execute(
                    export_sql,
                    [selected_model_version, vocab_value, vocab_value, *ns_param, start_rn, end_rn],
                )
                tmp_path.replace(shard_path)  # atomic on same filesystem
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            total_files_written += 1
            total_rows_exported += end_rn - start_rn + 1

    typer.echo(
        "Export complete. "
        f"Wrote {total_rows_exported:,} row(s) across {total_files_written} file(s)."
    )
    if total_files_skipped:
        typer.echo(f"Skipped {total_files_skipped} existing file(s) (use --overwrite to replace).")


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


@app.command("status")
def status_cmd(
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Embedding store path to inspect"),
    ] = None,
    model_version_prefix: Annotated[
        str | None,
        typer.Option(
            "--model-version",
            help="Limit breakdown to the model version starting with this prefix",
        ),
    ] = None,
    backfill_from_logs: Annotated[
        bool,
        typer.Option(
            "--backfill-from-logs",
            help="Parse logs and upsert missing model hash mappings before listing",
        ),
    ] = False,
    log_dir: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_LOG_DIR", help="Directory containing gpu-embed logs"),
    ] = None,
) -> None:
    """Show a summary of what is currently stored in the embeddings store."""
    from gpu_embedder import store as st
    from gpu_embedder.report import embedded_summary, list_model_versions

    cfg_overrides: dict[str, Any] = {}
    if db is not None:
        cfg_overrides["db"] = db
    if log_dir is not None:
        cfg_overrides["log_dir"] = log_dir
    cfg = EmbedConfig(**cfg_overrides)

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    if backfill_from_logs:
        result = _backfill_model_registry_from_logs(conn, log_dir=cfg.log_dir)
        if result is not None:
            added, failed = result
            typer.echo(f"Backfill complete from logs: {added} mapping(s) upserted, {failed} failed.")

    versions = list_model_versions(conn)
    if not versions:
        typer.echo("No embeddings found in the database.")
        raise typer.Exit(0)

    registry_by_version = {row.model_version: row for row in st.list_model_registry(conn)}

    typer.echo(f"\nStore: {cfg.db}\n")
    typer.echo(f"{'MODEL VERSION':18}  {'CONCEPTS':>9}  {'FIRST EMBEDDED':20}  LAST EMBEDDED")
    typer.echo("-" * 75)
    for v in versions:
        registry_entry = registry_by_version.get(v.model_version)
        model_name = registry_entry.model_id if registry_entry is not None else "(unknown)"
        quant_scheme = (
            registry_entry.quantization_scheme
            if registry_entry is not None
            else "unknown"
        )
        typer.echo(
            f"{v.short_hash}…  {v.count:>9,}  "
            f"{v.first_embedded_at.strftime('%Y-%m-%d %H:%M'):20}  "
            f"{v.last_embedded_at.strftime('%Y-%m-%d %H:%M')}"
        )
        typer.echo(f"model={model_name}")
        typer.echo(f"quant={quant_scheme}")
        typer.echo(f"full_hash={v.model_version}")
        typer.echo()

    # Resolve which model version to break down
    mv_filter: str | None = None
    if model_version_prefix:
        matched = [v for v in versions if v.model_version.startswith(model_version_prefix)]
        if not matched:
            typer.echo(
                f"No model version starting with '{model_version_prefix}' found.",
                err=True,
            )
            raise typer.Exit(1)
        mv_filter = matched[0].model_version
        typer.echo(f"Breakdown for model version: {matched[0].short_hash}…\n")
    else:
        mv_filter = versions[0].model_version  # most recent
        typer.echo(f"Breakdown for most recent model version: {versions[0].short_hash}…\n")

    rows = embedded_summary(conn, model_version=mv_filter)
    if not rows:
        typer.echo("No rows found for this model version.")
        raise typer.Exit(0)

    typer.echo(f"{'VOCABULARY':22}  {'DOMAIN':22}  {'EMBEDDED':>9}")
    typer.echo("-" * 60)
    for r in rows:
        typer.echo(
            f"{r.vocabulary_id or '(null)':<22}  "
            f"{r.domain_id or '(null)':<22}  "
            f"{r.embedded:>9,}"
        )
    total = sum(r.embedded for r in rows)
    typer.echo("-" * 60)
    typer.echo(f"{'TOTAL':<22}  {'':22}  {total:>9,}\n")


@app.command("model-registry")
def model_registry_cmd(
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Embedding store path to inspect"),
    ] = None,
    backfill_from_logs: Annotated[
        bool,
        typer.Option(
            "--backfill-from-logs",
            help="Parse logs and upsert missing model hash mappings before listing",
        ),
    ] = False,
    log_dir: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_LOG_DIR", help="Directory containing gpu-embed logs"),
    ] = None,
) -> None:
    """Show model hash to model-id/revision mappings from parquet metadata."""
    from gpu_embedder import store as st

    cfg_overrides: dict[str, Any] = {}
    if db is not None:
        cfg_overrides["db"] = db
    if log_dir is not None:
        cfg_overrides["log_dir"] = log_dir

    cfg = EmbedConfig(**cfg_overrides)
    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    if backfill_from_logs:
        result = _backfill_model_registry_from_logs(conn, log_dir=cfg.log_dir)
        if result is not None:
            added, failed = result
            typer.echo(f"Backfill complete from logs: {added} mapping(s) upserted, {failed} failed.")

    rows = st.list_model_registry(conn)
    if not rows:
        typer.echo("No model registry entries found.")
        raise typer.Exit(0)

    typer.echo(f"\nStore: {cfg.db}")
    typer.echo(f"Registry source: {cfg.db}/_meta/model_registry")
    typer.echo()
    typer.echo(
        f"{'MODEL VERSION':18}  {'RECORDED AT':20}  {'REVISION':12}  "
        f"{'PRECISION':9}  {'QUANT':9}  {'POOLING':7}  MODEL ID"
    )
    typer.echo("-" * 140)
    for row in rows:
        revision_label = row.model_revision or "default"
        typer.echo(
            f"{row.model_version[:16]}…  "
            f"{row.recorded_at.strftime('%Y-%m-%d %H:%M'):20}  "
            f"{revision_label[:12]:12}  "
            f"{row.precision[:9]:9}  "
            f"{row.quantization_scheme[:9]:9}  "
            f"{row.pooling[:7]:7}  "
            f"{row.model_id}"
        )
    typer.echo()


# ---------------------------------------------------------------------------
# coverage subcommand
# ---------------------------------------------------------------------------


@app.command("coverage")
def coverage_cmd(
    csv_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Explicit CONCEPT.csv path(s) (defaults to <vocab-dir>/CONCEPT.csv)"),
    ] = None,
    vocab_dir: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_VOCAB_DIR", help="Directory containing CONCEPT.csv"),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Embedding store path to compare against"),
    ] = None,
    model_version_prefix: Annotated[
        str | None,
        typer.Option(
            "--model-version",
            help=(
                "Limit comparison to embeddings for this model version prefix "
                "(default: most recent)"
            ),
        ),
    ] = None,
    show_complete: Annotated[
        bool,
        typer.Option(
            "--show-complete/--gaps-only",
            help="Show fully-embedded groups in a separate section (enabled by default)",
        ),
    ] = True,
    csv_output: Annotated[
        Path | None,
        typer.Option(
            "--csv",
            "-o",
            help="Write the aggregated coverage report to a CSV file",
        ),
    ] = None,
) -> None:
    """Compare a CONCEPT.csv against the embeddings store to identify gaps."""
    from gpu_embedder import store as st
    from gpu_embedder.report import VocabCoverage, coverage_report, list_model_versions

    cfg_overrides: dict[str, Any] = {}
    if db is not None:
        cfg_overrides["db"] = db
    if vocab_dir is not None:
        cfg_overrides["vocab_dir"] = vocab_dir
    cfg = EmbedConfig(**cfg_overrides)

    # Resolve CSV paths
    paths: list[Path]
    if csv_paths:
        paths = list(csv_paths)
    else:
        default = cfg.vocab_dir / "CONCEPT.csv"
        if not default.exists():
            typer.echo(
                f"ERROR: {default} not found. Use --vocab-dir or pass explicit CSV path(s).",
                err=True,
            )
            raise typer.Exit(1)
        paths = [default]

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    # Resolve model version filter
    mv_filter: str | None = None
    if model_version_prefix:
        versions = list_model_versions(conn)
        matched = [v for v in versions if v.model_version.startswith(model_version_prefix)]
        if not matched:
            typer.echo(
                f"No model version starting with '{model_version_prefix}' found.",
                err=True,
            )
            raise typer.Exit(1)
        mv_filter = matched[0].model_version
        typer.echo(f"Using model version: {mv_filter[:16]}…\n")
    else:
        versions = list_model_versions(conn)
        if versions:
            mv_filter = versions[0].model_version  # most recent
            typer.echo(f"Using most recent model version: {mv_filter[:16]}…\n")
        else:
            typer.echo("No embeddings found in store — all concepts will show as gaps.\n")

    # Accumulate coverage across all provided CSVs
    agg: dict[tuple[str, str], VocabCoverage] = defaultdict(
        lambda: VocabCoverage(vocabulary_id="", domain_id="", total=0, embedded=0)
    )
    for p in paths:
        typer.echo(f"Scanning {p} …")
        for r in coverage_report(conn, p, model_version=mv_filter):
            key = (r.vocabulary_id, r.domain_id)
            existing = agg[key]
            agg[key] = VocabCoverage(
                vocabulary_id=r.vocabulary_id,
                domain_id=r.domain_id,
                total=existing.total + r.total,
                embedded=existing.embedded + r.embedded,
            )

    all_rows = sorted(agg.values(), key=lambda r: (r.vocabulary_id, r.domain_id))

    gap_rows = [r for r in all_rows if r.gap > 0]
    complete_rows = [r for r in all_rows if r.gap == 0]

    if not all_rows:
        typer.echo("No rows found in source CSV(s).")
        raise typer.Exit(0)

    if not gap_rows and not show_complete:
        typer.secho("All concepts in the source CSV(s) are fully embedded.", fg=typer.colors.GREEN)
        raise typer.Exit(0)

    if gap_rows:
        typer.echo("\nGroups With Gaps")
        typer.echo(
            f"{'VOCABULARY':22}  {'DOMAIN':22}  {'SOURCE':>9}  {'EMBEDDED':>9}  "
            f"{'GAP':>9}  COVERAGE"
        )
        typer.echo("-" * 89)
        for r in gap_rows:
            coverage_str = f"{r.pct:5.1f}%"
            typer.echo(
                f"{r.vocabulary_id or '(null)':<22}  "
                f"{r.domain_id or '(null)':<22}  "
                f"{r.total:>9,}  "
                f"{r.embedded:>9,}  "
                f"{r.gap:>9,}  "
                f"{coverage_str}"
            )
    else:
        typer.secho("\nGroups With Gaps\n(none)", fg=typer.colors.GREEN)

    if show_complete:
        if complete_rows:
            typer.echo("\nFully Embedded Groups")
            typer.echo(
                f"{'VOCABULARY':22}  {'DOMAIN':22}  {'SOURCE':>9}  {'EMBEDDED':>9}  "
                f"{'GAP':>9}  COVERAGE"
            )
            typer.echo("-" * 89)
            for r in complete_rows:
                typer.echo(
                    f"{r.vocabulary_id or '(null)':<22}  "
                    f"{r.domain_id or '(null)':<22}  "
                    f"{r.total:>9,}  "
                    f"{r.embedded:>9,}  "
                    f"{r.gap:>9,}  "
                    "100.0%"
                )
        else:
            typer.echo("\nFully Embedded Groups\n(none)")

    total_source = sum(r.total for r in all_rows)
    total_embedded = sum(r.embedded for r in all_rows)
    total_gap = total_source - total_embedded
    overall_pct = 100.0 * total_embedded / total_source if total_source else 0.0
    typer.echo("-" * 89)
    typer.echo(
        f"{'TOTAL':<22}  {'':22}  {total_source:>9,}  {total_embedded:>9,}  "
        f"{total_gap:>9,}  {overall_pct:5.1f}%\n"
    )

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "vocabulary_id",
                    "domain_id",
                    "source_count",
                    "embedded_count",
                    "gap_count",
                    "coverage_pct",
                ]
            )
            for row in all_rows:
                writer.writerow(
                    [
                        row.vocabulary_id,
                        row.domain_id,
                        row.total,
                        row.embedded,
                        row.gap,
                        round(row.pct, 1),
                    ]
                )
        typer.echo(f"Coverage CSV written: {csv_output}")


# ---------------------------------------------------------------------------
# cleanup subcommand
# ---------------------------------------------------------------------------


def _resolve_cleanup_model_version(
    versions: list[ModelVersionInfo],
    registry_by_version: dict[str, ModelRegistryEntry],
    model_version_prefix: str | None,
) -> str:
    """Resolve the model version to clean up, interactively when not specified.

    With a prefix, require a *unique* match (a destructive command should never
    guess between candidates).  Without one, render a numbered menu and prompt.
    Returns the full model_version hash. Exits the program on bad input.
    """
    def _model_label(model_version: str) -> str:
        entry = registry_by_version.get(model_version)
        if entry is None:
            return "(unknown model — not in registry)"
        revision = entry.model_revision or "default"
        return f"{entry.model_id} (revision={revision})"

    if model_version_prefix:
        matched = [v for v in versions if v.model_version.startswith(model_version_prefix)]
        if not matched:
            typer.echo(
                f"No model version starting with '{model_version_prefix}' found.",
                err=True,
            )
            raise typer.Exit(1)
        if len(matched) > 1:
            typer.echo(
                f"'{model_version_prefix}' matches {len(matched)} model versions; "
                "use a longer, unambiguous prefix:",
                err=True,
            )
            for v in matched:
                typer.echo(f"  {v.short_hash}…  {_model_label(v.model_version)}", err=True)
            raise typer.Exit(1)
        return matched[0].model_version

    typer.echo("\nStored model versions:")
    typer.echo(f"  {'#':>3}  {'MODEL VERSION':18}  {'CONCEPTS':>9}  MODEL")
    typer.echo("  " + "-" * 78)
    for idx, v in enumerate(versions, start=1):
        typer.echo(
            f"  {idx:>3}  {v.short_hash}…  {v.count:>9,}  {_model_label(v.model_version)}"
        )
    choice = typer.prompt("\nSelect a model version to clean up by number", type=int)
    if choice < 1 or choice > len(versions):
        typer.echo(f"Invalid selection: {choice}.", err=True)
        raise typer.Exit(1)
    return str(versions[choice - 1].model_version)


def _resolve_cleanup_vocabularies(
    available: list[tuple[str | None, int]],
    requested: list[str],
    all_vocabularies: bool,
) -> list[str] | None:
    """Resolve which vocabularies to delete.

    Returns ``None`` to mean "every vocabulary for the model" (a full wipe of
    the model version) or a concrete list of vocabulary IDs.  Interactive when
    neither ``--all-vocabularies`` nor ``--vocabulary-id`` is given.  Exits on
    bad input.
    """
    available_named = [vocab for vocab, _ in available if vocab is not None]

    if all_vocabularies:
        return None

    if requested:
        unknown = [v for v in requested if v not in available_named]
        if unknown:
            typer.echo(
                "Vocabulary ID(s) not present for this model version: "
                + ", ".join(unknown),
                err=True,
            )
            raise typer.Exit(1)
        # De-duplicate while preserving the order the user gave.
        seen: set[str] = set()
        ordered: list[str] = []
        for v in requested:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        return ordered

    # Interactive menu.
    typer.echo("\nVocabularies for this model version:")
    typer.echo(f"  {'#':>3}  {'VOCABULARY':24}  {'CONCEPTS':>9}")
    typer.echo("  " + "-" * 42)
    for idx, (vocab, count) in enumerate(available, start=1):
        typer.echo(f"  {idx:>3}  {vocab or '(null)':24}  {count:>9,}")
    typer.echo(f"  {'A':>3}  {'all of the above (whole model version)':<24}")
    raw = typer.prompt(
        "\nSelect vocabularies to delete (comma-separated numbers, or 'A' for all)"
    )
    answer = raw.strip().lower()
    if answer in {"a", "all"}:
        return None

    selected: list[str] = []
    seen_idx: set[int] = set()
    for piece in answer.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.isdigit():
            typer.echo(f"Invalid selection: '{piece}'.", err=True)
            raise typer.Exit(1)
        num = int(piece)
        if num < 1 or num > len(available):
            typer.echo(f"Selection out of range: {num}.", err=True)
            raise typer.Exit(1)
        if num in seen_idx:
            continue
        seen_idx.add(num)
        vocab = available[num - 1][0]
        if vocab is None:
            typer.echo(
                "The (null) vocabulary cannot be selected individually; "
                "use 'A' to delete the whole model version.",
                err=True,
            )
            raise typer.Exit(1)
        selected.append(vocab)

    if not selected:
        typer.echo("Nothing selected; aborting.", err=True)
        raise typer.Exit(1)
    return selected


@app.command("cleanup")
def cleanup_cmd(
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="Embedding store path to clean up"),
    ] = None,
    model_version_prefix: Annotated[
        str | None,
        typer.Option(
            "--model-version",
            help="Model version hash prefix to delete (must match exactly one)",
        ),
    ] = None,
    vocabulary_id: Annotated[
        list[str] | None,
        typer.Option(
            "--vocabulary-id",
            help="Vocabulary IDs to delete (repeatable or comma-delimited)",
        ),
    ] = None,
    all_vocabularies: Annotated[
        bool,
        typer.Option(
            "--all-vocabularies",
            help="Delete every vocabulary for the model version (the whole model)",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be deleted, then stop"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt (use with care)"),
    ] = False,
) -> None:
    """Delete embeddings for a chosen model version and vocabularies.

    Cautious by design: it previews exactly what will be removed and requires
    confirmation before deleting anything. Run with no options for a guided,
    interactive selection of the model version and vocabularies.
    """
    from gpu_embedder import store as st
    from gpu_embedder.report import list_model_versions

    cfg = EmbedConfig(**({"db": db} if db is not None else {}))

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    versions = list_model_versions(conn)
    if not versions:
        typer.echo("No embeddings found in the store; nothing to clean up.")
        raise typer.Exit(0)

    if all_vocabularies and vocabulary_id:
        typer.echo(
            "ERROR: pass either --all-vocabularies or --vocabulary-id, not both.",
            err=True,
        )
        raise typer.Exit(1)

    registry_by_version = {entry.model_version: entry for entry in st.list_model_registry(conn)}

    model_version = _resolve_cleanup_model_version(
        versions, registry_by_version, model_version_prefix
    )
    available = st.list_vocabulary_counts(conn, model_version)
    requested = _split_multi_values(vocabulary_id)
    vocabularies = _resolve_cleanup_vocabularies(available, requested, all_vocabularies)

    delete_count = st.count_embeddings(conn, model_version, vocabularies)
    total_for_model = st.count_rows(conn, model_version)
    entry = registry_by_version.get(model_version)
    model_label = (
        f"{entry.model_id} (revision={entry.model_revision or 'default'})"
        if entry is not None
        else "(unknown model — not in registry)"
    )

    typer.echo("\nPlanned deletion")
    typer.echo(f"  Store:          {cfg.db}")
    typer.echo(f"  Model version:  {model_version[:16]}…  {model_label}")
    if vocabularies is None:
        typer.echo("  Vocabularies:   ALL (the entire model version will be removed)")
    else:
        typer.echo(f"  Vocabularies:   {', '.join(vocabularies)}")
    typer.echo(f"  Embeddings to delete: {delete_count:,} of {total_for_model:,} for this model.")

    if delete_count == 0:
        typer.echo("\nNothing matches the selection; nothing to delete.")
        raise typer.Exit(0)

    if dry_run:
        typer.secho("\nDry run: no changes were made.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    will_empty_model = delete_count >= total_for_model

    if not yes:
        typer.secho(
            "\nWARNING: this permanently deletes embeddings and cannot be undone.",
            fg=typer.colors.RED,
        )
        # Deleting an entire model version is the highest-risk case; require the
        # user to retype its short hash, not just a y/N.
        if will_empty_model:
            typed = typer.prompt(
                f"To remove the WHOLE model version, retype its hash prefix "
                f"({model_version[:16]})"
            )
            if typed.strip() != model_version[:16]:
                typer.echo("Confirmation did not match; aborting.", err=True)
                raise typer.Exit(1)
        else:
            typer.confirm(
                f"Permanently delete {delete_count:,} embedding(s)?",
                default=False,
                abort=True,
            )

    deleted = st.delete_embeddings(
        conn, model_version=model_version, vocabulary_ids=vocabularies
    )
    fingerprints_removed = st.delete_csv_fingerprints(conn, model_version)

    remaining = st.count_rows(conn, model_version)
    if remaining == 0:
        st.delete_model_metadata(conn, model_version)
        typer.echo(
            f"Removed model registry/cache entries for {model_version[:16]}… "
            "(no embeddings remain)."
        )

    typer.secho(f"\nDeleted {deleted:,} embedding(s).", fg=typer.colors.GREEN)
    if fingerprints_removed:
        typer.echo(
            f"Invalidated {fingerprints_removed} CSV fingerprint(s) so a later "
            "`embed` run will re-read the source CSV(s)."
        )
    typer.echo(f"{remaining:,} embedding(s) remain for this model version.")
