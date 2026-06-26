"""Typer CLI entry point for gpu-embedder.

Subcommands:
  gpu-embed embed [OPTIONS] [CSV_PATH...]   — batch-embed concepts
  gpu-embed cpt4  [OPTIONS]                 — populate CPT-4 names via Athena Java tool

This module is intentionally thin: all logic lives in config, ingest, embed,
and store.  cli.py is excluded from coverage requirements.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from typer.main import get_command

from gpu_embedder import __version__
from gpu_embedder.config import EmbedConfig
from gpu_embedder.ingest import read_csv
from gpu_embedder.models import FilterSpec

app = typer.Typer(
    name="gpu-embed",
    help="Batch-embed OHDSI Athena concepts with SapBERT into DuckDB.",
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    invoke_without_command=True,
)

logger = logging.getLogger(__name__)


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
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="DuckDB output file"),
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
        typer.Option("--vocabulary-id", help="Filter: vocabulary IDs to include (repeatable)"),
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
    separator: Annotated[
        str | None,
        typer.Option(
            envvar="GPU_EMBED_SEPARATOR",
            help="Separator between concatenated text fields",
        ),
    ] = None,
) -> None:
    """Batch-embed Athena CONCEPT.csv rows with SapBERT and store in DuckDB."""
    # Build config, allowing CLI overrides
    cfg_overrides: dict[str, object] = {}
    if vocab_dir is not None:
        cfg_overrides["vocab_dir"] = vocab_dir
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
    if ingest_engine is not None:
        cfg_overrides["ingest_engine"] = ingest_engine
    if force:
        cfg_overrides["force"] = True
    if text_field:
        cfg_overrides["text_fields"] = text_field
    if separator is not None:
        cfg_overrides["separator"] = separator

    cfg = EmbedConfig(**cfg_overrides)

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

    # Resolve CSV paths
    paths: list[Path]
    if csv_paths:
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
    spec = FilterSpec(
        vocabulary_ids=vocabulary_id or [],
        domain_ids=domain_id or [],
        concept_class_ids=concept_class_id or [],
        standard_concepts=(
            [None if v in ("", "null", "NULL") else v for v in standard_concept]
            if standard_concept
            else []
        ),
        invalid_reasons=invalid_reason or [],
    )

    # Load all rows with DuckDB pushdown filtering
    filtered = []
    for p in paths:
        filtered.extend(read_csv(p, spec=spec, engine=cfg.ingest_engine))

    typer.echo(f"Loaded {len(filtered)} rows after {cfg.ingest_engine} filtering.")

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
        typer.echo("Nothing to embed.")
        raise typer.Exit(0)

    # Open DuckDB
    from gpu_embedder import store as st

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    # Determine skip set
    from gpu_embedder.embed import compute_model_version, embed_all, load_model

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
    model_version = compute_model_version(cfg.model, revision=cfg.model_revision)
    typer.echo(f"model_version={model_version[:16]}…")

    skip_ids: set[int] = set()
    if not cfg.force:
        skip_ids = st.get_existing_ids(conn, model_version)

    to_embed = [r for r in filtered if r.concept_id not in skip_ids]
    skipped = len(filtered) - len(to_embed)
    typer.echo(f"Skipping {skipped} already-embedded, embedding {len(to_embed)} …")

    if not to_embed:
        typer.echo("Nothing new to embed. Use --force to re-embed.")
        raise typer.Exit(0)

    embed_started = time.perf_counter()
    embedded = embed_all(
        to_embed,
        mdl,
        tok,
        cfg.device,
        cfg.batch_size,
        cfg.max_length,
        cfg.text_fields,
        cfg.separator,
        model_version,
    )
    embed_seconds = time.perf_counter() - embed_started
    typer.echo(f"Embedding phase: {embed_seconds:.2f}s for {len(embedded)} rows.")

    write_started = time.perf_counter()
    st.upsert_rows(conn, embedded)
    write_seconds = time.perf_counter() - write_started
    typer.echo(f"Write phase: {write_seconds:.2f}s for {len(embedded)} rows.")

    total = st.count_rows(conn, model_version)
    typer.echo(
        f"Done. Embedded {len(embedded)} concepts. "
        f"Total stored for this model version: {total}."
    )


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
# status subcommand
# ---------------------------------------------------------------------------


@app.command("status")
def status_cmd(
    db: Annotated[
        Path | None,
        typer.Option(envvar="GPU_EMBED_DB", help="DuckDB file to inspect"),
    ] = None,
    model_version_prefix: Annotated[
        str | None,
        typer.Option(
            "--model-version",
            help="Limit breakdown to the model version starting with this prefix",
        ),
    ] = None,
) -> None:
    """Show a summary of what is currently stored in the embeddings database."""
    from gpu_embedder import store as st
    from gpu_embedder.report import embedded_summary, list_model_versions

    cfg = EmbedConfig(**({"db": db} if db is not None else {}))

    if not cfg.db.exists():
        typer.echo(f"Database not found: {cfg.db}")
        raise typer.Exit(0)

    conn = st.open_db(cfg.db)
    st.ensure_schema(conn)

    versions = list_model_versions(conn)
    if not versions:
        typer.echo("No embeddings found in the database.")
        raise typer.Exit(0)

    typer.echo(f"\nDatabase: {cfg.db}\n")
    typer.echo(f"{'MODEL VERSION':18}  {'CONCEPTS':>9}  {'FIRST EMBEDDED':20}  LAST EMBEDDED")
    typer.echo("-" * 75)
    for v in versions:
        typer.echo(
            f"{v.short_hash}…  {v.count:>9,}  "
            f"{v.first_embedded_at.strftime('%Y-%m-%d %H:%M'):20}  "
            f"{v.last_embedded_at.strftime('%Y-%m-%d %H:%M')}"
        )
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
        typer.Option(envvar="GPU_EMBED_DB", help="DuckDB file to compare against"),
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
    """Compare a CONCEPT.csv against the embeddings database to identify gaps."""
    from gpu_embedder import store as st
    from gpu_embedder.report import VocabCoverage, coverage_report, list_model_versions

    cfg_overrides: dict[str, object] = {}
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
            typer.echo("No embeddings found in database — all concepts will show as gaps.\n")

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
