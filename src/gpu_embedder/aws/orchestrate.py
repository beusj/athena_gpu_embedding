"""High-level AWS embedding orchestration: submit, run-shard, collect.

These three functions implement the runbook's target execution model end to end
while keeping every external dependency injectable:

* :func:`submit_run` — move the filtered Athena concepts to AWS: shard, upload
  the shards plus a :class:`RunManifest` to S3, and submit one AWS Batch array
  job covering every shard.
* :func:`run_shard` — the per-task worker: download one input shard, embed it,
  and upload the vectors back to S3. The actual embedding is delegated to an
  injected ``embed_fn`` so the orchestration is testable without a GPU.
* :func:`collect_run` — export the embeddings back: download every output shard,
  validate dimension/model-version, and merge into the local DuckDB store.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_embedder.aws import artifacts, sharding
from gpu_embedder.aws.artifacts import RunManifest
from gpu_embedder.aws.config import AwsConfig
from gpu_embedder.models import ConceptRow, EmbeddedRow

if TYPE_CHECKING:
    import duckdb

    from gpu_embedder.aws.batch import JobScheduler
    from gpu_embedder.aws.s3 import ObjectStore

logger = logging.getLogger(__name__)

# embed_fn(rows, manifest) -> embedded rows. Injected so submit/collect/run can
# be exercised without loading a model.
EmbedFn = Callable[[list[ConceptRow], RunManifest], list[EmbeddedRow]]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSubmission:
    run_id: str
    job_id: str
    num_shards: int
    total_rows: int


@dataclass(frozen=True)
class CollectSummary:
    run_id: str
    shards: int
    rows_imported: int
    model_version: str | None


# ---------------------------------------------------------------------------
# run id
# ---------------------------------------------------------------------------


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable, collision-resistant run id (UTC timestamp + suffix)."""
    stamp = (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def submit_run(
    rows: list[ConceptRow],
    *,
    cfg: AwsConfig,
    store: ObjectStore,
    scheduler: JobScheduler,
    model: str,
    model_revision: str | None,
    model_version: str | None,
    text_fields: list[str],
    separator: str,
    max_length: int,
    batch_size: int,
    run_id: str | None = None,
    command: list[str] | None = None,
    now: datetime | None = None,
) -> RunSubmission:
    """Shard *rows*, upload them with a manifest, and submit a Batch array job."""
    if not rows:
        raise ValueError("Nothing to submit: 0 rows after filtering.")
    if not cfg.job_queue or not cfg.job_definition:
        raise ValueError(
            "AWS Batch is not configured. Set GPU_EMBED_AWS_JOB_QUEUE and "
            "GPU_EMBED_AWS_JOB_DEFINITION (or pass --job-queue/--job-definition)."
        )

    cfg.require_bucket()
    run_id = run_id or new_run_id(now)
    shards = sharding.make_shards(rows, cfg.shard_size)
    if len(shards) > cfg.max_array_size:
        raise ValueError(
            f"{len(shards)} shards exceed max_array_size={cfg.max_array_size}. "
            "Increase --shard-size or --max-array-size, or narrow the filter."
        )

    manifest = RunManifest(
        run_id=run_id,
        created_at=now or datetime.now(tz=UTC),
        model=model,
        model_revision=model_revision,
        model_version=model_version,
        embedding_dim=cfg.embedding_dim,
        text_fields=text_fields,
        separator=separator,
        max_length=max_length,
        batch_size=batch_size,
        num_shards=len(shards),
        total_rows=len(rows),
    )

    with tempfile.TemporaryDirectory(prefix="gpu_embedder_submit_") as tmp:
        tmp_dir = Path(tmp)
        for index, shard in enumerate(shards):
            shard_path = tmp_dir / f"shard-{index:05d}.ndjson"
            artifacts.write_concept_rows(shard_path, shard)
            store.upload_file(shard_path, cfg.input_key(run_id, index))

    store.put_text(cfg.manifest_key(run_id), manifest.to_json())

    # Each array task derives its shard index from AWS_BATCH_JOB_ARRAY_INDEX.
    job_command = command if command is not None else [
        "gpu-embed",
        "aws-run-shard",
        "--run-id",
        run_id,
    ]
    # Prefixes are always resolved to concrete strings by AwsConfig validators.
    environment = {
        "GPU_EMBED_AWS_RUN_ID": run_id,
        "GPU_EMBED_AWS_ENVIRONMENT": cfg.environment,
        "GPU_EMBED_AWS_S3_BUCKET": cfg.require_bucket(),
        "GPU_EMBED_AWS_S3_INPUT_PREFIX": str(cfg.s3_input_prefix),
        "GPU_EMBED_AWS_S3_OUTPUT_PREFIX": str(cfg.s3_output_prefix),
    }

    job_id = scheduler.submit_array_job(
        job_name=f"gpu-embed-{run_id}",
        job_queue=cfg.job_queue,
        job_definition=cfg.job_definition,
        array_size=len(shards),
        command=job_command,
        environment=environment,
    )

    logger.info(
        "Submitted run %s as job %s (%d shards, %d rows)",
        run_id,
        job_id,
        len(shards),
        len(rows),
    )
    return RunSubmission(
        run_id=run_id, job_id=job_id, num_shards=len(shards), total_rows=len(rows)
    )


# ---------------------------------------------------------------------------
# run-shard (worker)
# ---------------------------------------------------------------------------


def run_shard(
    *,
    run_id: str,
    shard_index: int,
    cfg: AwsConfig,
    store: ObjectStore,
    embed_fn: EmbedFn,
) -> int:
    """Embed a single shard and upload the vectors. Returns rows embedded."""
    manifest = RunManifest.from_json(store.get_text(cfg.manifest_key(run_id)))

    with tempfile.TemporaryDirectory(prefix="gpu_embedder_shard_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.ndjson"
        store.download_file(cfg.input_key(run_id, shard_index), input_path)
        rows = artifacts.read_concept_rows(input_path)
        logger.info("Shard %d: embedding %d rows", shard_index, len(rows))

        embedded = embed_fn(rows, manifest)
        artifacts.validate_embedded_rows(
            embedded, embedding_dim=cfg.embedding_dim
        )

        output_path = tmp_dir / "output.ndjson"
        artifacts.write_embedded_rows(output_path, embedded)
        store.upload_file(output_path, cfg.output_key(run_id, shard_index))

    logger.info("Shard %d: uploaded %d embeddings", shard_index, len(embedded))
    return len(embedded)


# ---------------------------------------------------------------------------
# collect (export back)
# ---------------------------------------------------------------------------


def collect_run(
    *,
    run_id: str,
    cfg: AwsConfig,
    store: ObjectStore,
    conn: duckdb.DuckDBPyConnection,
    expected_model_version: str | None = None,
) -> CollectSummary:
    """Download, validate, and merge a run's output shards into DuckDB."""
    from gpu_embedder import store as duckdb_store

    all_keys = store.list_keys(cfg.output_prefix_for(run_id))
    keys = sorted(k for k in all_keys if k.endswith(".ndjson"))
    if not keys:
        raise ValueError(
            f"No output artifacts found for run {run_id} under "
            f"s3://.../{cfg.output_prefix_for(run_id)} — has the job finished?"
        )

    # Resolve the expected model version from the manifest when not given.
    if expected_model_version is None:
        try:
            manifest = RunManifest.from_json(
                store.get_text(cfg.manifest_key(run_id))
            )
            expected_model_version = manifest.model_version
        except Exception:
            logger.warning("Could not load manifest for run %s; skipping "
                           "model-version validation", run_id)

    total = 0
    seen_versions: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="gpu_embedder_collect_") as tmp:
        tmp_dir = Path(tmp)
        for key in keys:
            local_path = tmp_dir / Path(key).name
            store.download_file(key, local_path)
            rows = artifacts.read_embedded_rows(local_path)
            artifacts.validate_embedded_rows(
                rows,
                embedding_dim=cfg.embedding_dim,
                expected_model_version=expected_model_version,
            )
            duckdb_store.upsert_rows(conn, rows)
            total += len(rows)
            seen_versions.update(r.model_version for r in rows)

    resolved_version = expected_model_version
    if resolved_version is None and len(seen_versions) == 1:
        resolved_version = next(iter(seen_versions))

    logger.info(
        "Collected run %s: %d shards, %d rows imported", run_id, len(keys), total
    )
    return CollectSummary(
        run_id=run_id,
        shards=len(keys),
        rows_imported=total,
        model_version=resolved_version,
    )


# ---------------------------------------------------------------------------
# default embed function (GPU worker)
# ---------------------------------------------------------------------------


def default_embed_fn(device: str = "auto") -> EmbedFn:
    """Build an :data:`EmbedFn` that runs SapBERT locally (used in the container).

    Imports :mod:`gpu_embedder.embed` lazily so this module stays importable
    without torch/transformers loaded.
    """
    from gpu_embedder.config import _auto_device
    from gpu_embedder.embed import compute_model_version, embed_all, load_model

    resolved_device = _auto_device() if device == "auto" else device

    def _embed(rows: list[ConceptRow], manifest: RunManifest) -> list[EmbeddedRow]:
        model, tokenizer = load_model(
            manifest.model, resolved_device, revision=manifest.model_revision
        )
        model_version = manifest.model_version or compute_model_version(
            manifest.model, revision=manifest.model_revision
        )
        return embed_all(
            rows,
            model,
            tokenizer,
            resolved_device,
            manifest.batch_size,
            manifest.max_length,
            manifest.text_fields,
            manifest.separator,
            model_version,
        )

    return _embed
