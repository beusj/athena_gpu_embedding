"""AwsConfig: settings for the optional AWS execution path.

All fields read from environment variables (prefix ``GPU_EMBED_AWS_``) and from
a ``.env`` file in the working directory, mirroring :class:`EmbedConfig`. These
settings are intentionally separate from :class:`gpu_embedder.config.EmbedConfig`
so that the local embed path carries no AWS configuration surface.
"""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AwsConfig(BaseSettings):
    """Runtime configuration for the ``gpu-embed aws-*`` subcommands."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GPU_EMBED_AWS_", extra="ignore"
    )

    # Connectivity
    region: str | None = None

    # Environment label. Matches the Terraform `environment` (e.g. academic-dev,
    # academic-prod) so the S3 layout below lines up with `infra/aws` outputs.
    environment: str = "academic-dev"

    # S3 layout. The default prefixes mirror the Terraform `prefix_scope`
    # (``<s3_prefix_root>/<environment>``), so a run submitted by the CLI lands
    # exactly where the provisioned bucket policy/lifecycle rules expect it.
    # Either prefix may be overridden explicitly; otherwise it is derived.
    s3_bucket: str | None = None
    s3_prefix_root: str = "gpu-embed"
    s3_input_prefix: str | None = None
    s3_output_prefix: str | None = None

    # AWS Batch
    job_queue: str | None = None
    job_definition: str | None = None
    spot_preferred: bool = True

    # Sharding
    # Rows per shard. One AWS Batch array task is submitted per shard, so this
    # trades parallelism against per-task startup overhead.
    shard_size: int = 50_000
    # Hard ceiling on the number of array tasks for a single run.
    max_array_size: int = 1_000

    # Validation invariant: SapBERT emits 768-dim vectors.
    embedding_dim: int = 768

    @model_validator(mode="after")
    def _validate_sizes(self) -> AwsConfig:
        if self.shard_size <= 0:
            raise ValueError("shard_size must be greater than 0")
        if self.max_array_size <= 0:
            raise ValueError("max_array_size must be greater than 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        return self

    @model_validator(mode="after")
    def _resolve_prefixes(self) -> AwsConfig:
        """Derive env-scoped prefixes from the prefix root when not overridden."""
        base = f"{self.s3_prefix_root}/{self.environment}"
        if self.s3_input_prefix is None:
            self.s3_input_prefix = f"{base}/input"
        if self.s3_output_prefix is None:
            self.s3_output_prefix = f"{base}/output"
        return self

    # -- derived helpers ----------------------------------------------------

    def require_bucket(self) -> str:
        """Return the configured S3 bucket or raise a clear error."""
        if not self.s3_bucket:
            raise ValueError(
                "No S3 bucket configured. Set GPU_EMBED_AWS_S3_BUCKET in .env "
                "or pass --s3-bucket."
            )
        return self.s3_bucket

    def input_key(self, run_id: str, shard_index: int) -> str:
        """S3 key for a single input shard of *run_id*."""
        return f"{self.s3_input_prefix}/{run_id}/shard-{shard_index:05d}.ndjson"

    def output_key(self, run_id: str, shard_index: int) -> str:
        """S3 key for a single output (embeddings) shard of *run_id*."""
        return f"{self.s3_output_prefix}/{run_id}/shard-{shard_index:05d}.ndjson"

    def manifest_key(self, run_id: str) -> str:
        """S3 key for a run's manifest JSON."""
        return f"{self.s3_input_prefix}/{run_id}/manifest.json"

    def output_prefix_for(self, run_id: str) -> str:
        """S3 prefix under which all output shards for *run_id* live."""
        return f"{self.s3_output_prefix}/{run_id}/"
