"""EmbedConfig: all runtime settings for gpu-embedder.

All fields are readable from environment variables (prefix GPU_EMBED_) and from
a .env file in the working directory.  CLI flags always override env values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbedConfig(BaseSettings):
    """Runtime configuration for the `gpu-embed embed` subcommand."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="GPU_EMBED_", extra="ignore")

    # Paths
    vocab_dir: Path = Path("athena_vocab")
    db: Path = Path("embeddings.duckdb")

    # Model
    model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    # Specific HuggingFace commit hash / branch / tag to pin the model revision.
    # None means use the upstream default branch (typically "main").
    model_revision: str | None = None
    device: str = "auto"
    batch_size: int = 256
    max_length: int = 128
    ingest_engine: Literal["duckdb", "python"] = "duckdb"

    # Text construction
    text_fields: Annotated[list[str], NoDecode] = ["concept_name"]
    separator: str = " "

    # Behaviour
    force: bool = False

    @field_validator("text_fields", mode="before")
    @classmethod
    def parse_text_fields(cls, v: object) -> list[str]:
        """Allow comma-separated string from env var."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def resolve_device(self) -> EmbedConfig:
        if self.device == "auto":
            resolved = _auto_device()
            self.device = resolved
            if resolved == "cpu":
                logger.warning(
                    "No GPU backend detected; using CPU. "
                    "torch.version.cuda=%s, cuda_available=%s, mps_available=%s",
                    torch.version.cuda,
                    torch.cuda.is_available(),
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
                )
            else:
                logger.info("Selected device=%s", resolved)
        return self
