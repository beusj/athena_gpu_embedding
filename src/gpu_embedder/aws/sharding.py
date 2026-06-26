"""Partition concept rows into evenly sized shards for AWS Batch array jobs."""

from __future__ import annotations

import logging

from gpu_embedder.models import ConceptRow

logger = logging.getLogger(__name__)


def make_shards(rows: list[ConceptRow], shard_size: int) -> list[list[ConceptRow]]:
    """Split *rows* into contiguous shards of at most *shard_size* rows each.

    Order is preserved, so re-running the same filter yields the same shard
    boundaries. An empty input yields an empty list.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than 0")

    shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
    logger.info(
        "Partitioned %d rows into %d shard(s) of up to %d rows",
        len(rows),
        len(shards),
        shard_size,
    )
    return shards
