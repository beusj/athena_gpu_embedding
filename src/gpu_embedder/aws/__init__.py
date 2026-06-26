"""Optional AWS execution path for gpu-embedder.

This subpackage is an *additive, opt-in* remote execution mode. It does not
change the default local `embed` path, and it never imports ``boto3`` at module
import time — the AWS SDK is only required when one of the ``aws-*``
subcommands actually runs, and is imported lazily inside :mod:`s3` /
:mod:`batch`.

The end-to-end flow mirrors the runbook in
``docs/runbooks/aws_embedding_execution_plan.md``:

1. **move to AWS** — shard the filtered Athena concepts, upload the shards and a
   run manifest to S3 (:func:`orchestrate.submit_run`);
2. **embed on AWS** — each AWS Batch array task embeds one shard and writes the
   vectors back to S3 (:func:`orchestrate.run_shard`);
3. **export back** — download the output artifacts, validate them, and merge
   them into the local DuckDB store (:func:`orchestrate.collect_run`).
"""

from __future__ import annotations

__all__ = ["config", "artifacts", "sharding", "s3", "batch", "orchestrate"]
