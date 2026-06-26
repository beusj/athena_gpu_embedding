# Module outline: batch_gpu

Scope:
- AWS Batch queue(s)
- Compute environment(s)
- Job definition(s) for embedding workers

Key inputs (planned):
- region
- vpc/subnet/security-group identifiers
- preferred instance families (g5, g6e)
- spot/on-demand mix
- job vcpu/memory/gpu settings
- ecr image uri (pinned digest)
- retry strategy and timeout

Key outputs (planned):
- job queue arn/name
- job definition arn/name
- compute environment arn/name

Notes:
- Keep jobs stateless; read/write artifacts via S3.
- Prefer shard durations in 5-20 minute range to reduce Spot interruption waste.
