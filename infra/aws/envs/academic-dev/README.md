# academic-dev environment outline

Purpose:
- Low-risk benchmark and integration environment for AWS embedding runs.

Should provision:
- Dev S3 prefixes/bucket policy scope
- Dev KMS key policy bindings
- Dev Batch queue + compute environment (Spot-first)
- Dev IAM execution roles

Suggested tags:
- Environment=academic-dev
- Project=gpu-embedder
- Owner=<team>

Exit criteria to move beyond dev:
- Resource utilization benchmark completed
- Cost per 1M concepts accepted
- Retry/interruption behavior stable
- Completion-manifest import path validated
