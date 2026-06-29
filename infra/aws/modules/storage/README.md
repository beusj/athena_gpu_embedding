# Module outline: storage

Scope:
- S3 bucket(s) or prefix policies for gpu-embed artifacts
- Encryption defaults and lifecycle policies

Key inputs (planned):
- bucket name or existing bucket reference
- kms key arn
- env name (academic-dev / academic-prod)
- lifecycle retention settings

Key outputs (planned):
- bucket name/arn
- canonical prefixes for inputs/outputs/checkpoints/logs

Recommended prefix model:
- gpu-embed/<env>/inputs/
- gpu-embed/<env>/outputs/
- gpu-embed/<env>/checkpoints/
- gpu-embed/<env>/logs/

Notes:
- Block public access.
- Require SSE-KMS.
- Keep completion manifests as source of truth for downstream import.
