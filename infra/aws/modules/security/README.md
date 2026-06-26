# Module outline: security

Scope:
- IAM roles/policies for Batch execution and artifact access
- KMS key and policy bindings
- Audit wiring prerequisites

Key inputs (planned):
- env name
- bucket arn/prefix scopes
- kms key settings
- principal role mappings

Key outputs (planned):
- batch task execution role arn
- batch job role arn
- kms key arn (if managed here)

Baseline controls:
- Least-privilege IAM (prefix-scoped read/write)
- No secrets in command args/environment when avoidable
- CloudTrail management + S3 data events enabled in environment
