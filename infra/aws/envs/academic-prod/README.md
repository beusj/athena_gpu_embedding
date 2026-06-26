# academic-prod environment outline

Purpose:
- Controlled production-grade execution for full embedding runs.

Should provision:
- Prod S3 prefixes/bucket policy scope
- Prod KMS key policy bindings
- Prod Batch queue + compute environment (Spot with on-demand fallback)
- Prod IAM execution roles with least privilege

Operational requirements:
- Explicit run approvals/change control
- Budget alarms and retry-storm alarms enabled
- S3 data event audit enabled
- Lifecycle policies applied

Promotion gate from dev:
- Proven benchmark profile
- Successful full dry-run with manifest-based import
