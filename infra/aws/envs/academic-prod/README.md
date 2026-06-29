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

## Restricted-permission mode (workload-only Terraform)

Use this mode when IAM/S3 bucket governance is handled in an external admin portal and
the execution profile cannot read/manage bucket configuration resources.

Recommended settings in `terraform.tfvars`:

- `use_precreated_iam_roles = true`
- `manage_storage_resources = false`
- `precreated_bucket_name` / `precreated_bucket_arn` / `precreated_prefix_scope` (optional but recommended)

One-time migration note:

If this environment previously managed `module.storage` resources, remove those storage entries
from Terraform state before first workload-only apply to avoid planned destroys in restricted mode.

## How to run (batch-only mode)

Runbook commands are maintained in one place:

- `docs/runbooks/aws_interaction_guide.md`
