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
