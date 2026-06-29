# AWS Interaction Guide (Batch-Only Mode)

This guide is the operational command reference for day-to-day AWS interaction.
Use it with restricted-permission environments where IAM and bucket governance
are managed outside this repository.

## Scope

- Running Terraform in workload-only mode
- Syncing `CONCEPT.csv` to S3 safely
- Submitting and monitoring AWS Batch jobs
- Stopping queued/running jobs

Planning and architecture context remains in:

- `docs/runbooks/aws_embedding_execution_plan.md`

Data export and warehouse loading steps remain in:

- `docs/runbooks/s3_to_snowflake_load.md`

## Store and export checklist

For downstream S3/Snowflake loading flows, use `export` as the default handoff.
Only use `migrate-store` when you explicitly need a full parquet mirror of the
local `.duckdb` store.

```bash
# Default local fast-write store
uv run gpu-embed status --db embeddings.duckdb

# Standard Snowflake handoff artifact
uv run gpu-embed export exports/parquet --db embeddings.duckdb

# Optional: full parquet mirror materialization from local DuckDB
uv run gpu-embed migrate-store --db embeddings.duckdb
```

After export, sync `exports/parquet/` directly to S3 as the default path.

Expected migration behavior:

- Throughput may decline over time as larger partitions are processed.
- This is normal if progress logs continue to update with increasing
  `partitions`/`rows`/`files` counters.
- Prefer `migrate-store` for migration-only runs; `status` adds summary queries.

## Prerequisites

- Environment configured for workload-only mode:
  - `use_precreated_iam_roles=true`
  - `manage_storage_resources=false`
- AWS CLI configured with usable credentials (default profile or named profile)
- Run commands from `infra/aws/envs/academic-dev` unless noted otherwise

## 1) Authenticate and select profile

Credential precedence (highest to lowest): explicit command flags,
environment variables, then default profile/config files.

- `--profile <name>` on a command
- `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- Default profile in `~/.aws/config` and `~/.aws/credentials`

```bash
# Default profile flow
aws sso login
```

If you use a named profile, either set it once for the shell session or append
`--profile <aws-profile>` to each AWS CLI command:

```bash
export AWS_PROFILE=<aws-profile>
```

## 2) Ensure infrastructure is current

```bash
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars -auto-approve
```

## 3) Sync CONCEPT.csv to S3 (upload only if missing)

```bash
BUCKET="gpu-embedder-artifacts-chic"
KEY="gpu-embed/dev/inputs/raw/CONCEPT/CONCEPT.csv"
LOCAL="/c/Users/Jonat/Documents/Development/General/gpu_embedding/athena_vocab/CONCEPT.csv"

aws s3api head-object --bucket "$BUCKET" --key "$KEY" >/dev/null 2>&1 \
  || aws s3 cp "$LOCAL" "s3://$BUCKET/$KEY"
```

Verify object exists:

```bash
aws s3 ls "s3://$BUCKET/$KEY"
```

## 4) Submit a baseline test job

```bash
aws batch submit-job \
  --job-name gpu-embed-test-$(date +%Y%m%d-%H%M%S) \
  --job-queue gpu-embed-dev \
  --job-definition gpu-embed-worker-dev \
  --container-overrides '{"command":["gpu-embed","embed","--csv-path","s3://gpu-embedder-artifacts-chic/gpu-embed/dev/inputs/raw/CONCEPT/CONCEPT.csv"]}'
```

## 5) Submit a limited run (batch size + vocab filters)

```bash
aws batch submit-job \
  --job-name gpu-embed-limited-$(date +%Y%m%d-%H%M%S) \
  --job-queue gpu-embed-dev \
  --job-definition gpu-embed-worker-dev \
  --container-overrides '{"command":["gpu-embed","embed","--batch-size","32","--vocabulary-id","LOINC","--vocabulary-id","SNOMED","--csv-path","s3://gpu-embedder-artifacts-chic/gpu-embed/dev/inputs/raw/CONCEPT/CONCEPT.csv"]}'
```

## 6) Monitor jobs

```bash
aws batch list-jobs --job-queue gpu-embed-dev --job-status RUNNABLE
aws batch list-jobs --job-queue gpu-embed-dev --job-status RUNNING
aws batch list-jobs --job-queue gpu-embed-dev --job-status SUCCEEDED
aws batch list-jobs --job-queue gpu-embed-dev --job-status FAILED
```

For one job ID:

```bash
aws batch describe-jobs --jobs <job-id>
```

## 7) Stop jobs

Stop queued jobs:

```bash
aws batch cancel-job --job-id <job-id> --reason "Stopped by operator"
```

Stop running jobs:

```bash
aws batch terminate-job --job-id <job-id> --reason "Stopped by operator"
```

If the execution profile lacks these permissions, use the admin profile or request:

- `batch:CancelJob`
- `batch:TerminateJob`

## Common failures

- `AccessDenied ... batch:SubmitJob`: execution profile missing submit permission.
- Job stuck `RUNNABLE`: no worker capacity yet (compute placement/capacity issue).
- Runtime file errors after start: CSV object missing or command path invalid.