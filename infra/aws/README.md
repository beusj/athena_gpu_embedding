# AWS Infrastructure

Terraform for the optional AWS execution path of `gpu-embedder` (see the
repo-root README "AWS execution path" and
`docs/runbooks/aws_embedding_execution_plan.md`).

Status: **modules implement real resources**, gated behind create-vs-reuse
toggles so the same code works in an open account or a locked-down academic one.
Terraform here provisions the *control/infra plane*; the *data plane* (shard →
S3 → Batch → DuckDB) is the `gpu-embed aws-*` CLI.

## Layout

- `envs/academic-dev/` — dev/test benchmarking wiring.
- `envs/academic-prod/` — controlled production wiring.
- `modules/storage/` — S3 artifact bucket: public-access block, SSE
  (KMS or AES256), versioning, and lifecycle rules for the input/output/logs
  prefixes.
- `modules/security/` — IAM job role (scoped S3+KMS artifact access), ECS task
  execution role, and EC2 instance role/profile for the compute environment.
- `modules/batch_gpu/` — AWS Batch managed compute environment (Spot or
  on-demand), job queue, and GPU job definition.

## Create-vs-reuse toggles (read this for academic accounts)

Locked-down academic accounts frequently forbid creating IAM, KMS, or VPC
resources. Every such decision is a variable with a conservative escape hatch:

| Toggle | Default | When false… |
|--------|---------|-------------|
| `create_bucket` | `true` | Bucket is pre-provisioned; module only derives ARNs/prefixes and applies **no** changes to it. |
| `create_roles` | `true` | Supply `existing_job_role_arn`, `existing_task_execution_role_arn`, `existing_instance_profile_arn`; module passes them through. |
| `create_batch` | `true` | No compute env/queue/job-def created (outputs null) — use while waiting on GPU service-quota approval. |

Other lockdown-relevant inputs:

- `permissions_boundary_arn` — attached to every created role (often mandatory).
- `kms_key_arn` — `null` uses SSE-S3 (AES256); set it to use a CMK you were granted.
- `subnet_ids` / `security_group_ids` — **bring-your-own VPC**; the modules never
  create networking.
- GPU capacity: new accounts almost always have a **0 vCPU quota for G/VT Spot
  and On-Demand instances**. Request a Service Quota increase for the relevant
  family (e.g. "Running On-Demand G and VT instances") before `create_batch`.

## Deployment order

1. Build + push the worker image (`docker/Dockerfile.aws`) to ECR; set
   `container_image`.
2. From an env folder (start with `envs/academic-dev/`):
   ```bash
   cp terraform.tfvars.example terraform.tfvars   # fill in values
   terraform init
   terraform plan
   terraform apply
   ```
3. `terraform output cli_env` prints the exact `GPU_EMBED_AWS_*` values to put in
   your `.env` (or export) so the CLI targets what you just provisioned.
4. Promote to `envs/academic-prod/` only after cost/perf and reliability gates
   pass.

## Terraform output → CLI env var

| `terraform output` | `.env` / env var | Used by |
|--------------------|------------------|---------|
| `bucket_name` | `GPU_EMBED_AWS_S3_BUCKET` | submit / run-shard / collect |
| `prefix_scope` | `GPU_EMBED_AWS_S3_PREFIX_ROOT` + `GPU_EMBED_AWS_ENVIRONMENT` | all |
| `batch_job_queue_name` | `GPU_EMBED_AWS_JOB_QUEUE` | submit |
| `batch_job_definition_name` | `GPU_EMBED_AWS_JOB_DEFINITION` | submit |

`terraform output cli_env` emits all of these as a ready-to-use map.
