# AWS Infrastructure Outline

This directory is a planning-level scaffold for AWS execution of gpu-embedder workloads.

Status: Terraform skeleton only (module/env interfaces defined; no AWS resources created yet).

## Layout

- envs/academic-dev/
  - Environment wiring for dev/test benchmarking.
- envs/academic-prod/
  - Environment wiring for controlled production runs.
- modules/batch_gpu/
  - AWS Batch (queue, compute environment, job definition).
- modules/storage/
  - S3 buckets/prefix policies/lifecycle for embedding artifacts.
- modules/security/
  - IAM, KMS, and audit-control primitives.

Each environment directory now includes:

- `versions.tf`
- `variables.tf`
- `main.tf`
- `outputs.tf`
- `terraform.tfvars.example`

Each module directory now includes:

- `variables.tf`
- `main.tf`
- `outputs.tf`

## Intended deployment order

1. storage
2. security
3. batch_gpu
4. env-specific wiring (academic-dev first)

## Design constraints

- Keep local execution mode as default path.
- AWS mode is optional and should be explicitly selected via config.
- S3 is the primary artifact plane (inputs, shard outputs, manifests, logs).
- Use immutable run ids and completion manifests for import safety.

## Next implementation steps

1. Choose Terraform or CDK and keep one toolchain per repo.
2. Implement concrete AWS resources in `modules/storage`, `modules/security`, and `modules/batch_gpu`.
3. Apply in `envs/academic-dev` first and benchmark g5 vs g6e.
4. Promote to `envs/academic-prod` only after cost/perf and reliability gates pass.

## Suggested workflow

From an environment folder (example: `infra/aws/envs/academic-dev`):

1. Copy `terraform.tfvars.example` to `terraform.tfvars` and fill values.
2. Run `terraform init`.
3. Run `terraform plan`.
4. Run `terraform apply` only after module resources are implemented and reviewed.
