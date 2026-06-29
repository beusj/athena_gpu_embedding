# Runbook Plan: AWS-Based Embedding Execution (Out-of-Main-Path)

This document is a planning-only proposal for running embedding workloads on AWS.
It does **not** change the default/local execution path in either repository.

## Why this plan exists

You may want GPU throughput beyond a local workstation while keeping the current
pipeline contracts unchanged:

- `gpu-embedder` remains the focused embedding CLI.
- `llm_concept_mapping` still treats embeddings as versioned artifacts
  (`embed_model_version`) and keeps stage logic unchanged.

The goal is to add an **optional remote execution path** later, not replace the
current one.

## Scope and non-goals

### In scope

- AWS infrastructure options for batch embeddings
- Price/performance tradeoffs and benchmarking method
- Planned config + wiring changes (future work)
- Operational flow to produce and ingest embeddings safely

### Out of scope

- Immediate code implementation
- Replacing Snowflake/Fabric portability strategy
- Reworking stage scoring/retrieval logic

## Environment assumptions (academic AWS)

- Deployment is in an academic AWS environment with HIPAA compliance controls available.
- This project does not intend to process or share PHI, PII, or institutional IP.
- Even without regulated data, controls should align with HIPAA-ready guardrails to simplify
  review and prevent accidental data handling drift.

---

## Recommended AWS architecture (price/perf first)

## Option A (recommended): AWS Batch on ECS EC2 GPU (Spot-first)

Use this as the default AWS path for large, resumable batch runs.

**Why**
- Strong cost control with Spot capacity
- Good fit for long-running, restartable embedding jobs
- Minimal orchestration overhead vs full EKS

**Core components**
- Container image: ECR
- Job orchestration: AWS Batch (managed queue + compute environment)
- Compute: ECS EC2 GPU instances (Spot preferred, On-Demand fallback)
- Artifact storage: S3 (input shards, output embeddings, run manifests)
- Optional metadata DB: DynamoDB (job/run status) or just Batch + CloudWatch

**Good starter instance families**
- `g5` (NVIDIA A10G): generally strong baseline and broad availability
- `g6e` (NVIDIA L4): often better perf/$ for transformer inference workloads
- Avoid `p4d/p5` initially unless profiling proves they beat `g5/g6e` on $/concept

## Option B: SageMaker Processing jobs

Use if your team prefers fully managed ML job UX over cost minimization.

**Tradeoff**
- Easier managed experience
- Usually higher cost than Batch for equivalent throughput

## Option C: EKS + Karpenter

Use only when embedding becomes a continuously running platform capability.

**Tradeoff**
- Maximum flexibility
- Highest operational complexity

---

## Target execution model

Keep embedding output as an artifact, then ingest into warehouse tables.

1. Partition concept rows into shards (by vocabulary/domain/range).
2. Submit one AWS Batch job per shard.
3. Each job runs containerized embedder in FP32 mode (preserve invariant).
4. Write outputs to S3 as versioned parquet/ndjson plus manifest.
5. Validate completeness + vector dimension/model_version.
6. Bulk load into `concept_embeddings` (Snowflake now; Fabric later).

This preserves current stage contracts and keeps retrieval behavior unchanged.

## Read/write path recommendation

Use S3 as the default read/write layer for AWS execution:

- Read shard inputs from S3 (or generate shard manifests then upload once).
- Write per-shard embedding outputs to S3.
- Write run metadata (manifest, counts, checksums, completion markers) to S3.

This keeps workers stateless, improves retry behavior under Spot interruption,
and decouples GPU compute from downstream warehouse import.

When not to use S3 as primary I/O:

- Only for special cases needing low-latency shared POSIX semantics during a single job
  (consider EFS/FSx then). For this batch workload, S3 should remain the default.

## Source code + data movement model

- The runtime image can pull this repository from GitHub during build/CI and publish a pinned image to ECR.
- Batch jobs should run the pinned image tag/digest (not git clone on every job start).
- Input data (`CONCEPT.csv` or prepared shards) should be staged in S3 before submit.

Data files needed for embedding-only runs:

- Required: `CONCEPT.csv` (or derived shard files with the required concept fields).
- Optional: existing partial embedding state for resume logic.
- Not required for embedding itself: other Athena vocabulary files (`CONCEPT_ANCESTOR`, `CONCEPT_RELATIONSHIP`, etc.) unless a separate pipeline step uses them.

Note on CPT-4:

- If CPT-4 name population has already been done, only the resulting `CONCEPT.csv` content is needed for embedding.
- If CPT-4 enrichment has not been run yet, do that upstream first; do not couple Java/UMLS enrichment into AWS embedding workers.

Note on partial local store (`embeddings.lance` / `embeddings.duckdb`):

- Prefer artifact-first shard outputs + manifest/checkpoint over sharing a mutable store file across workers.
- If carrying forward a local partial state is necessary, upload it to an S3 checkpoint prefix as a snapshot input to a single merge/import step, not as a concurrently written shared store.

## Implementation lessons learned (sanitized)

This section captures deployment findings from iterative infrastructure bring-up.
It intentionally omits account identifiers, URLs, and organization-specific details.

### IAM and role model

- AWS Batch on EC2 requires distinct trust principals and cannot be collapsed to one role.
- Practical minimum is three IAM roles plus one instance profile:
  - Batch service role (trusted by `batch.amazonaws.com`)
  - EC2 instance role (trusted by `ec2.amazonaws.com`)
  - Job/container role (trusted by `ecs-tasks.amazonaws.com`)
  - Instance profile associated to the EC2 instance role
- Task execution role can be optional in this workload when container behavior and image access allow it.
- Spot-specific role can be avoided by disabling Spot during bootstrap.

### Permission pitfalls observed

- Control-plane provisioning requires `iam:PassRole` for the service/instance/job roles.
- Batch setup requires create/register permissions for compute environments, queues, and job definitions.
- Missing CloudWatch Logs permissions on the Batch service role can cause compute environment state `INVALID`.
- Terraform state operations may require IAM read/list permissions (for example role/policy introspection), even when resources already exist.
- Terraform execution roles also need S3 bucket configuration read permissions when storage resources are in scope.
  At minimum, include `s3:GetEncryptionConfiguration` and `s3:GetLifecycleConfiguration` on the target bucket ARN.

### Incident-specific lessons (Batch compute environment recovery)

- Role ARN path precision matters. A role created under a non-root path (for example `/customroles/`) has a different ARN than
  the root-path form, and Batch fails with `sts:AssumeRole` if the ARN is wrong.
- Existing trust policy correctness is not sufficient if the compute environment references a different ARN string.
- Batch compute environments can retain stale `INVALID` status reasons during IAM propagation and may require explicit delete/recreate.
- Pre-created role + instance profile names should be verified from live IAM before first apply in restricted environments.

### Minimal policy snippet for Terraform execution role

Use this as a baseline inline or managed policy for the Terraform execution role when applying this stack.
Replace `<bucket-name>` and scope further if your environment allows.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BatchControlPlane",
      "Effect": "Allow",
      "Action": [
        "batch:CreateComputeEnvironment",
        "batch:UpdateComputeEnvironment",
        "batch:DeleteComputeEnvironment",
        "batch:DescribeComputeEnvironments",
        "batch:CreateJobQueue",
        "batch:UpdateJobQueue",
        "batch:DeleteJobQueue",
        "batch:DescribeJobQueues",
        "batch:RegisterJobDefinition",
        "batch:DeregisterJobDefinition",
        "batch:DescribeJobDefinitions"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassBatchRoles",
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": [
        "arn:aws:iam::*:role/*embed-batch-service*",
        "arn:aws:iam::*:role/*embed-batch-instance*",
        "arn:aws:iam::*:role/*embed-batch-job*",
        "arn:aws:iam::*:role/*gpu-embed-batch-*"
      ]
    },
    {
      "Sid": "ReadBucketConfigForTerraform",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:GetBucketVersioning",
        "s3:GetPublicAccessBlock",
        "s3:GetEncryptionConfiguration",
        "s3:GetLifecycleConfiguration"
      ],
      "Resource": "arn:aws:s3:::<bucket-name>"
    }
  ]
}
```

### Terraform and state behavior

- Interrupted applies can leave partial resources in AWS and stale/tainted state entries.
- After interruption, reconcile state deliberately (refresh/import/remove stale entries) before retrying apply.
- Avoid repeated `Ctrl+C` during long create operations unless clearly hung.

### S3 provisioning behavior

- Bucket adjunct resources (versioning, encryption, public access block, lifecycle) may complete at different speeds.
- Lifecycle configuration can take noticeably longer than other bucket sub-resources.
- If bucket existence checks return access errors, verify whether this is a true non-existence case vs policy/read restriction.

### Practical rollout strategy

- Use pre-created IAM resources in restricted environments (`use_precreated_iam_roles=true`).
- Reference existing role/profile ARNs through environment tfvars instead of creating IAM from Terraform.
- Start with on-demand (Spot disabled) to minimize role and policy complexity.
- Promote to Spot only after baseline apply is stable.

### Operating model for restricted IAM organizations

This model is functional and should be treated as the default for orgs where IAM roles/permission sets are managed outside Terraform.

Responsibility split:

- Platform/IAM admin portal (manual):
  - Create and update permission sets.
  - Create and update IAM roles and instance profiles.
  - Provision permission sets to target AWS accounts.
  - Approve any SCP exceptions needed for `sso-admin` actions.
- Project Terraform operators (automated):
  - Manage Batch, queue, job definition, networking, and storage resources.
  - Reference pre-created IAM ARNs in environment tfvars.
  - Run plan/apply and day-2 infrastructure changes.

Manual steps expected through GUI/admin portal:

- Initial environment bootstrap (roles, trust policies, instance profile associations).
- Permission-set changes for execution roles (including S3 read/list actions used by Terraform refresh).
- Re-provisioning permission-set updates to target accounts.

What remains fully automated after bootstrap:

- Regular Terraform applies for non-IAM infrastructure.
- Batch environment updates, queue/job-definition revisions, and storage updates within granted permissions.

Pre-apply preflight checklist (recommended before every environment apply):

- Confirm `use_precreated_iam_roles=true` for restricted environments.
- Confirm tfvars role/profile ARNs exactly match live IAM ARNs (including any path segments).
- Confirm instance profile exists and contains the expected EC2 instance role.
- Confirm execution permission set includes:
  - Batch control-plane actions used by this stack.
  - `iam:PassRole` for referenced Batch roles.
  - S3 bucket config reads (`GetEncryptionConfiguration`, `GetLifecycleConfiguration`, and related bucket reads).
- Run `terraform plan` with the execution profile before opening change windows.

Admin request template for new environment onboarding:

- Request type: Terraform execution role entitlement update.
- Environment: `<env-name>`.
- Target account: `<account-id>`.
- Required pre-created resources:
  - Batch service role ARN.
  - Batch EC2 role ARN + instance profile ARN.
  - Batch job role ARN.
- Required execution permissions:
  - Batch create/update/delete/describe for compute env, queues, and job definitions.
  - `iam:PassRole` for the above Batch roles.
  - S3 bucket configuration read actions for the environment artifact bucket.
- Validation criteria:
  - `aws sts get-caller-identity` succeeds under execution profile.
  - `terraform plan` completes without IAM or S3 access-denied errors.

### Implementation pattern in this repository (workload-only mode)

For `infra/aws/envs/academic-dev` and `infra/aws/envs/academic-prod`, use these settings to keep Terraform focused on workload resources only:

- `use_precreated_iam_roles=true`
- `manage_storage_resources=false`

Behavior in this mode:

- `module.storage` is not managed by Terraform in the environment stack.
- Environment outputs (`bucket_name`, `bucket_arn`, `prefix_scope`) resolve from pre-created values.
- Batch resources (compute environment, queue, job definition, security group) remain Terraform-managed.

Migration checklist for an existing state:

1. Confirm pre-created bucket and prefix values are set in env tfvars.
2. Run a one-time state cleanup (admin-operated) to remove storage resources from this env state.
3. Re-run `terraform plan` using execution profile and confirm no S3 config read failures.

Example state cleanup commands (run from env directory, once):

```bash
terraform state rm module.storage.aws_s3_bucket.artifacts
terraform state rm module.storage.aws_s3_bucket_versioning.artifacts
terraform state rm module.storage.aws_s3_bucket_public_access_block.artifacts
terraform state rm module.storage.aws_s3_bucket_server_side_encryption_configuration.artifacts
terraform state rm module.storage.aws_s3_bucket_lifecycle_configuration.artifacts
```

### Operational commands

Operational AWS command flows are intentionally separated from this planning runbook.

Use:

- `docs/runbooks/aws_interaction_guide.md`

### Security and documentation hygiene

- Keep environment-specific identifiers in local tfvars and out of committed docs.
- Do not document account numbers, SSO URLs, or internal portal links in runbooks.
- Use generic role/policy naming conventions and repository-relative guidance.

---

## Price-to-performance strategy

Exact prices vary by region/date and Spot market, so optimize by measured
`$ per 1M concepts` instead of instance hourly price alone.

## Primary metric

$$
\text{Cost per 1M concepts} = \frac{\text{Total run cost USD}}{\text{Embedded concepts}/1{,}000{,}000}
$$

Track together with:
- concepts/sec
- GPU utilization
- failure/retry rate
- median and p95 shard duration

## Benchmark protocol (before committing)

Run the same 1M-concept sample on at least:
- `g5.xlarge` (A10G)
- `g6e.xlarge` (L4)
- one larger size in each family (for scaling curve)

For each run, hold constant:
- model + revision
- quantization (`fp32` for `gpu-embedder`; existing llm path may keep its current setting)
- batch size search space
- tokenization settings and text fields

Then choose by lowest stable `$ per 1M concepts` at acceptable wall-clock.

## Pre-run utilization test plan (before full dataset)

Run a short resource-utilization gate before full production execution.

Test shape:

- 3 shard sizes (small/medium/large; target ~5 min / ~10 min / ~20 min runtime each)
- 2 candidate instance types (`g5.xlarge`, `g6e.xlarge`)
- 2–3 batch sizes per instance

Capture per run:

- GPU utilization and memory utilization
- CPU and RAM usage
- input rows/sec and embedded rows/sec
- retry/interruption behavior and median restart penalty
- cost estimate from runtime * effective instance price

Promotion gate for full run:

- no OOM on selected batch size
- p95 shard time within SLA target
- stable retries (no sustained retry storm)
- selected profile wins on both throughput and `$ per 1M concepts`

## Cost controls

- Spot-first with capped retries and checkpointed shards
- Small shard size (5–20 minutes) to reduce Spot interruption waste
- EBS `gp3` right-sized throughput, avoid overprovisioning
- Pre-pulled image + warm HuggingFace cache layer in AMI/image where practical
- Strict CloudWatch alarms on queue backlog and retry storm conditions

---

## Suggested S3 directory structure

Use one bucket (or one dedicated prefix) per environment with immutable run IDs.

Example:

- `s3://<bucket>/gpu-embed/<env>/inputs/raw/CONCEPT/<date>/CONCEPT.csv`
- `s3://<bucket>/gpu-embed/<env>/inputs/shards/<run_id>/manifest.json`
- `s3://<bucket>/gpu-embed/<env>/inputs/shards/<run_id>/part-00000.parquet`
- `s3://<bucket>/gpu-embed/<env>/outputs/<run_id>/shards/part-00000.ndjson.gz`
- `s3://<bucket>/gpu-embed/<env>/outputs/<run_id>/manifests/completion.json`
- `s3://<bucket>/gpu-embed/<env>/outputs/<run_id>/metrics/summary.json`
- `s3://<bucket>/gpu-embed/<env>/checkpoints/<run_id>/embeddings.lance` (optional snapshot only)
- `s3://<bucket>/gpu-embed/<env>/logs/<run_id>/worker-<job_id>.log`

Naming guidance:

- `run_id` should include timestamp + short git SHA + model-version prefix.
- Keep shard outputs immutable; retries write new attempt files and update manifest pointers.
- Treat `outputs/<run_id>/manifests/completion.json` as the source of truth for import eligibility.

Import handoff:

- Downstream import should consume only completion-manifest-declared shard files.
- Never import directly from in-progress shard prefixes.

---

## Artifact schema + import modes

Default recommendation:

- Primary output format: Parquet (for scale and warehouse load efficiency).
- Optional debug output: NDJSON (for easy inspection/troubleshooting).

Minimum artifact columns for `llm_concept_mapping` compatibility:

- `concept_id` (int)
- `vocabulary_id` (string)
- `domain_id` (string)
- `concept_name` (string)
- `standard_concept` (nullable string)
- `embedding` (array<float>, length 768)
- `embedded_at` (timestamp)
- `embed_model_version` (string; canonical version id)

Recommended optional columns:

- `concept_class_id`
- `invalid_reason`
- `embed_text` (the exact concatenated text used for embedding)
- `model_version_legacy` (existing gpu-embedder SHA-256 weights hash)
- `run_id`, `shard_id`, `attempt`

Import modes to support:

- Direct stage load: `S3 -> Snowflake external stage -> COPY/MERGE` (default for production)
- Local relay: `S3 -> local download -> Snowflake load` (fallback/manual mode)

In both modes, import must be idempotent on `(concept_id, embed_model_version)`.

---

## Model version naming alignment (cross-repo)

Current state:

- `gpu_embedding` stores a raw SHA-256 digest of weight files as `model_version`.
- `llm_concept_mapping` expects a namespaced version string format:
  `sapbert-<backend>-<quantization>-<digest10>`.

Recommendation:

- Define `embed_model_version` as the canonical cross-repo identifier using the
  `llm_concept_mapping` naming convention.
- Keep the existing gpu hash as `model_version_legacy` during transition for lineage.

Canonical digest input (pinned attributes, sorted JSON):

- model_name
- revision
- backend
- quantization
- pooling
- dimension
- normalize

Canonical ID format:

- `sapbert-<backend>-<quantization>-<sha256(pinned_attributes_json)[:10]>`

For current GPU embedder defaults, likely namespace example:

- `sapbert-torch-fp32-<digest10>`

Transition strategy:

1. Write both `embed_model_version` (canonical) and `model_version_legacy` in new AWS artifacts.
2. During import to Snowflake, populate `embed_model_version` from artifact directly.
3. Keep legacy value available for audit/backfill verification.
4. After validation period, treat canonical `embed_model_version` as the only operational key.

---

## Ingesting existing `embeddings.duckdb`

Yes, existing local DuckDB files can be migrated.

Recommended one-time backfill path:

1. Snapshot the current DuckDB file.
2. Export `concept_embeddings` to Parquet shards (include legacy and canonical model version columns).
3. Upload shards to S3 under a dedicated backfill run prefix.
4. Generate a completion manifest with row counts/checksums.
5. Load to Snowflake with MERGE on `(concept_id, embed_model_version)`.
6. Run post-load validation queries (counts by vocabulary/domain/model version).

Important migration decision:

- If historical DuckDB rows only have legacy hash versions, either:
  - map each legacy hash to one canonical `embed_model_version` (preferred when provenance is clear), or
  - preserve them as legacy-only history and re-embed into canonical namespace.

Validation checks after backfill:

- row counts match source export totals
- all vectors have dimension 768
- no duplicate `(concept_id, embed_model_version)`
- retrieval queries in `llm_concept_mapping` can target the expected canonical version

---

## Planned repository changes (future, not now)

## 1) gpu_embedding changes (optional AWS mode)

Keep local mode as default. Add an optional submit/remote mode later.

### Config additions (planned)

- `GPU_EMBED_EXECUTION_MODE=local|aws_batch` (default `local`)
- `GPU_EMBED_AWS_REGION`
- `GPU_EMBED_AWS_BATCH_QUEUE`
- `GPU_EMBED_AWS_BATCH_JOB_DEFINITION`
- `GPU_EMBED_AWS_S3_INPUT_PREFIX`
- `GPU_EMBED_AWS_S3_OUTPUT_PREFIX`
- `GPU_EMBED_AWS_MAX_ARRAY_SIZE`
- `GPU_EMBED_AWS_SPOT_PREFERRED=true|false`

### CLI/wiring additions (planned)

- Keep `gpu-embed embed` unchanged for local behavior.
- Add an AWS-oriented subcommand (example: `gpu-embed aws-submit`) that:
  - resolves filters and shard manifest
  - uploads shard inputs to S3
  - submits Batch array jobs
  - monitors completion and writes run summary
- Add `gpu-embed aws-collect` to validate outputs and optionally merge artifacts.

### Storage/output additions (planned)

Current local DuckDB output stays intact.
Add optional artifact writer:
- parquet/ndjson shard output with fields needed for warehouse upsert:
  - `concept_id`, `concept_name`, metadata columns
  - `embedding` (768 float array)
  - `embed_model_version`
  - `embedded_at`

## 2) llm_concept_mapping changes (integration wiring)

Do not change stage logic. Add ingestion path for external embeddings.

### Config additions (planned)

- `EMBEDDING_SOURCE=local|snowflake_udf|aws_artifact` (default remains current)
- `EMBEDDING_ARTIFACT_URI` (S3 prefix or staged warehouse location)
- `EMBEDDING_ARTIFACT_FORMAT=parquet|ndjson`
- `EMBEDDING_IMPORT_MODE=append|merge`

### Pipeline/CLI additions (planned)

- Add command to load externally generated embeddings into
  `concept_embeddings` with strict validation:
  - dimension must equal configured embedding dimension
  - model version must match expected pinned config (or explicit override)
  - idempotent merge on `(concept_id, embed_model_version)`
- Reuse existing retrieval and review code paths unchanged after load.

### SQL/warehouse additions (planned)

- Add bulk-load SQL templates in `src/concept_mapper/sql/` for importing
  staged artifact files into `concept_embeddings`.
- Keep warehouse-specific SQL isolated per current portability approach.

---

## Security and compliance considerations

- Keep any credential material in AWS Secrets Manager/SSM, not in job args.
- Restrict S3 prefixes and KMS keys per environment.
- Use VPC endpoints for S3/ECR/CloudWatch where required.
- Log model version, shard id, and row counts; never log sensitive keys.

## HIPAA-oriented baseline controls (recommended)

Apply these controls even though PHI/PII is out of scope:

- Private S3 buckets with public access blocked and ACLs disabled.
- SSE-KMS encryption for all S3 objects and logs.
- Least-privilege IAM for job roles (prefix-scoped read/write only).
- CloudTrail management events plus S3 data events for object-level audit.
- Lifecycle rules: expire intermediates, retain manifests/audit artifacts longer.
- Prefer private networking paths (S3/ECR/CloudWatch VPC endpoints).

---

## Infrastructure templates (recommended)

Yes: create infrastructure templates in a dedicated directory so the AWS path is reproducible,
reviewable, and environment-specific.

Suggested location:

- `infra/aws/`
- Outline scaffold now exists in this repo under `infra/aws/` with env/module README placeholders.

Suggested contents:

- `infra/aws/README.md` (how to deploy, required vars, environments)
- `infra/aws/envs/dev/` and `infra/aws/envs/prod/` (or `academic-dev` / `academic-prod`)
- `infra/aws/modules/batch_gpu/` (Batch queue, compute env, job definition)
- `infra/aws/modules/storage/` (S3 buckets, lifecycle, bucket policy)
- `infra/aws/modules/security/` (KMS, IAM roles/policies, CloudTrail wiring)

Tooling options:

- Terraform is the most portable/common choice for this stack.
- AWS CDK is also viable if your team prefers typed IaC and has strong TypeScript/Python CDK conventions.

MVP guidance:

- Start with one environment template that provisions S3 + Batch + IAM + KMS only.
- Add optional observability and advanced networking modules after first successful benchmark cycle.

---

## Rollout plan

## Phase 0: Baseline
- Record local throughput and cost proxy (time + machine class).

## Phase 1: Prototype (single vocabulary)
- Run one vocabulary shard set on AWS Batch Spot.
- Validate output schema, model_version, and idempotent merge into warehouse.

## Phase 2: Controlled production
- Route only embedding builds to AWS path.
- Keep pipeline retrieval/rerank execution where it already runs.

## Phase 3: Expand
- Add source-query embedding runs (`embed-sources`) when stable.
- Introduce auto-scaling and richer dashboards only if needed.

---

## Decision checklist

Adopt AWS embedding path when all are true:

- Measured `$ per 1M concepts` improves vs local baseline
- End-to-end latency meets wave SLAs
- Retry/interruption behavior is stable under Spot churn
- Import validation catches all schema/version mismatches
- No changes required to Stage 3/4/5 business logic

---

## Suggested first experiment

1. Build one container image for current embed runtime.
2. Benchmark 1M concepts on `g5.xlarge` and `g6e.xlarge` in one AWS region.
3. Compare concepts/sec and `$ per 1M concepts`.
4. Pick winner and run one full-vocabulary dry run.
5. If successful, implement only the minimal submit/import wiring above.
