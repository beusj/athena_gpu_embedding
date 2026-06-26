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

## Cost controls

- Spot-first with capped retries and checkpointed shards
- Small shard size (5–20 minutes) to reduce Spot interruption waste
- EBS `gp3` right-sized throughput, avoid overprovisioning
- Pre-pulled image + warm HuggingFace cache layer in AMI/image where practical
- Strict CloudWatch alarms on queue backlog and retry storm conditions

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
