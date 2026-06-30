# S3 to Snowflake Load Runbook

This runbook covers the operational flow for exporting embeddings to parquet,
copying to S3, and loading into Snowflake.

## Scope

- Sync parquet shards from the local embeddings store
- Authenticate to AWS CLI
- Copy or sync parquet shards to S3
- Load from S3 into Snowflake with `COPY INTO`
- Perform idempotent upserts with `MERGE`

## Prerequisites

- Embeddings store available (default local source: `embeddings.lance`)
- AWS CLI installed and configured (default profile or named profile)
- S3 bucket + path chosen for parquet dataset
- Snowflake `STORAGE INTEGRATION` with read access to the bucket path

## 1) Export parquet handoff dataset

Use `export` as the standard handoff path from the local store:

```bash
uv run gpu-embed export exports/parquet --db embeddings.lance
```

Export output is written as Hive-partitioned parquet:

`exports/parquet/model_version=<sha256>/vocabulary_id=<value>/part-*.parquet`

This matches the parquet store / `migrate-store` layout, so the S3 tree and
Snowflake external stage use one uniform layout regardless of which tool wrote
it. `export` is the curated path (a single model version, optionally filtered by
vocabulary/namespace); `migrate-store` mirrors the **full** store (every model
version and all rows) for platform portability:

```bash
uv run gpu-embed migrate-store --db embeddings.lance
```

which creates the same partition layout under `embeddings/`:

`embeddings/model_version=<sha256>/vocabulary_id=<value>/part-*.parquet`

Model hash provenance for the mirrored store layout is stored at:

`embeddings/_meta/model_registry/part-*.parquet`

```bash
uv run python -c "from pathlib import Path; print(len(list(Path('exports/parquet').glob('model_version=*/vocabulary_id=*/*.parquet'))))"
```

If you are using `migrate-store` on a large dataset, throughput may slow as
bigger partitions are processed. Treat this as expected if migration progress
logs keep advancing (`partitions`, `rows`, `files`, `eta_minutes`).

## 2) Authenticate to AWS CLI

Credential precedence (highest to lowest): explicit command flags,
environment variables, then default profile/config files.

- `--profile <name>` on a command
- `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- Default profile in `~/.aws/config` and `~/.aws/credentials`

```bash
# Confirm CLI
aws --version

# One-time default profile setup (access key flow)
aws configure

# Or one-time default SSO profile setup
aws configure sso

# Per-session SSO login (if using SSO)
aws sso login

# Optional named profile flow:
# aws configure --profile <aws-profile>
# aws configure sso --profile <aws-profile>
# aws sso login --profile <aws-profile>
# export AWS_PROFILE=<aws-profile>

# Verify identity and bucket access
AWS_PAGER="" aws sts get-caller-identity
AWS_PAGER="" aws s3 ls s3://<your-bucket>/
```

## 3) Copy parquet tree to S3

Use `cp --recursive` for a full push:

```bash
AWS_PAGER="" aws s3 cp \
  --recursive exports/parquet \
  s3://<your-bucket>/concept_embeddings/
```

Use `sync` for repeat runs:

```bash
AWS_PAGER="" aws s3 sync \
  exports/parquet \
  s3://<your-bucket>/concept_embeddings/
```

Expected layout (export flow):

```text
s3://<your-bucket>/concept_embeddings/
  model_version=<sha256>/
    vocabulary_id=SNOMED/part-00000.parquet
    vocabulary_id=LOINC/part-00000.parquet
    vocabulary_id=_null/part-00000.parquet
```

### Optional: curated export flow

Keep `gpu-embed export` for curated extracts (e.g., specific model version,
vocabulary subsets, or custom shard sizing):

```bash
uv run gpu-embed export exports/parquet \
  --db embeddings.lance \
  --model-version <model_version_prefix> \
  --vocabulary-id SNOMED,LOINC \
  --shard-rows 50000

AWS_PAGER="" aws s3 sync \
  exports/parquet \
  s3://<your-bucket>/concept_embeddings/
```

## 4) Load from S3 into a staging table (`COPY INTO`)

> The export parquet mirrors the GPU store, not concept-mapper's contract tables:
> `embedding` is a fixed-size float list, the version column is `model_version`
> (the **weights-file SHA-256** — the store's provenance identity), and there are
> extra columns (`namespace`, `concept_class_id`, `concept_code`, `invalid_reason`,
> `embed_text`, `source_id`, `mapping_wave`). Concept-mapper's contract tables use
> a **different** shape — `embedding VECTOR(FLOAT, 768)` and the version column
> `embed_model_version` — so we land the parquet in a staging table first (§4),
> then upsert into the contract tables with the right types and the shared
> retrieval version (§5). See ALIGNMENT.md §4.
>
> `source_id` / `mapping_wave` are NULL for Athena **target** concepts and
> populated for **source** concepts, so one staging load fans out to both contract
> tables: targets → `concept_embeddings`, sources → `source_concepts`.

```sql
-- One-time file format + external stage
CREATE OR REPLACE FILE FORMAT omop_parquet_ff TYPE = PARQUET;
CREATE OR REPLACE STAGE omop_embed_stage
  URL = 's3://<your-bucket>/concept_embeddings/'
  STORAGE_INTEGRATION = <your_storage_integration>
  FILE_FORMAT = omop_parquet_ff;

-- Staging table mirrors the parquet exactly (embedding as ARRAY, weights-hash
-- model_version). This is NOT a contract table — it is the load buffer for §5.
CREATE TABLE IF NOT EXISTS omop_mapping.concept_embeddings_stage (
  namespace STRING, concept_id BIGINT, concept_name STRING, domain_id STRING,
  vocabulary_id STRING, concept_class_id STRING, standard_concept STRING,
  concept_code STRING, invalid_reason STRING, embedding ARRAY, embed_text STRING,
  model_version STRING, embedded_at TIMESTAMP_NTZ, source_id STRING, mapping_wave STRING
);

TRUNCATE TABLE omop_mapping.concept_embeddings_stage;
COPY INTO omop_mapping.concept_embeddings_stage
FROM @omop_embed_stage
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = ABORT_STATEMENT;
```

Load a single model-version + vocabulary partition only by pointing `COPY` at a
sub-path, e.g. `@omop_embed_stage/model_version=<sha256>/vocabulary_id=SNOMED/`.

## 5) Upsert into concept-mapper's contract tables (`MERGE`)

The contract tables store `embedding` as `VECTOR(FLOAT, 768)` and key retrieval on
**`embed_model_version`** — the config-derived, engine-independent stamp that
concept-mapper's Stage 3 filters on (ALIGNMENT.md §4.2). It is **not** the
weights-hash `model_version` in the staging table. Get the exact value from the
GPU repo (it equals concept-mapper's `embed_model_version_from_settings()` for the
same pinned model):

```bash
uv run gpu-embed retrieval-version      # e.g. sapbert-cls-fp32-1a2b3c4d5e
```

```sql
-- Paste the value printed by `gpu-embed retrieval-version`:
SET embed_model_version = 'sapbert-cls-fp32-1a2b3c4d5e';

-- 5a) Athena TARGET concepts -> concept_embeddings (source_id IS NULL).
--     ARRAY -> VECTOR cast; idempotent on (concept_id, embed_model_version).
MERGE INTO omop_mapping.concept_embeddings t
USING (
  SELECT concept_id, vocabulary_id, domain_id, concept_name, standard_concept,
         embedding::VECTOR(FLOAT, 768) AS embedding, embedded_at
  FROM omop_mapping.concept_embeddings_stage
  WHERE source_id IS NULL
) s
  ON  t.concept_id = s.concept_id
  AND t.embed_model_version = $embed_model_version
WHEN MATCHED THEN UPDATE SET
  vocabulary_id = s.vocabulary_id, domain_id = s.domain_id,
  concept_name = s.concept_name, standard_concept = s.standard_concept,
  embedding = s.embedding, embedded_at = s.embedded_at
WHEN NOT MATCHED THEN INSERT (
  concept_id, vocabulary_id, domain_id, concept_name, standard_concept,
  embedding, embedded_at, embed_model_version
) VALUES (
  s.concept_id, s.vocabulary_id, s.domain_id, s.concept_name, s.standard_concept,
  s.embedding, s.embedded_at, $embed_model_version
);

-- 5b) SOURCE concepts -> source_concepts.query_embedding (source_id IS NOT NULL).
--     Rejoin on the natural (mapping_wave, source_id) key. Rows must already
--     exist from Stage 0; this only fills the query vector. Same FP32+CLS space.
MERGE INTO omop_mapping.source_concepts t
USING (
  SELECT source_id, mapping_wave,
         embedding::VECTOR(FLOAT, 768) AS query_embedding, embedded_at
  FROM omop_mapping.concept_embeddings_stage
  WHERE source_id IS NOT NULL
) s
  ON  t.mapping_wave = s.mapping_wave
  AND t.source_id = s.source_id
WHEN MATCHED THEN UPDATE SET
  query_embedding = s.query_embedding,
  embed_model_version = $embed_model_version,
  embedded_at = s.embedded_at;

TRUNCATE TABLE omop_mapping.concept_embeddings_stage;
```

> **Why the version is overridden, not copied.** The staging `model_version` is
> the weights-file SHA-256 (store provenance). Stage 3 compares query vs document
> vectors only when their `embed_model_version` matches, so both 5a and 5b stamp
> the single §4.2 value. Keep `gpu-embed retrieval-version` ≡ concept-mapper's
> `embed_model_version_from_settings()`; if they drift, semantic retrieval
> silently returns nothing.

### Optional: standalone portability table

For platform-portability dumps (not concept-mapper retrieval), you can still load
the parquet verbatim into a standalone table that mirrors the store
(`embedding ARRAY`, `model_version`), keyed on `(namespace, concept_id,
model_version)`. That table is independent of the contract tables Stage 3 reads.

## Troubleshooting

- `ExpiredToken` / auth failures: rerun `aws sso login` (or the same named-profile login command)
- `AccessDenied` on S3: verify profile permissions and bucket policy
- Snowflake stage read failures: validate `STORAGE INTEGRATION` trust and allowed locations
- Load errors on columns: confirm parquet field names and use `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE`
