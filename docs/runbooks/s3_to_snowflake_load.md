# S3 to Snowflake Load Runbook

This runbook covers the operational flow for moving parquet-backed embeddings
from the local store to S3 and loading them into Snowflake.

## Scope

- Sync parquet shards from the local embeddings store
- Authenticate to AWS CLI
- Copy or sync parquet shards to S3
- Load from S3 into Snowflake with `COPY INTO`
- Perform idempotent upserts with `MERGE`

## Prerequisites

- Embeddings store available (default: `embeddings/`)
- AWS CLI installed and configured (default profile or named profile)
- S3 bucket + path chosen for parquet dataset
- Snowflake `STORAGE INTEGRATION` with read access to the bucket path

## 1) Validate local embeddings parquet store

By default, embeddings are stored as:

`embeddings/model_version=<sha256>/vocabulary_id=<value>/part-*.parquet`

Model hash provenance is stored alongside the dataset at:

`embeddings/_meta/model_registry/part-*.parquet`

```bash
uv run gpu-embed migrate-store --db embeddings
uv run python -c "from pathlib import Path; print(len(list(Path('embeddings').glob('model_version=*/*.parquet'))))"
```

If you are migrating from a legacy `.duckdb` file and have not migrated yet,
run once to trigger automatic migration:

```bash
uv run gpu-embed migrate-store --db embeddings.duckdb
```

This creates `embeddings/model_version=.../*.parquet` and no manual pre-export is required.

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
  --recursive embeddings \
  s3://<your-bucket>/concept_embeddings/
```

Use `sync` for repeat runs:

```bash
AWS_PAGER="" aws s3 sync \
  embeddings \
  s3://<your-bucket>/concept_embeddings/
```

Expected layout:

```text
s3://<your-bucket>/concept_embeddings/
  model_version=<sha256>/vocabulary_id=SNOMED/part-00000.parquet
  model_version=<sha256>/vocabulary_id=LOINC/part-00000.parquet
  model_version=<sha256>/vocabulary_id=_null/part-00000.parquet
```

### Optional: curated export flow

Keep `gpu-embed export` for curated extracts (e.g., specific model version,
vocabulary subsets, or custom shard sizing):

```bash
uv run gpu-embed export exports/parquet \
  --db embeddings \
  --model-version <model_version_prefix> \
  --vocabulary-id SNOMED,LOINC \
  --shard-rows 50000

AWS_PAGER="" aws s3 sync \
  exports/parquet \
  s3://<your-bucket>/concept_embeddings/
```

## 4) Load from S3 into Snowflake (`COPY INTO`)

```sql
-- One-time file format
CREATE OR REPLACE FILE FORMAT omop_parquet_ff
  TYPE = PARQUET;

-- One-time external stage
CREATE OR REPLACE STAGE omop_embed_stage
  URL = 's3://<your-bucket>/concept_embeddings/'
  STORAGE_INTEGRATION = <your_storage_integration>
  FILE_FORMAT = omop_parquet_ff;

-- Target table (example)
CREATE TABLE IF NOT EXISTS concept_embeddings (
  concept_id BIGINT,
  concept_name STRING,
  domain_id STRING,
  vocabulary_id STRING,
  concept_class_id STRING,
  standard_concept STRING,
  concept_code STRING,
  invalid_reason STRING,
  embedding ARRAY,
  embed_text STRING,
  model_version STRING,
  embedded_at TIMESTAMP_NTZ
);

-- Bulk load all vocabulary directories
COPY INTO concept_embeddings
FROM @omop_embed_stage
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = ABORT_STATEMENT;
```

Load a single model-version + vocabulary partition only:

```sql
COPY INTO concept_embeddings
FROM @omop_embed_stage/model_version=<sha256>/vocabulary_id=SNOMED/
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = ABORT_STATEMENT;
```

## 5) Idempotent load with staging + `MERGE` (recommended)

If you rerun loads often, use a staging table and upsert on
`(concept_id, model_version)`.

```sql
CREATE TABLE IF NOT EXISTS concept_embeddings_stage LIKE concept_embeddings;

COPY INTO concept_embeddings_stage
FROM @omop_embed_stage
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = ABORT_STATEMENT;

MERGE INTO concept_embeddings t
USING concept_embeddings_stage s
  ON t.concept_id = s.concept_id
 AND t.model_version = s.model_version
WHEN MATCHED THEN UPDATE SET
  concept_name = s.concept_name,
  domain_id = s.domain_id,
  vocabulary_id = s.vocabulary_id,
  concept_class_id = s.concept_class_id,
  standard_concept = s.standard_concept,
  concept_code = s.concept_code,
  invalid_reason = s.invalid_reason,
  embedding = s.embedding,
  embed_text = s.embed_text,
  embedded_at = s.embedded_at
WHEN NOT MATCHED THEN INSERT (
  concept_id,
  concept_name,
  domain_id,
  vocabulary_id,
  concept_class_id,
  standard_concept,
  concept_code,
  invalid_reason,
  embedding,
  embed_text,
  model_version,
  embedded_at
) VALUES (
  s.concept_id,
  s.concept_name,
  s.domain_id,
  s.vocabulary_id,
  s.concept_class_id,
  s.standard_concept,
  s.concept_code,
  s.invalid_reason,
  s.embedding,
  s.embed_text,
  s.model_version,
  s.embedded_at
);

TRUNCATE TABLE concept_embeddings_stage;
```

## Troubleshooting

- `ExpiredToken` / auth failures: rerun `aws sso login` (or the same named-profile login command)
- `AccessDenied` on S3: verify profile permissions and bucket policy
- Snowflake stage read failures: validate `STORAGE INTEGRATION` trust and allowed locations
- Load errors on columns: confirm parquet field names and use `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE`
