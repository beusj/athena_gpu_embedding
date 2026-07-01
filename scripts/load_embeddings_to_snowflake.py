"""
Load S3 embedding parquet files into Snowflake via a temporary internal stage.

Usage (run from llm_concept_mapping/ which has snowflake-connector-python):
  cd /data/data_models/llm_concept_mapping
  uv run python ../athena_gpu_embedding/scripts/load_embeddings_to_snowflake.py
  uv run python ../athena_gpu_embedding/scripts/load_embeddings_to_snowflake.py --sync-only
  uv run python ../athena_gpu_embedding/scripts/load_embeddings_to_snowflake.py --load-only

Steps:
  1. aws s3 sync  → local exports/parquet/
  2. Create Snowflake internal named stage
  3. PUT all parquet files to the stage (parallel, auto-compressed)
  4. COPY INTO concept_embeddings_stage (staging buffer)
  5. MERGE targets  → omop_mapping.concept_embeddings
  6. MERGE sources  → omop_mapping.source_concepts  (query_embedding)
  7. Truncate + drop staging table and internal stage

Configuration (reads llm_concept_mapping/.env via dotenv):
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH,
  SNOWFLAKE_DATABASE, MAPPING_SCHEMA, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE

Requires:
  uv sync --extra cpu   (snowflake-snowpark-python is a dep of concept_mapper)
  AWS credentials in environment or ~/.aws (default profile)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_REPO = REPO_ROOT.parent / "llm_concept_mapping"
LOCAL_PARQUET_DIR = REPO_ROOT / "exports" / "parquet"

# Defaults — overridden by env vars (GPU_EMBED_S3_BUCKET, GPU_EMBED_S3_PREFIX,
# GPU_EMBED_RETRIEVAL_VERSION, GPU_EMBED_STAGE_NAME, GPU_EMBED_STAGING_TABLE).
# See athena_gpu_embedding/.env.example for documentation.
_DEFAULT_S3_BUCKET = "gpu-embedder-artifacts"
_DEFAULT_S3_PREFIX = "gpu-embed/dev/concept_embeddings/"
_DEFAULT_EMBED_MODEL_VERSION = "sapbert-cls-fp32-d34a93eed7"
_DEFAULT_STAGE_NAME = "omop_embed_load_stage"
_DEFAULT_STAGING_TABLE = "concept_embeddings_stage"

# ---------------------------------------------------------------------------
# Load .env from llm_concept_mapping (same Snowflake creds)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env files from both repos; existing env vars always win."""
    env_paths = [REPO_ROOT / ".env", LLM_REPO / ".env"]
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                print(f"[info] loaded env from {env_path}")
    except ImportError:
        # Manual fallback — no python-dotenv available
        def _parse(env_path: Path) -> None:
            if not env_path.exists():
                return
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            print(f"[info] loaded env from {env_path} (manual parser)")
        for env_path in env_paths:
            _parse(env_path)


# ---------------------------------------------------------------------------
# Step 1: S3 sync
# ---------------------------------------------------------------------------

def _cfg(key: str, default: str) -> str:
    """Read a config value from env, falling back to the hardcoded default."""
    return os.environ.get(key, default)


def sync_from_s3() -> None:
    s3_uri = f"s3://{_cfg('GPU_EMBED_S3_BUCKET', _DEFAULT_S3_BUCKET)}/{_cfg('GPU_EMBED_S3_PREFIX', _DEFAULT_S3_PREFIX)}"
    LOCAL_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[step 1] Syncing {s3_uri} -> {LOCAL_PARQUET_DIR}")
    subprocess.run(
        [
            "aws", "s3", "sync",
            s3_uri, str(LOCAL_PARQUET_DIR),
            "--exclude", "*",
            "--include", "*.parquet",
        ],
        check=True,
    )
    files = list(LOCAL_PARQUET_DIR.rglob("*.parquet"))
    print(f"[step 1] Sync complete: {len(files)} parquet files locally")


# ---------------------------------------------------------------------------
# Snowflake connection (reuse concept-mapper config)
# ---------------------------------------------------------------------------

def _build_conn_params() -> dict:
    account  = os.environ["SNOWFLAKE_ACCOUNT"]
    user     = os.environ["SNOWFLAKE_USER"]
    database = os.environ.get("SNOWFLAKE_DATABASE", "CHIC_REG_DEV")
    schema   = os.environ.get("MAPPING_SCHEMA", "omop_mapping")
    warehouse= os.environ.get("SNOWFLAKE_WAREHOUSE", "CHIC_WH_STD")
    role     = os.environ.get("SNOWFLAKE_ROLE", "SF_HIC_DDU")
    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")

    params: dict = dict(
        account=account, user=user, database=database,
        schema=schema, warehouse=warehouse, role=role,
    )
    if key_path:
        from cryptography.hazmat.primitives.serialization import (  # type: ignore[import]
            load_pem_private_key, Encoding, PrivateFormat, NoEncryption
        )
        key_bytes = Path(key_path).read_bytes()
        passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "").encode() or None
        private_key = load_pem_private_key(key_bytes, password=passphrase)
        params["private_key"] = private_key.private_bytes(
            Encoding.DER, PrivateFormat.PKCS8, NoEncryption()
        )
    else:
        raise RuntimeError("SNOWFLAKE_PRIVATE_KEY_PATH must be set")
    return params


def _get_connection():
    """Return a low-level snowflake.connector connection (not Snowpark)."""
    import snowflake.connector  # type: ignore[import]
    params = _build_conn_params()
    print(f"[sf] connecting: account={params['account']} user={params['user']} "
          f"db={params['database']} schema={params['schema']}")
    conn = snowflake.connector.connect(**params)
    print("[sf] connected")
    return conn


# ---------------------------------------------------------------------------
# Step 2-3: Create stage + PUT files
# ---------------------------------------------------------------------------

def create_stage(cur) -> None:
    stage = _cfg('GPU_EMBED_STAGE_NAME', _DEFAULT_STAGE_NAME)
    print(f"\n[step 2] Creating internal stage {stage}")
    cur.execute(f"CREATE STAGE IF NOT EXISTS {stage} FILE_FORMAT = (TYPE = PARQUET)")
    print(f"[step 2] Stage {stage} ready")


def put_files(conn, workers: int = 4, overwrite: bool = True) -> None:
    stage = _cfg('GPU_EMBED_STAGE_NAME', _DEFAULT_STAGE_NAME)
    files = sorted(LOCAL_PARQUET_DIR.rglob("*.parquet"))
    overwrite_flag = "TRUE" if overwrite else "FALSE"
    print(f"\n[step 3] Uploading {len(files)} parquet files to @{stage} "
          f"({workers} workers, AUTO_COMPRESS=FALSE, OVERWRITE={overwrite_flag})")

    import snowflake.connector  # type: ignore[import]
    params = _build_conn_params()
    errors: list[str] = []

    def _put_one(f: Path) -> tuple[str, str]:
        # Each thread uses its own connection — snowflake.connector is not
        # thread-safe when sharing a connection across threads.
        thread_conn = snowflake.connector.connect(**params)
        try:
            cur = thread_conn.cursor()
            # AUTO_COMPRESS=FALSE: parquet files are already compressed
            # internally (SNAPPY/ZSTD). Adding GZIP on top causes COPY INTO to
            # see a .parquet.gz file while the file format says TYPE=PARQUET,
            # which silently skips the file.
            result = cur.execute(
                f"PUT 'file://{f}' @{stage} "
                f"AUTO_COMPRESS=FALSE OVERWRITE={overwrite_flag}",
                timeout=600,
            ).fetchall()
            status = result[0][6] if result else "UNKNOWN"  # column: status
            cur.close()
            return f.name, status
        finally:
            thread_conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_put_one, f): f for f in files}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            name, status = future.result()
            done += 1
            if status not in ("UPLOADED", "SKIPPED"):
                errors.append(f"{name}: {status}")
            if done % 20 == 0 or done == len(files):
                print(f"[step 3] {done}/{len(files)} processed (last: {name} → {status})")

    if errors:
        print(f"[step 3] WARNING: {len(errors)} file(s) had unexpected status:")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"[step 3] All {len(files)} files staged successfully")

    # Verify stage file count matches local file count
    verify_conn = snowflake.connector.connect(**params)
    try:
        vcur = verify_conn.cursor()
        vcur.execute(f"LIST @{stage}")
        staged = vcur.fetchall()
        vcur.close()
    finally:
        verify_conn.close()
    print(f"[step 3] Verification: {len(files)} local files, {len(staged)} files in stage")
    if len(staged) < len(files):
        print(f"[step 3] WARNING: {len(files) - len(staged)} files missing from stage — "
              f"re-run with --put-only to retry")


# ---------------------------------------------------------------------------
# Step 4: Create staging table + COPY INTO
# ---------------------------------------------------------------------------

STAGING_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {staging_table} (
    namespace        STRING,
    concept_id       BIGINT,
    concept_name     STRING,
    domain_id        STRING,
    vocabulary_id    STRING,
    concept_class_id STRING,
    standard_concept STRING,
    concept_code     STRING,
    invalid_reason   STRING,
    embedding        ARRAY,
    embed_text       STRING,
    model_version    STRING,
    embedded_at      TIMESTAMP_NTZ,
    source_id        STRING,
    mapping_wave     STRING
)
"""

def copy_into_staging(cur) -> None:
    stage = _cfg('GPU_EMBED_STAGE_NAME', _DEFAULT_STAGE_NAME)
    staging_table = _cfg('GPU_EMBED_STAGING_TABLE', _DEFAULT_STAGING_TABLE)
    print(f"\n[step 4] Creating/truncating staging table {staging_table}")
    cur.execute(STAGING_DDL_TEMPLATE.format(staging_table=staging_table))
    cur.execute(f"TRUNCATE TABLE {staging_table}")

    print(f"[step 4] COPY INTO {staging_table} from @{stage}")
    cur.execute(f"""
        COPY INTO {staging_table}
        FROM @{stage}
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
        ON_ERROR = CONTINUE
        PURGE = FALSE
    """)
    copy_results = cur.fetchall()
    loaded = sum(r[3] for r in copy_results if r[3] is not None)  # rows_loaded column
    skipped = sum(1 for r in copy_results if r[1] == 'LOAD_SKIPPED')
    failed = sum(1 for r in copy_results if r[1] == 'LOAD_FAILED')
    print(f"[step 4] COPY results: {loaded:,} rows loaded, {skipped} files skipped, {failed} files failed")
    if failed > 0 or skipped > 0:
        print("[step 4] Failed/skipped files:")
        for r in copy_results:
            if r[1] in ('LOAD_FAILED', 'LOAD_SKIPPED'):
                print(f"  {r[0]}  status={r[1]}  errors={r[5]}")

    cur.execute(f"SELECT COUNT(*) FROM {staging_table}")
    row = cur.fetchone()
    print(f"[step 4] Staging table row count: {row[0]:,}")


# ---------------------------------------------------------------------------
# Step 5a: MERGE into concept_embeddings (target Athena concepts)
# ---------------------------------------------------------------------------

def merge_target_concepts(cur, embed_model_version: str) -> None:
    staging_table = _cfg('GPU_EMBED_STAGING_TABLE', _DEFAULT_STAGING_TABLE)
    print(f"\n[step 5a] MERGE target concepts -> concept_embeddings (version={embed_model_version})")
    cur.execute(f"""
        MERGE INTO concept_embeddings t
        USING (
            SELECT
                concept_id,
                vocabulary_id,
                domain_id,
                concept_name,
                standard_concept,
                embedding::VECTOR(FLOAT, 768) AS embedding,
                embedded_at
            FROM {staging_table}
            WHERE source_id IS NULL
              AND embedding IS NOT NULL
        ) s
          ON  t.concept_id         = s.concept_id
          AND t.embed_model_version = '{embed_model_version}'
        WHEN MATCHED THEN UPDATE SET
            vocabulary_id    = s.vocabulary_id,
            domain_id        = s.domain_id,
            concept_name     = s.concept_name,
            standard_concept = s.standard_concept,
            embedding        = s.embedding,
            embedded_at      = s.embedded_at
        WHEN NOT MATCHED THEN INSERT (
            concept_id, vocabulary_id, domain_id, concept_name,
            standard_concept, embedding, embedded_at, embed_model_version
        ) VALUES (
            s.concept_id, s.vocabulary_id, s.domain_id, s.concept_name,
            s.standard_concept, s.embedding, s.embedded_at, '{embed_model_version}'
        )
    """)
    cur.execute(f"SELECT COUNT(*) FROM concept_embeddings WHERE embed_model_version = '{embed_model_version}'")
    row = cur.fetchone()
    print(f"[step 5a] concept_embeddings row count for this version: {row[0]:,}")


# ---------------------------------------------------------------------------
# Step 5b: MERGE into source_concepts (query embeddings)
# ---------------------------------------------------------------------------

def merge_source_concepts(cur, embed_model_version: str) -> None:
    staging_table = _cfg('GPU_EMBED_STAGING_TABLE', _DEFAULT_STAGING_TABLE)
    print(f"\n[step 5b] MERGE source query vectors -> source_concepts (version={embed_model_version})")
    cur.execute(f"""
        MERGE INTO source_concepts t
        USING (
            SELECT
                source_id,
                mapping_wave,
                embedding::VECTOR(FLOAT, 768) AS query_embedding,
                embedded_at
            FROM {staging_table}
            WHERE source_id IS NOT NULL
              AND embedding IS NOT NULL
        ) s
          ON  t.mapping_wave = s.mapping_wave
          AND t.source_id    = s.source_id
        WHEN MATCHED THEN UPDATE SET
            query_embedding     = s.query_embedding,
            embed_model_version = '{embed_model_version}',
            embedded_at         = s.embedded_at
    """)
    cur.execute(f"SELECT COUNT(*) FROM source_concepts WHERE embed_model_version = '{embed_model_version}'")
    row = cur.fetchone()
    print(f"[step 5b] source_concepts with embeddings for this version: {row[0]:,}")


# ---------------------------------------------------------------------------
# Step 6: Cleanup
# ---------------------------------------------------------------------------

def cleanup(cur) -> None:
    stage = _cfg('GPU_EMBED_STAGE_NAME', _DEFAULT_STAGE_NAME)
    staging_table = _cfg('GPU_EMBED_STAGING_TABLE', _DEFAULT_STAGING_TABLE)
    print(f"\n[step 6] Truncating staging table")
    cur.execute(f"TRUNCATE TABLE {staging_table}")
    print(f"[step 6] Dropping internal stage @{stage}")
    cur.execute(f"DROP STAGE IF EXISTS {stage}")
    print("[step 6] Cleanup complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load S3 embeddings into Snowflake")
    parser.add_argument("--sync-only", action="store_true", help="Only sync from S3, skip Snowflake load")
    parser.add_argument("--load-only", action="store_true", help="Skip S3 sync, load from existing local files")
    parser.add_argument("--no-cleanup", action="store_true", help="Leave staging table and stage after load")
    parser.add_argument("--put-only", action="store_true", help="Only PUT files to stage, skip COPY/MERGE")
    parser.add_argument("--merge-only", action="store_true", help="Skip PUT/COPY, re-run only the MERGE steps from existing staging table")
    parser.add_argument("--rollback", action="store_true", help=f"DELETE all rows from concept_embeddings for version {_DEFAULT_EMBED_MODEL_VERSION} and exit")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip files already in the stage (OVERWRITE=FALSE). Default is OVERWRITE=TRUE.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel PUT workers (default: 4)")
    parser.add_argument("--embed-model-version",
                        default=_cfg('GPU_EMBED_RETRIEVAL_VERSION', _DEFAULT_EMBED_MODEL_VERSION),
                        help="embed_model_version to stamp (default: GPU_EMBED_RETRIEVAL_VERSION env var or hardcoded fallback)")
    args = parser.parse_args()

    _load_env()

    if not args.load_only and not args.merge_only and not args.rollback:
        sync_from_s3()

    if args.sync_only:
        print("[done] S3 sync complete. Re-run with --load-only to load into Snowflake.")
        return

    conn = _get_connection()
    try:
        cur = conn.cursor()

        if args.rollback:
            print(f"[rollback] Deleting all rows from concept_embeddings where "
                  f"embed_model_version = '{args.embed_model_version}'")
            cur.execute(f"DELETE FROM concept_embeddings "
                        f"WHERE embed_model_version = '{args.embed_model_version}'")
            cur.execute("SELECT COUNT(*) FROM concept_embeddings")
            print(f"[rollback] Done. concept_embeddings now has {cur.fetchone()[0]:,} rows total.")
            cur.close()
            return

        if not args.merge_only:
            create_stage(cur)
            put_files(conn, workers=args.workers, overwrite=not args.no_overwrite)
            if args.put_only:
                print("[done] PUT complete. Stage left in place. Re-run with --merge-only to finish.")
                cur.close()
                return
            copy_into_staging(cur)

        merge_target_concepts(cur, args.embed_model_version)
        merge_source_concepts(cur, args.embed_model_version)

        if not args.no_cleanup:
            cleanup(cur)
        else:
            print("[info] --no-cleanup set; staging table and stage left in place")

        cur.close()
    finally:
        conn.close()

    print("\n[done] Load complete.")


if __name__ == "__main__":
    main()
