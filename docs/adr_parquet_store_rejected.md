# ADR: Parquet-backed store rejected in favour of native DuckDB table

**Date:** 2026-06  
**Status:** Decided — do not revisit without addressing every issue below

---

## Context

During a refactor to make embeddings Snowflake-loadable, a Parquet-sharded
storage backend was implemented as the primary write target, with DuckDB used
only as an in-memory query layer on top of the files.  After a full production
migration (12 M rows, ~94 GB) the approach was abandoned and the default was
reverted to a native DuckDB persistent table (`embeddings.duckdb`).

---

## Problems encountered (in order of discovery)

### 1. No native upsert — rewrites entire partition on every write

Parquet files are immutable.  "Upsert" in DuckDB's Parquet layer means:
_read the partition into memory, filter duplicates, write a new file_.  On a
12 M-row table partitioned by `(model_version, vocabulary_id)`, a single
checkpoint write touched megabytes of data for a few hundred new rows.
There is no equivalent of a B-tree index or WAL.

### 2. Write amplification grew with table size

Because the whole shard had to be rewritten, write cost scaled with the
_existing_ data in the partition, not with the _new_ data being added.
Checkpointing every N batches became slower the further into a run we were —
the opposite of the desired behaviour.

### 3. `_copy_relation_to_partitioned_shards` re-ranked per shard

The initial implementation sorted and ranked rows inside each shard loop
instead of once per partition, causing O(shards × rows) work.  This was fixed,
but the fix was a band-aid: the root cause was the rewrite-on-write model.

### 4. `_refresh_view` called after every checkpoint

The `GLOB`-backed DuckDB view over the Parquet tree had to be recreated
whenever new files appeared.  The first implementation called `_refresh_view`
after every single `upsert_rows` call inside the embed loop — adding a full
directory scan + `CREATE OR REPLACE VIEW` to every checkpoint.  Deferring the
call to end-of-run masked the cost but did not remove it; any crash before
that point left the view stale.

### 5. Compression choice was a hidden write bottleneck

ZSTD (the DuckDB default for Parquet) added meaningful CPU cost per shard
rewrite.  Switching to Snappy improved throughput, but that just exposed
issue #2 more clearly: even with fast compression, rewriting 10 MB to add 200
rows is wasteful.

### 6. Startup cost: full table scan to build existing-ID set

`get_existing_ids` returned a `set[int]` of every already-embedded concept ID.
With 12 M rows this transferred ~96 MB of integers from DuckDB to Python on
every run before a single embedding was computed.  The fix (`filter_unembedded_rows`,
a SQL anti-join against only the current batch's candidates) was only necessary
because the Parquet layer had no index and no cheap `EXISTS` check.

### 7. Shard management was all custom code

Partition layout (`model_version=*/vocabulary_id=*/part-*.parquet`), shard
size bounds, file naming, and the glob pattern in the view had to be maintained
manually.  Every schema change required auditing all of those callsites.  A
native DuckDB table handles this transparently.

### 8. No enforced uniqueness at the file layer

The `(concept_id, model_version)` PRIMARY KEY is enforced only at the DuckDB
layer; the Parquet files themselves allow duplicates.  Any tool that reads the
files directly (Spark, polars, Snowflake `COPY INTO`) would ingest duplicates
silently if rows were written before deduplication was applied.

---

## Decision

**Use a native DuckDB persistent file (`.duckdb` suffix) as the primary write
store.**  The Parquet export path (`gpu-embed export`) is retained for
Snowflake handoff, but it is generated on-demand from the DuckDB table, not
maintained as the live store.

The hybrid rule is encoded in `store.open_db`:
- path ends in `.duckdb` → native DuckDB table, full PK enforcement, B-tree
  index, fast `EXISTS`/anti-join, no rewrite amplification.
- path is a directory → Parquet-backed store (migration artifact / explicit
  opt-in only).

---

## When Parquet-backed might be worth revisiting

Only consider it again if **all** of the following are true:

1. The dataset exceeds ~100 M rows and the `.duckdb` file itself becomes a
   bottleneck (e.g., WAL flush or memory-map limits).
2. A write path that avoids full-partition rewrites is available (e.g., Delta
   Lake / Iceberg with merge-on-read).
3. Direct Parquet consumption by downstream systems (Snowflake, Spark) is
   required without an export step.
4. A benchmarked prototype confirms write throughput ≥ the DuckDB-table path
   at the target scale.
