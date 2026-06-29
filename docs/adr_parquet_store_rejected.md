# ADR: Parquet-backed store rejected in favour of native DuckDB table

**Date:** 2026-06  
**Status:** Decided — **but see the 2026-06 addendum at the bottom**, which
re-benchmarks the central premise and finds it no longer matches the code.

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

---

## Addendum (2026-06): benchmarks now contradict the central premise

After this ADR was written, two things changed: (a) the parquet write path was
rewritten to be **append-only merge-on-read** (new shards per checkpoint +
read-time dedup view; see `store._copy_relation_to_partitioned_shards` /
`_refresh_view`) — it no longer rewrites whole partitions, and (b) the whole
pipeline moved to Arrow-native batch I/O. A storage-only benchmark (synthetic
768-d float32 vectors, real `store.py` code for the DuckDB/parquet backends,
`deltalake` 1.6.1 for Delta; no GPU) was run at 200k and at **2M rows
(2 model_versions × 1M, then re-embed 200k changed rows in one model_version)**.

### 2M-row results

| Metric | DuckDB table (current default) | Parquet (current code) | Delta (delta-rs, no Spark) |
|---|---|---|---|
| Write/100k checkpoint (mean) | 12.8s | 6.9s | 1.2s |
| Write scaling (first→last) | 9.2→12.4s (grows) | 6.9→7.1s (flat) | 1.9→1.2s (flat) |
| Full 2M load wall-clock | 353s | 180s | 59s |
| Re-embed 200k (upsert) | **49.9s** | 13.9s (+dupes) | 10.2s (MERGE, 0 dupes) |
| Read — full COUNT | 1.8ms | 710ms | 62ms |
| Read after re-embed | 1.4ms | 858ms (degraded) | 140ms |
| On-disk | 12.1 GB | 6.8 GB (+200k stale rows) | 6.2 GB (compacted) |
| Duplicates / compaction | 0 (PK) | 200k stale, **none** | 0, OPTIMIZE cheap |

### What the data overturns

- **Issues #1 and #2 (rewrite-on-write, amplification growing with size) no
  longer describe the code.** The current parquet path's writes are *flat*
  (6.9→7.1s). It is the **DuckDB table** whose per-checkpoint write cost grows
  (9.2→12.4s) and whose re-embed of changed rows is the real bottleneck:
  **~50s for 200k rows at 2M total**, and it worsens toward the production 12M.
  That is the exact "changed `embed_text`" path `embed` exercises.
- **The current parquet weakness is the read side, not writes:** unbounded
  duplicate accumulation (200k stale rows after one re-embed) with **no
  compaction**, and a full-table dedup COUNT of ~710ms at 2M.
- **delta-rs (no Spark/JVM) addresses both** and the key scaling risk holds up:
  partitioned by `model_version`, the re-embed MERGE pruned to that partition's
  30 files, ran in ~10s with **zero duplicates**, kept reads clean (140ms), and
  OPTIMIZE compaction was cheap. It addresses original issues #1, #2, #7, #8.
- **Spark is still not justified** — all of the above ran single-node in 59s.

### Caveats (do not over-read)

- Synthetic random vectors, single process, local SSD; 2M ≈ ⅙ of production
  (12M). Trends are consistent across 200k→2M but absolute ms scale up.
- delta-rs MERGE rewrites whole *files* containing any matched row, so the
  favorable ~10s/0-dup result depends on the re-embed being **model_version-
  scoped** (partition-prunable). A re-embed scattered across many
  vocabularies/model_versions would amplify and must be measured separately
  before committing.
- OPTIMIZE looked near-free only because the preceding MERGE had already
  consolidated the touched partition.

### Revised stance

The "do not revisit" disposition above is **superseded for the Delta-rs case**:
revisit criterion #4 (write throughput ≥ DuckDB table) is met by a wide margin,
and criteria #2 (a merge-on-read write path) and #3 (direct downstream
consumption without an export step) are satisfied by Delta's transaction log.
Criterion #1 (>100M rows) is *not* met, so this is not yet a forcing function —
but the DuckDB-table re-embed cost is becoming one on its own. Recommended next
step is a flag-gated `delta` backend behind the existing `store.open_db` path
dispatch, validated against a realistic scattered re-embed at larger scale.
Spark/Delta-on-Spark remains rejected.
