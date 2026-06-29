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

**Standing decision (unchanged by this addendum):** pure DuckDB
(`embeddings.duckdb`) is the recommended live store; the parquet backend is a
demoted opt-in migration/export artifact (README "Storage model"). The
benchmarks below characterize the *costs* of that chosen backend and inform any
future revisit — they do **not** by themselves reopen the decision. A hard
architectural constraint also applies: **no separate client-server RDBMS**
(Postgres/etc.). Concepts flow in and embeddings flow out continuously, so a
server round-trip on every move is unacceptable; this keeps the field to
*embedded, file-based* stores (DuckDB, parquet, Delta-rs, Lance, …), and makes
Spark/Delta-on-Spark doubly irrelevant.

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
- **delta-rs (no Spark/JVM) addresses the read side** (ACID log, OPTIMIZE
  compaction, lock-free multi-reader concurrency) — *but its MERGE upsert does
  not survive a realistic re-embed; see the scattered-re-embed result below,
  which reverses the initial lean toward delta-rs MERGE.*
- **Spark is still not justified** — all of the above ran single-node in 59s.

### Scattered re-embed result (2026-06): delta-rs MERGE reintroduces the amplification

The favorable "10s / 0-duplicate" MERGE above used 200k **contiguous** ids,
which cluster into whole files. Real re-embeds (an Athena refresh changing a
subset of concept names) scatter across the row-space, and delta-rs MERGE
rewrites whole *files* containing any matched row. Measured on a 1M-row
`model_version`, re-embedding the same 200k rows:

| Re-embed 200k of 1M | files rewritten | updated | rows copied (amplification) | output rows |
|---|---|---|---|---|
| Delta MERGE, contiguous ids | 6 of 30 | 200k | **0** (0.0×) | 200k |
| Delta MERGE, scattered ids | **30 of 30** | 200k | **800k (4.0×)** | **1,000,000** |
| Delta MERGE, scattered, `[mv, vocabulary_id]` | 40 of 40 | 200k | **800k (4.0×)** | 1,000,000 |
| Parquet append (current), contiguous | n/a | — | — | 200k (+200k stale) |
| Parquet append (current), scattered | n/a | — | — | 200k (+200k stale) |

- **Scattered MERGE rewrote the entire partition** (all files, 1M rows) to
  change 200k — 4× amplification that scales with *partition size, not change
  size*. At 12M this rewrites the whole model_version partition per re-embed.
  This is precisely original issues #1/#2, reborn inside Delta.
- **Finer partitioning does not help** (identical 4.0×): an Athena refresh
  scatters changes uniformly, so every partition's files get a mix and all are
  rewritten. Partitioning only helps when changes concentrate.
- **The current append-only parquet path is scatter-insensitive** — identical
  cost contiguous vs scattered (append writes only the 200k changed rows as new
  shards, O(changes)). Its only cost is read-side duplicate growth, which is
  bounded and fixable by periodic compaction.

### Revised stance (supersedes the lean above)

The real choice is a **write strategy**, not an engine:

- *Upsert-on-write* — DuckDB `INSERT OR REPLACE` (~50s/200k at 2M and growing)
  or Delta `MERGE` (rewrites whole partition on scatter). Both scale re-embed
  cost with table/partition size; both lose at scale on the re-embed path.
- *Append + merge-on-read + periodic compaction* — what the current parquet
  backend already does, minus compaction. Writes are always O(changes) and
  scatter-insensitive; read cost is bounded by compacting on a schedule, off
  the write critical path.

Recommendations (all within the embedded / no-separate-RDBMS constraint):

1. **Reject delta-rs MERGE for the re-embed path** — it is the ADR's
   amplification with a nicer API (4× on a scattered re-embed).
2. **Staying pure-DuckDB (the standing decision) is defensible.** Its costs are
   the re-embed time (~50s/200k at 2M and growing), no cross-process concurrency
   (exclusive file lock), and ~2× disk. If the *lock* is the only pain, solve it
   *within* DuckDB via in-process cursor-per-thread concurrency
   (`duckdb_concurrency.md`), not a backend change.
3. **If the re-embed cost or concurrency limit becomes the real bottleneck**, the
   correct write shape is **append + merge-on-read + periodic compaction** — which
   the (demoted) parquet backend already implements bar compaction, and which
   Delta in *append* mode (not MERGE) also provides with an ACID log. Either
   reopens the parquet decision, so do it deliberately, not by drift.
4. **Lance / LanceDB — benchmarked, recommended, and now ADOPTED (2026-06-29);
   see `adr_lance_store_proposal.md`.** Its deletion-vector update model made a
   scattered 200k re-embed **O(changes)** (~200k rows written, ~3s) instead of
   Delta MERGE's O(partition) (1M rows, 4×), with ACID snapshot isolation and
   multi-reader concurrency confirmed. Given the ACID + concurrency requirement,
   Lance is the adopted live store and is now the **default** backend
   (`embeddings.lance`); Delta-rs is rejected (MERGE amplifies; append mode adds
   no native upsert over the parquet path). Implemented in `store.py` + the
   `migrate-lance` / `compact` CLI commands (see `adr_lance_store_proposal.md`).

Criterion #4 (write throughput ≥ DuckDB table) is met for *writes*, but the
re-embed amplification means a wholesale parquet/Delta-MERGE switch is **not**
warranted. Spark/Delta-on-Spark remains rejected (and is moot under the
embedded-only, no-separate-RDBMS constraint).

### Caveats (do not over-read)

- Synthetic random vectors, single process, local SSD; 2M ≈ ⅙ of production
  (12M). Trends are consistent across 200k→2M but absolute ms scale up.
- The favorable ~10s/0-dup MERGE result depended on the re-embed being
  contiguous (file-clustered). The scattered case — the realistic one — was
  measured above and shows 4× amplification; that is the result to trust for
  this workload, not the contiguous figure.
- OPTIMIZE looked near-free only because the preceding (contiguous) MERGE had
  already consolidated the touched partition.

### Migration & Snowflake-handoff considerations (must address before any switch)

Any backend change is a *migration*, not a greenfield choice — there is already
a substantial production `.duckdb` store with many embeddings. Two requirements
gate a switch:

1. **Migrate the existing DuckDB-table data into the new backend.** Machinery
   already exists and should be the template: `store._migrate_legacy_if_needed`
   ATTACHes a legacy `.duckdb` read-only and streams it into partitioned shards
   (`_copy_relation_to_partitioned_shards`). A `delta` backend would reuse the
   same ATTACH-and-stream pattern, writing Arrow batches via `write_deltalake`
   instead of `COPY ... TO parquet`. The migration must be re-runnable and not
   require holding the whole table in memory.
2. **Preserve the Snowflake handoff.** The `export` command
   (`cli.export_cmd`) emits plain sharded Parquet that Snowflake reads via
   `COPY INTO` / external stage; this is independent of the live store backend
   and **must keep working** regardless of what backend embed writes to. Note a
   Delta table *is* Parquet files plus a `_delta_log/`; Snowflake can consume
   Delta directly (external table / Iceberg-style), but that is more setup than
   the existing plain-Parquet export. Safest stance: keep `export` → plain
   Parquet as the canonical Snowflake path even if the live store moves to
   Delta, so the downstream contract does not change.
