# DuckDB concurrency: embed (writes) vs export/status (reads)

**Date:** 2026-06
**Status:** Notes / options — no change made yet

---

## Problem

We want `embed` (long-running, intermittent writes) to run while `export` and
`status` (read-only) run too. Today each is a separate `gpu-embed` process, and
each calls `store.open_db` (`store.py`), so each opens the `.duckdb` file
independently.

## The constraint (verified on duckdb 1.5.4)

- **Across processes:** DuckDB allows exactly one process to hold a `.duckdb`
  file. A second process gets
  `IOException: Could not set lock on file ... Conflicting lock is held`.
  **A read-only open also fails** while a writer holds the file. So concurrent
  `embed` + `export`/`status` against the same `.duckdb` is impossible by
  design. (See <https://duckdb.org/docs/stable/connect/concurrency>.)
- **Within one process:** the opposite. DuckDB's MVCC lets one writer and any
  number of readers run concurrently on the same database instance with
  snapshot isolation — verified: a reader thread watched `COUNT(*)` climb
  `1 → 501` while a writer thread inserted, with no blocking, no lock error,
  and no stale-read crash.
- **One rule for threading:** a single `DuckDBPyConnection` is **not** safe to
  call from multiple threads at once. The supported pattern is one
  `conn.cursor()` per thread — each cursor is an independent connection sharing
  the same instance, transactions, and MVCC.

Current behaviour worth knowing: `embed_cmd` opens the connection at the start
of the run and never closes it (`cli.py`), holding the RW lock for the *entire*
run, not just during checkpoint writes — so it locks everyone out start to
finish, not only at the moment of a write.

## Options

1. **Single owning process, `cursor()` per thread** (true live concurrency).
   One process owns the connection; embed does its upserts while export/status
   run as threads that each call `conn.cursor()` and read. Readers see a
   consistent snapshot, never block the writer, never hit the lock. Requires a
   structural change — a long-lived process (e.g. a `gpu-embed serve` daemon, or
   embed dispatching export/status as in-process tasks) instead of three
   independent CLI invocations. Embed is the only writer, so no write-write
   conflicts.

2. **Separate processes, snapshot copy for readers** (cheapest in code, but see
   caveat). Readers can't touch the live file, so periodically `CHECKPOINT` and
   copy the `.duckdb` (or `EXPORT DATABASE`) to a snapshot path, and point
   `export`/`status` at the copy.

   > **Caveat — do not assume this is cheap.** The production `.duckdb` is
   > **tens of GB** (12 M rows ≈ 94 GB pre-parquet; see
   > `adr_parquet_store_rejected.md`). Copying tens of GB on every
   > `status`/`export` is expensive in both wall-clock and disk, and a plain
   > file copy of a live, mid-write DuckDB file is **not** crash-consistent —
   > it must go through `CHECKPOINT` + a quiescent copy (or `EXPORT DATABASE`),
   > which competes with the embed run for I/O. This makes Option 2 a poor fit
   > for frequent `status` polling at our data scale; treat it as a
   > coarse-grained, occasional-snapshot tool at best.

3. **Parquet-backed store** (already implemented in `store.py`, but
   ADR-rejected). The one mode that is inherently multi-process: embed appends
   immutable shards, export/status open `:memory:` DuckDB reading the shards via
   globs — many concurrent readers, no file lock, a reader can run during an
   embed. Rejected as the live write store for write amplification
   (whole-partition rewrite per checkpoint); see
   `adr_parquet_store_rejected.md`. Buys concurrency at a write cost already
   evaluated and declined.

4. **IPC.** Keep separate processes but have the embed process expose a small
   endpoint that runs read queries on a cursor and returns results; export/status
   become thin clients. Most plumbing; only worth it if consolidating into one
   process is not viable.

## Leaning

For **live** reads mid-run → Option 1 (in-process, cursor-per-thread). Option 2
(snapshot copy) is tempting because it needs no architecture change, but the
tens-of-GB copy cost makes it unsuitable for anything more than infrequent
snapshots — do not reach for it as the default just because it is the least
code.

---

## Benchmark (2026-06): does a file-based backend make concurrency cheap?

The cross-process lock above is a property of the **DuckDB-table** backend
only. The file-based backends (Options 3 above) sidestep it — many reader
processes can scan the files while embed writes — so the concurrency question
is entangled with the storage-backend question. A storage-only benchmark
(2M rows = 2 model_versions × 1M, then a 200k re-embed) was run to see whether a
file backend is viable now that the write path is append-only and Arrow-native.
Full numbers and caveats live in `adr_parquet_store_rejected.md` (2026-06
addendum); the concurrency-relevant takeaways:

- The current **parquet** backend already gives lock-free multi-reader access
  and flat append writes (~6.9s/100k at 2M, no growth), but reads carry a
  dedup-view cost (~710ms full COUNT) and accumulate duplicates with **no
  compaction**.
- **delta-rs** (Delta Lake without Spark/JVM) gives the same lock-free
  multi-reader concurrency *plus* an ACID transaction log (readers see only
  committed snapshots) and managed OPTIMIZE compaction — but **only in append
  mode**. Its MERGE upsert reintroduces write amplification on realistic
  (scattered) re-embeds — see the scattered-re-embed result in
  `adr_parquet_store_rejected.md` (4× amplification, rewrites the whole
  partition). So MERGE is the part of Delta this workload must avoid.

Given pure DuckDB is the standing live backend, the proportionate concurrency
answer is **in-process, cursor-per-thread** (Option 1): no backend change, and
DuckDB's MVCC already supports one writer plus concurrent readers in a single
process. Cross-process concurrency is impossible with one `.duckdb` file
(exclusive lock), and the snapshot-copy workaround is unattractive at tens of GB.

True cross-process multi-reader access requires a **file-based** backend — the
demoted parquet path + compaction, Delta-rs in *append* mode (not MERGE), or
Lance — but that reopens the storage-backend decision
(`adr_parquet_store_rejected.md`) and is constrained to **embedded** stores (no
separate client-server RDBMS, by requirement). Spark is not relevant under that
constraint.

When the requirement is **ACID + cross-process concurrency** (not just
concurrency), pure DuckDB cannot satisfy it and the recommended path is **Lance**
as the live store — benchmarked O(changes) upserts with ACID snapshot isolation.
See `adr_lance_store_proposal.md`.
