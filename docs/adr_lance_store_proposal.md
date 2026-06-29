# ADR: Lance as the ACID + concurrent embedding store

**Date:** 2026-06
**Status:** **Accepted** (2026-06-29) — Lance is adopted as the live store for
the ACID + cross-process-concurrency requirement; **delta-rs is rejected** (its
`MERGE` amplifies on scattered re-embeds, and its append mode offers no native
upsert/dedup, only the parquet backend's discipline plus an ACID log). DuckDB
remains the default backend; Lance is opt-in via a `.lance` store path.
Supersedes the delta-rs lean in `adr_parquet_store_rejected.md` (2026-06
addendum). A flag-gated prototype now exists (see "Implementation" below).

---

## Implementation (prototype, 2026-06-29)

Behind the existing `store.open_db` path dispatch — a `.lance` store path selects
the Lance backend, so nothing changes unless that path is chosen:

- **Backend in `store.py`:** `merge_insert` upsert (O(changes)), DuckDB-as-query
  layer (`concept_embeddings` is a view over the registered Lance dataset, so all
  readers/`export` are unchanged), `delete` by predicate, and the model registry
  reused from the parquet `_meta/model_registry` layout. csv-fingerprint and
  model-version caches are no-ops (as for parquet — re-hash/re-read rather than
  risk a stale hit).
- **CLI:** `gpu-embed migrate-lance` (streaming, re-runnable `duckdb → lance`),
  `gpu-embed compact` (`compact_files` + `cleanup_old_versions`), and `export`
  works unchanged (`lance → parquet` via the view).
- **Dependency:** optional `pylance` extra (`uv sync --extra lance`); imported
  lazily so the default backends do not require it. Tests in
  `tests/unit/test_store_lance.py` cover upsert/merge, classify, delete,
  registry, compaction, migration, and `lance → parquet` export.
- **Layout:** a `.lance` store is a container dir — dataset at
  `<store>.lance/concept_embeddings.lance/`, registry at `<store>.lance/_meta/`
  — so Lance maintenance never touches the metadata and vice versa. Migration
  casts to the canonical Arrow schema so a post-migration `embed` merges without
  a schema clash.

Open validation items remain in "Caveats" (compaction memory at 12M; a concrete
retention policy; export-to-parquet throughput at scale).

---

## Driver

The hard requirement is **ACID + cross-process concurrency**: `embed` writes
(intermittent, append + re-embed of changed rows) must run while `export` /
`status` read, ideally from separate processes. Secondary: embedded only (**no
separate client-server RDBMS** — concepts flow in and embeddings out
continuously), migrate the existing `embeddings.duckdb`, preserve the Snowflake
Parquet handoff, and reduce future operational friction.

## Candidate elimination

| Backend | ACID | Cross-process concurrency | Verdict |
|---|---|---|---|
| Pure DuckDB (current live store) | ✅ | ❌ exclusive file lock | fails concurrency |
| Raw parquet (demoted backend) | ❌ hand-rolled | ✅ | fails ACID |
| Delta-rs (append mode) | ✅ txn log | ✅ multi-reader | shortlist |
| **Lance** | ✅ versioned manifest | ✅ multi-reader | **recommended** |

Pure DuckDB cannot meet the primary constraint (the exclusive `.duckdb` lock —
see `duckdb_concurrency.md`), so the live store moves off it for this workload.

## Deciding benchmark (1M rows, scattered 200k re-embed)

The re-embed is *scattered* (an Athena refresh changes a subset of names across
the row-space) — the workload that broke Delta MERGE (rewrote the whole
partition, 4× amplification). Synthetic 768-d float32 vectors, same harness as
the runs in `adr_parquet_store_rejected.md`.

| | DuckDB `INSERT OR REPLACE` | Delta-rs `MERGE` | Lance `merge_insert` |
|---|---|---|---|
| Re-embed 200k scattered | ~50s (at 2M, grows) | ~10s, rewrote **1,000,000** rows | ~3s, wrote **200,000** rows |
| Amplification | — | **O(partition)** (4×) | **O(changes)** (1×) |
| Update model | in-place upsert | copy-on-write file rewrite | deletion vectors + new fragment |

Lance, measured:
- Append load flat (~0.65s/100k).
- `merge_insert` of 200k scattered: ~3.07s, **+200k physical rows, +200k
  deletions, ~618 MB written ≈ 200,073 rows** — O(changes), not O(partition).
  Fragments 10→11.
- **ACID snapshot isolation confirmed**: a reader pinned to version v10 kept
  seeing the pre-update snapshot while a fresh handle saw v11 (atomic commit).
- Read after update: 1M logical rows, exactly 200k marked changed, **no
  duplicate inflation** (dedup is native via deletion vectors).
- `compact_files`: 11→1 fragment in ~7s, reclaims tombstoned rows.
- Bonus: native ANN vector index (aligns with the README stretch `query`).

## Recommendation: adopt Lance as the live embedding store

1. Satisfies ACID + concurrency; native upsert is **O(changes)** — no Delta
   MERGE amplification and no "never use MERGE" discipline; fastest re-embed of
   the three; embedded, no server.
2. **Migration** reuses the existing template: `store._migrate_legacy_if_needed`
   ATTACHes `embeddings.duckdb` read-only and streams Arrow batches — point that
   stream at `lance.write_dataset(..., mode="append")` instead of `COPY ... TO
   parquet`. Re-runnable, streaming, no full-table-in-memory.
3. **Snowflake handoff unchanged**: `export` reads Lance → plain sharded Parquet
   (Snowflake does not read Lance natively; Parquet remains the contract).
4. Slot a `lance` backend behind the existing `store.open_db` path dispatch
   (e.g. a `.lance` directory), alongside `duckdb`/`parquet`.

## Caveats (validate before committing)

- Synthetic data, single process, 1M rows (≈1/12 of production), local SSD.
  `merge_insert` being O(changes) is structural and should hold at 12M; but
  compaction cost grows with dataset size (scheduled, off the write path).
- **Version retention**: Lance keeps old versions for time-travel until cleaned
  up; compaction transiently doubled disk (3.1 GB → 6.8 GB pre-vacuum). Set a
  retention/cleanup policy or disk grows.
- Heavy analytical scan / export-to-Parquet throughput was not stress-tested
  beyond `count_rows`; validate export throughput at scale.
- New dependency (`pylance`). If its maturity/ops are a concern, **Delta-rs in
  append mode is the fallback** — same ACID + concurrency, but requires the
  append + scheduled-compaction discipline (never MERGE) to avoid amplification.
