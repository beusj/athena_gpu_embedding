# Performance: moving bulk data between Python and DuckDB

This note records why all bulk Python→DuckDB transfer in `store.py` goes through
Arrow, and the two slow approaches that preceded it — so they are not
reintroduced. The authoritative rule lives in `AGENTS.md` ("DuckDB" section);
this is the background.

## The rule

To move data that **scales with row count** into DuckDB, build a columnar
`pyarrow.Table`, `conn.register(...)` it, and `INSERT ... SELECT` (or JOIN)
against the registered relation. DuckDB ingests Arrow zero-copy and fully
vectorised. This covers:

- **Embedding writes** — `_embedded_rows_to_arrow()` → `INSERT OR REPLACE ...
  SELECT` (embedding encoded as a `FixedSizeList` of float32 → `FLOAT[768]`).
- **Change detection** — `classify_rows_requiring_embedding()` registers the
  `(namespace, concept_id, embed_text)` candidate columns and LEFT JOINs them
  against `concept_embeddings`.

## Approaches that were tried and abandoned

### 1. Single `executemany` for the whole batch (original)
The first `upsert_rows()` did one `executemany` over all rows. The process
appeared to hang in the write phase: each row crossed the Python→DuckDB boundary
individually and the PRIMARY KEY index was maintained per row, with a ~3KB
`FLOAT[768]` payload each. Catastrophic on large batches.

### 2. Chunked `executemany` (interim)
Splitting the insert into fixed-size chunks gave incremental progress and helped
small runs, but it is the *same* per-row binding underneath — still far too slow
for millions of rows. Superseded by Arrow (≈100× faster on the embedding write).

### 3. `unnest(?::T[])` array params (interim, for classify)
To avoid `executemany`, the candidate columns were briefly passed as
`unnest(?::BIGINT[])` / `unnest(?::VARCHAR[])` params. Binding a large Python
list as a single array value is **~quadratic** in DuckDB: ~54s for 300k
candidates and effectively unbounded at millions — it presented as a multi-minute
stall after model load (GPU idle, no progress bar, growing WAL) before any
embedding started. Replaced with Arrow registration (1M candidates in ~1.5s).

`unnest(?::T[])` remains acceptable only for **small, bounded** lists (e.g. a
handful of filter values), never for a per-concept column.

## Phases, for context

Work after the model loads runs in three phases — keep them distinct when
diagnosing a stall:

1. **classify** (`classify_rows_requiring_embedding`, CPU/DuckDB) — decides which
   concepts are new/changed/unchanged. No GPU.
2. **embed** (`embed_all`, GPU) — the SapBERT/BioLORD forward pass over the
   rows classify selected (the `Embedding (cuda)` progress bar).
3. **upsert** (`upsert_rows`) — write vectors via Arrow.

A hang with the GPU idle and no progress bar is the classify/write data path,
not the model — check the Arrow path first.
