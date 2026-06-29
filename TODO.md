# TODO — gpu-embedder

The original phase-1..7 implementation plan is **complete**. The full pipeline
(config → ingest → embed → store → CLI) shipped long ago and has grown well
beyond what the plan described, so this file now tracks **what shipped** (the
original plan plus everything added since) and **what remains**, rather than the
not-yet-started checklist it began as.

Status legend: `[x]` shipped · `[ ]` not done.

---

## Status at a glance

- **Storage backends (3), selected by the `--db` path suffix:**
  - `.lance` → Lance store — **the default** (`embeddings.lance`). ACID,
    cross-process readers, O(changes) `merge_insert`. Metadata (CSV fingerprints
    + weight-hash cache) persists in a `<store>.lance/_meta/meta.duckdb` sidecar.
  - `.duckdb` → native DuckDB single-writer table (the prior default).
  - directory (no suffix) → parquet shards — an export/migration artifact, not a
    live write store.
  - See `docs/adr_lance_store_proposal.md` (Lance accepted as default) and
    `docs/adr_parquet_store_rejected.md`.
- **CLI subcommands (10):** `embed`, `cpt4`, `migrate-store`, `migrate-lance`,
  `compact`, `export`, `status`, `model-registry`, `coverage`, `cleanup`.
- Change-detection, namespaces, selectable pooling, and source-concept embedding
  all shipped (details below).

---

## Done — original plan (Phases 1–7)

- [x] **Phase 1 — Scaffolding.** `pyproject.toml` (`uv`), `.env.example`,
  package + test layout, `conftest.py` with the `gpu` mark + auto-skip, fixture
  TSVs, `fail_under = 80` coverage gate.
  - Divergence: ingest uses **DuckDB**, not `polars`/`pandas` (the plan's
    tentative dep was never added). `torch` is routed via the `cpu`/`gpu` extras;
    `pylance` is a **base** dependency now that Lance is the default.
- [x] **Phase 2 — Config & models.** `EmbedConfig(BaseSettings)` (env prefix
  `GPU_EMBED_`, device auto-detect), `ConceptRow`, `FilterSpec`, `SCHEMA_DDL`,
  plus `CSV_FINGERPRINTS_DDL` / `MODEL_VERSION_CACHE_DDL`. Default `db` is
  `embeddings.lance`.
- [x] **Phase 3 — Ingest & filtering.** `read_csv` / `filter_rows` (pure),
  DuckDB scan+filter engine (with a Python fallback), multi-file input.
- [x] **Phase 4 — Embedding.** `compute_model_version` (weight SHA-256, with
  pooling folded in), `load_model` (FP32), `embed_batch`, `embed_all`. CLS **and**
  mean pooling are supported (`--pooling`).
- [x] **Phase 5 — Store.** Shipped, but the API the plan sketched changed:
  - `get_existing_ids(conn, model_version) -> set[int]` was **removed** —
    materializing the existing-ID set did not scale. Change-detection is now
    `classify_rows_requiring_embedding(...)`, a SQL anti-join of the candidate
    batch against the store (see `docs/adr_parquet_store_rejected.md`).
  - Real public surface: `open_db`, `close_store`, `ensure_schema`,
    `refresh_view`, `classify_rows_requiring_embedding`, `upsert_rows` (Arrow
    bulk-load; `merge_insert` for Lance), `count_rows` / `count_embeddings` /
    `list_vocabulary_counts`, `delete_embeddings`, `delete_csv_fingerprints`,
    `delete_model_metadata`, `upsert_model_registry` / `list_model_registry`,
    `get_csv_fingerprint` / `upsert_csv_fingerprint`, `get_cached_model_version`
    / `upsert_model_version_cache`, `compact`, `migrate_duckdb_to_lance`.
  - Backend dispatch is by path suffix in `_resolve_paths`; the Lance metadata
    sidecar is wired through the private `_meta_conn` helper.
- [x] **Phase 6 — CLI.** `embed` (default) + `cpt4` shipped, and the surface grew
  to ten subcommands (see "at a glance"). The `embed` flow's skip step is no
  longer a `get_existing_ids` set — it is the anti-join `classify` plus the
  CSV-fingerprint and model-version-cache short-circuits.
- [x] **Phase 7 — Polish & CI.** ruff (`line-length = 100`, `E/W/F/I/UP`),
  `mypy --strict`, logging, `tqdm` progress, coverage gate.

---

## Done — shipped beyond the original plan

- [x] **Lance store backend — the DEFAULT.** ACID + concurrent reads, O(changes)
  `merge_insert`, the `_meta/meta.duckdb` metadata sidecar, and the
  `migrate-lance` + `compact` commands. (ADR accepted.)
- [x] **Parquet store backend** (directory `--db`) + `migrate-store`. This is the
  original plan's `--output-format parquet` stretch item — shipped as a real
  backend, then demoted to an export/migration artifact (Lance superseded it as
  the live store).
- [x] **Source-concept embedding** (`--source-parquet`) for concept-mapper inputs,
  round-tripping on `(mapping_wave, source_id)`.
- [x] **Namespaces** (`namespace` / `source_namespace`) so source and Athena
  `concept_id`s cannot collide on the PK `(namespace, concept_id, model_version)`.
- [x] **Selectable pooling** (`--pooling cls|mean`), folded into `model_version`.
- [x] **Incremental change-detection:** CSV fingerprints + a weight-hash cache to
  skip unchanged inputs and avoid re-hashing the model every run.
- [x] **Reporting / maintenance commands:** `export` (sharded by
  `model_version` / `vocabulary_id`), `status`, `coverage`, `model-registry`,
  `cleanup`.

---

## Remaining / future

- [ ] **`query` subcommand** — semantic search over the store. Lance ships a
  native ANN vector index (IVF/PQ), so build this on Lance's index rather than a
  brute-force cosine top-K scan in DuckDB.
- [ ] **Lance scale validation** (see ADR "Caveats"): compaction time + transient
  disk at ~12M rows, a concrete version-retention policy, and export-to-parquet
  throughput at scale. (Compaction *memory* was measured a non-issue — sub-linear,
  ~1–1.5 GB up to 1.5M rows.)
- [ ] **Streaming / chunked CSV reads** for very large Athena files (>10M rows).
- [ ] **Multi-GPU batching** (e.g. `DataParallel`).
- [ ] **`--backend postgres`** via a `store_pg.py` — listed for completeness, but
  the no-separate-RDBMS constraint (see `AGENTS.md`) makes this unlikely.
