# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

---

## What this project is

`gpu-embedder` is a focused CLI tool that batch-embeds OHDSI Athena
`CONCEPT.csv` files using **SapBERT** (FP32, CUDA GPU) and persists vectors to
a local **Lance** store (`embeddings.lance` by default; a `.duckdb` path selects
the DuckDB backend instead). It is intentionally single-purpose: no pipeline
orchestration, no LLM calls, no network I/O beyond the initial HuggingFace
model download.

Key invariants:
- **Idempotent by default.** `(namespace, concept_id, model_version)` is the
  unique key; rows that already exist are silently skipped unless `--force` is
  passed. `namespace` defaults to `athena` for Athena standard concepts; source
  datasets pass a distinct namespace so their (possibly colliding) `concept_id`s
  stay separate.
- **FP32 only.** No fp16/bf16 quantization. The `embed.py` module must never
  call `.half()` or `.to(torch.bfloat16)` on the model or tensors.
- **One embedding model at a time.** `model_version` is a SHA-256 digest of the
  model weights file(s), computed at startup. Never mix versions in a single
  DuckDB file unless intentional.

---

## Setup & commands

```bash
uv sync --group dev       # install project + dev deps (Python ≥ 3.12, < 3.14)
uv run pytest             # run tests (unit + skips GPU tests on CPU-only machines)
uv run pytest tests/unit/ # fast unit tests only
uv run ruff check src tests
uv run gpu-embed --help   # CLI entry point
```

> **CUDA torch requires Python ≤ 3.13.** PyTorch CUDA wheels are not published
> for Python 3.14+. `pyproject.toml` constrains `requires-python = ">=3.12,<3.14"`
> so uv will not create a 3.14 venv. If one already exists, recreate it:
> `uv venv --python 3.13 && uv sync --group dev`.
> `[tool.uv.sources]` in `pyproject.toml` routes `torch` to the CUDA 13.0 index
> on Linux/Windows automatically, so `uv sync` is sufficient. For ad-hoc installs:
> `uv pip install torch --torch-backend=auto` (auto-detects driver version).

---

## Project structure

```
src/gpu_embedder/
├── cli.py        # Typer app; `embed` + `cpt4` subcommands; thin — delegates to other modules
├── config.py     # EmbedConfig (Pydantic BaseSettings); env prefix GPU_EMBED_; loads .env
├── models.py     # ConceptRow / EmbeddedRow (slots dataclasses), DuckDB DDL constants, FilterSpec
├── ingest.py     # read_csv() → filter_rows(); pure, no I/O side effects
├── embed.py      # load_model(), compute_model_version(), embed_batch()
└── store.py      # open_db(), ensure_schema(), classify_rows_requiring_embedding(), upsert_rows()
```

`cli.py` is the only module allowed to call all others. No other module imports
from `cli.py`.

---

## Code conventions

### General

- **Python ≥ 3.12.** Use `pathlib.Path`, `match` statements, `|` unions.
- Line length **100**. Ruff with `py312` target.
- All public functions have type annotations. No `Any` except in explicit shim
  code (mark with `# type: ignore[misc]` and a comment explaining why).
- **Typed dataclasses are the contract between modules.** `ingest.py` returns
  `list[ConceptRow]`; `embed.py` consumes it. No raw dicts across boundaries.
  `ConceptRow`/`EmbeddedRow` are `@dataclass(slots=True)` (not Pydantic) because
  millions are built per run and Pydantic instantiation dominated CSV load. The
  light coercion the old validators did (`concept_id`→int, empty/`"NULL"`→None)
  now lives in the DuckDB scan SELECT (`ingest._coerced_scan_columns`); keep
  validation/coercion there, not in per-row Python. `EmbedConfig` stays Pydantic
  `BaseSettings` (it's config, not a hot path).
- **`.env` is the single source of truth for local config.** `EmbedConfig`
  must call `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`
  so that any `.env` present is loaded automatically. Never hard-code paths or
  keys anywhere in source. `.env` is gitignored; `.env.example` is the committed
  template and must stay in sync with `EmbedConfig`.

### CSV / Athena

- The default vocabulary directory is `GPU_EMBED_VOCAB_DIR` (default
  `athena_vocab/`). When no explicit `CSV_PATH` arguments are given, the CLI
  reads `<vocab_dir>/CONCEPT.csv`.
- **DuckDB is the default CSV engine.** Read/filter Athena TSVs through DuckDB
  and coerce types in the scan SELECT so large files are narrowed and typed
  before any Python-level row objects are built.
- `ingest_engine` can be set to `python` for a pure-Python fallback, but the
  default path should remain DuckDB.
- Athena CSVs use **tab-separated values** (`\t`) with a header row. Always
  open with `sep="\t"` and `dtype=str` (then coerce). Do not assume column
  order.
- Treat `standard_concept` and `invalid_reason` as **nullable strings** — empty
  cell and literal `"NULL"` both map to Python `None`.
- Filtering is **additive AND** across different column types but **OR** within
  a single column type (e.g. `--vocabulary-id LOINC --vocabulary-id SNOMED`
  keeps rows where vocabulary is LOINC _or_ SNOMED).
- The shorthand `--invalid-reason valid` must map to `NULL / ""` (both).
- **`--vocabulary-id` defaults to a curated highest-yield set, not "all".** When
  no `--vocabulary-id` is given, `cli.py` materializes `DEFAULT_VOCABULARY_IDS`
  (in `models.py`) into the `FilterSpec` so the filter, fingerprint, and
  `filter_spec_hash` all stay consistent. The reserved sentinel
  `--vocabulary-id all` clears the filter to embed every vocabulary. Keep the
  default as exact, case-sensitive Athena `vocabulary_id` strings.

### CPT-4 population

- Athena does not include CPT-4 concept names in the raw CSV download; they
  must be populated via the bundled `cpt4.jar` using a UMLS license key.
- `UMLS_API_KEY` and `CPT4_JAR` live in `.env` and are **not** prefixed with
  `GPU_EMBED_` because they are not `EmbedConfig` fields — they are consumed
  directly by the `cpt4` subcommand in `cli.py`.
- The `cpt4` subcommand in `cli.py` invokes Java as a subprocess:
  ```python
  subprocess.run(
      ["java", f"-Dumls-apikey={api_key}", "-jar", str(jar_path), "5"],
      check=True,
      cwd=vocab_dir,
  )
  ```
- Never store the UMLS API key in logs, exceptions, or error messages.
  Truncate or redact before surfacing to the user.
- Java resolution order for the `cpt4` subcommand is: `PATH` → `JAVA_HOME` →
  common Windows install locations (Adoptium/Oracle/Microsoft/JetBrains). Emit
  a clear error if no Java executable can be found.

### Embedding

- Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (768-dim) is the
  configurable default (`--model` / `GPU_EMBED_MODEL`); any 768-dim biomedical
  encoder is accepted. See the README's "Choosing an embedding model" for
  alternatives such as BioLORD-2023. The invariants below (FP32, 768-dim store
  column, `.cpu().numpy()` collection) hold regardless of which model is
  selected. Pooling is selectable rather than fixed (see below).
- Pin the revision via `GPU_EMBED_MODEL_REVISION` (commit hash, branch, or tag)
  so downloads are reproducible. Pass as `revision=` to both
  `AutoModel.from_pretrained` and `AutoTokenizer.from_pretrained`. `None` uses
  the upstream default branch (not recommended for production).
- Always run in **FP32** (`model.float()`). Never call `.half()`.
- Pool strategy: **selectable** (`--pooling` / `GPU_EMBED_POOLING`), default
  `cls` (CLS token, `last_hidden_state[:, 0, :]`); `mean` is mask-aware mean
  pooling for sentence-transformers models (e.g. BioLORD-2023). Output is
  L2-normalized either way. **Pooling is part of `model_version`** — non-default
  pooling is folded into the digest (see the run-variant note below).
- Tokenize with `max_length=128`, `truncation=True`, `padding=True`.
- Process in batches (`batch_size` from config). Move tensors to device; do not
  accumulate GPU tensors across batches (call `.cpu().numpy()` before
  collecting).
- `compute_model_version()` must hash the actual weights on disk (not the model
  name string) — use SHA-256 over the `pytorch_model.bin` or `model.safetensors`
  file. This should be stable across runs for the same checkpoint.
- **Run variants belong in the `model_version`, not a per-row column.** Anything
  that changes the embeddings without changing the weights file — quantization,
  precision, **or pooling strategy** — must be folded into the digest, or
  otherwise-identical variants collide on the primary key. For example, a `mean`
  run reusing the `cls` `model_version` would be seen as "already embedded" and
  silently skipped. `compute_model_version(..., precision=, quantization_scheme=,
  pooling=)` folds these in *only when non-default* (`fp32`/`none`/`cls` returns
  the bare weights hash, so existing stores are unaffected). When you add a new
  run variant, thread it through all three call sites —
  `compute_model_version`, `upsert_model_registry`, **and** the
  `model_version_cache` key (`get_cached_model_version` /
  `upsert_model_version_cache`, whose PK includes the variant) — and keep
  `model_registry` as the human-readable provenance. Pooling is wired this way;
  precision/quantization default to fp32/none (the project is FP32-only).

### DuckDB

- Schema DDL lives in `models.py` as a module-level constant string
  `SCHEMA_DDL`. `store.py` calls `conn.execute(SCHEMA_DDL)` with `IF NOT
  EXISTS`; never drop or alter existing tables.
- The embedding column is `FLOAT[768]` (DuckDB array type). `EmbeddedRow.embedding`
  stays a Python `list[float]` across module boundaries — do not serialize to
  JSON or bytes.
- **Source provenance round-trips as nullable columns, not a hash reversal.**
  `concept_embeddings` carries nullable `source_id` + `mapping_wave` (NULL for
  Athena; populated for `--source-parquet` runs). For a source row the BIGINT
  `concept_id` is only a one-way hash of `source_id`
  (`ingest._stable_source_concept_id`), so the original key must be stored
  explicitly to rejoin concept-mapper's `source_concepts` on
  `(mapping_wave, source_id)`. These two columns are **not** in the primary key
  (the hashed `concept_id` already disambiguates within a namespace) and must
  stay threaded through `ConceptRow`, `SCHEMA_DDL`, `_EMBEDDING_COLUMNS`,
  `_embedded_rows_to_arrow()`, both parquet view projections, the shard writer,
  and the `export` COPY. `read_source_parquet` reads `mapping_wave` defensively
  (NULL when a legacy source parquet lacks it).
- **Moving bulk Python data into DuckDB always goes through Arrow.** Build a
  columnar `pyarrow.Table` and `conn.register(...)` it, then `INSERT ... SELECT`
  or JOIN against the registered relation. This is the *only* approved path for
  anything that scales with row count — embedding writes
  (`_embedded_rows_to_arrow()` → `INSERT OR REPLACE ... SELECT`, ~100× faster
  than the old `executemany` on the wide `FLOAT[768]` column) **and** the
  change-detection candidate set in `classify_rows_requiring_embedding()`
  (register `(namespace, concept_id, embed_text)`, then LEFT JOIN). Two
  anti-patterns are **banned for large inputs** — both have bitten this project:
  - `conn.executemany("INSERT ... VALUES (?)", rows)` — per-row Python→DuckDB
    binding; pathological for the embedding column.
  - `... unnest(?::T[])` with a big Python list bound as the array param —
    DuckDB materialises the whole list as one value ~quadratically (tens of
    seconds at a few hundred thousand rows; effectively hangs at millions).
    `unnest(?::T[])` is fine only for *small, bounded* lists (e.g. a handful of
    filter values); never for a per-concept column.
- `classify_rows_requiring_embedding()` is the single entry point for deciding
  what to embed: it returns the rows needing work plus `(new, changed, unchanged)`
  counts, skipping rows already embedded for the model version *and* re-embedding
  rows whose `embed_text` changed. (The older `get_existing_ids` /
  `filter_unembedded_rows` helpers were removed — do not reintroduce parallel
  variants.)
- The store context is tracked in a `weakref.WeakKeyDictionary` keyed by the
  connection object — not `id(conn)` (which leaks and can alias a closed
  connection after the id is recycled).
- DuckDB connection is opened **once** per CLI invocation and passed down;
  modules do not open their own connections.

### Testing

- `pytest` with `asyncio_mode = "auto"` (future-proofing). Coverage gate **80%**
  (`fail_under` in `pyproject.toml`).
- Unit tests live in `tests/unit/` and must run on CPU-only machines with no
  HuggingFace model download. Inject a `FakeEmbedder` that returns
  deterministic random vectors of shape `(n, 768)`.
- GPU/integration tests live in `tests/integration/` and are marked
  `@pytest.mark.gpu`. They are auto-skipped when `torch.cuda.is_available()`
  is false.
- Never hit the network in tests. Monkeypatch `transformers.AutoModel.from_pretrained`
  and `transformers.AutoTokenizer.from_pretrained` in unit tests.
- Fixture CSVs live in `tests/fixtures/` and are minimal TSV files (5-10 rows).

---

## What NOT to do

- **Do not commit `.env`** — only `.env.example` belongs in source control.
  Keep `.env.example` in sync with every new `EmbedConfig` field.
- **Do not log or print `UMLS_API_KEY`** — truncate or redact in any error
  output.
- **Do not quantize** the model (`fp16`, `int8`, `bnb`, etc.).
- **Do not add a pipeline orchestration layer** (no Prefect, Airflow, etc.).
- **Do not read or write anything other than local files and the local store
  (Lance or DuckDB).** No Snowflake, no S3, no HTTP in production code paths.
- **Do not embed SQL in Python f-strings.** Any non-trivial SQL goes in a
  `.sql` file under `src/gpu_embedder/sql/` and is loaded via a helper.
- **Do not pass lists directly as bind parameters to DuckDB `IN` clauses.**
  For a *small, bounded* list, `IN (SELECT unnest(?::BIGINT[]))` is fine. For
  anything that scales with row count, register an Arrow table and JOIN/semi-join
  against it — never bind a large list as an `unnest(?::T[])` param (quadratic;
  see the bulk-data rule under "DuckDB").
- **Do not silently swallow embedding errors.** Log and re-raise; a partial
  batch should not produce a partial write.
- **Do not use Parquet as the primary write store.** A Parquet-backed backend
  was tried and abandoned after production use — it caused severe write
  amplification (full shard rewrite per checkpoint), no native upsert/PK
  enforcement, stale-view bugs, and startup cost proportional to total table
  size. See `docs/adr_parquet_store_rejected.md` for the full post-mortem.
  Parquet is for `gpu-embed export` (Snowflake handoff) only.
- **Live write backends:** `.lance` (**default**; ACID, cross-process readers,
  O(changes) `merge_insert` upserts — the adopted answer to the ACID +
  concurrency requirement, see `docs/adr_lance_store_proposal.md`) and `.duckdb`
  (native single-writer table; fast, but an exclusive file lock means no
  cross-process concurrency). `pylance` is a base dependency. The lance backend's
  fingerprint and weight-hash caches persist in a `<store>.lance/_meta/meta.duckdb`
  sidecar (its main query connection is in-memory) — route them through
  `store._meta_conn`, which returns the main connection for `.duckdb` and `None`
  for parquet (still a no-op there). Both keep `gpu-embed export` → plain parquet
  as the unchanged Snowflake contract.

---

## Adding a new filter column

1. Add the column name to `FilterSpec` in `models.py`.
2. Add the corresponding `--filter-column` option in `cli.py`
   (repeatable, `list[str]`).
3. Extend `filter_rows()` in `ingest.py` to apply the new predicate.
4. Add a unit test in `tests/unit/test_ingest.py`.

---

## Adding a new output backend

The only write abstraction is `store.py`. Backends are selected by **store-path
dispatch** in `store._resolve_paths` (`.duckdb` → table, `.lance` → Lance dir,
other dir → parquet), not a separate `--backend` flag. To add another backend:
1. Add a branch to `_resolve_paths` / `_StoreContext` and implement the same
   interface within `store.py` (as duckdb/parquet/lance do): `open_db`,
   `ensure_schema`, `refresh_view`, `classify_rows_requiring_embedding`,
   `upsert_rows`, `count_rows`, `delete_embeddings`, and the registry/meta
   functions. File backends expose `concept_embeddings` as a DuckDB view so all
   readers and `export` work unchanged.
2. Keep it opt-in (a new path suffix) so the default `.lance` behaviour is
   untouched; gate any heavy dependency behind an optional extra + lazy import.
3. Do not modify `embed.py` or `ingest.py`.

---

## Git workflow

- Feature branches off `main`; do not push directly to `main`.
- Commit messages: imperative mood, explain the why.
- Do not include model identifiers or assistant attribution in commits or code
  comments.
