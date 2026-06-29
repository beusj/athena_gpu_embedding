# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

---

## What this project is

`gpu-embedder` is a focused CLI tool that batch-embeds OHDSI Athena
`CONCEPT.csv` files using **SapBERT** (FP32, CUDA GPU) and persists vectors to
a local **DuckDB** database. It is intentionally single-purpose: no pipeline
orchestration, no LLM calls, no network I/O beyond the initial HuggingFace
model download.

Key invariants:
- **Idempotent by default.** `(concept_id, model_version)` is the unique key;
  rows that already exist are silently skipped unless `--force` is passed.
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
├── models.py     # ConceptRow (Pydantic), DuckDB DDL constant, FilterSpec
├── ingest.py     # read_csv() → filter_rows(); pure, no I/O side effects
├── embed.py      # load_model(), compute_model_version(), embed_batch()
└── store.py      # open_db(), ensure_schema(), get_existing_ids(), upsert_rows()
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
- **Pydantic models are the contract between modules.** `ingest.py` returns
  `list[ConceptRow]`; `embed.py` consumes it. No raw dicts across boundaries.
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
  before Pydantic validation so large files are narrowed early.
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

- Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (768-dim).
- Pin the revision via `GPU_EMBED_MODEL_REVISION` (commit hash, branch, or tag)
  so downloads are reproducible. Pass as `revision=` to both
  `AutoModel.from_pretrained` and `AutoTokenizer.from_pretrained`. `None` uses
  the upstream default branch (not recommended for production).
- Always run in **FP32** (`model.float()`). Never call `.half()`.
- Pool strategy: **CLS token** (`last_hidden_state[:, 0, :]`), L2-normalized.
- Tokenize with `max_length=128`, `truncation=True`, `padding=True`.
- Process in batches (`batch_size` from config). Move tensors to device; do not
  accumulate GPU tensors across batches (call `.cpu().numpy()` before
  collecting).
- `compute_model_version()` must hash the actual weights on disk (not the model
  name string) — use SHA-256 over the `pytorch_model.bin` or `model.safetensors`
  file. This should be stable across runs for the same checkpoint.

### DuckDB

- Schema DDL lives in `models.py` as a module-level constant string
  `SCHEMA_DDL`. `store.py` calls `conn.execute(SCHEMA_DDL)` with `IF NOT
  EXISTS`; never drop or alter existing tables.
- The embedding column is `FLOAT[768]` (DuckDB array type). Insert as a Python
  `list[float]` — do not serialize to JSON or bytes.
- `get_existing_ids(conn, model_version)` returns a `set[int]` of `concept_id`s
  that already have an embedding for that model version. Check before embedding,
  not after.
- All writes use a single `executemany` / `INSERT OR REPLACE` per batch, not
  row-by-row.
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
- **Do not read or write anything other than local files and DuckDB.** No
  Snowflake, no S3, no HTTP in production code paths.
- **Do not embed SQL in Python f-strings.** Any non-trivial SQL goes in a
  `.sql` file under `src/gpu_embedder/sql/` and is loaded via a helper.
- **Do not pass lists directly as bind parameters to DuckDB `IN` clauses.**
  Use `IN (SELECT unnest(?::BIGINT[]))` or build the filter in Python before
  the query.
- **Do not silently swallow embedding errors.** Log and re-raise; a partial
  batch should not produce a partial write.
- **Do not use Parquet as the primary write store.** A Parquet-backed backend
  was tried and abandoned after production use — it caused severe write
  amplification (full shard rewrite per checkpoint), no native upsert/PK
  enforcement, stale-view bugs, and startup cost proportional to total table
  size. See `docs/adr_parquet_store_rejected.md` for the full post-mortem.
  The `.duckdb` native table is the only supported live write backend.
  Parquet is for `gpu-embed export` (Snowflake handoff) only.

---

## Adding a new filter column

1. Add the column name to `FilterSpec` in `models.py`.
2. Add the corresponding `--filter-column` option in `cli.py`
   (repeatable, `list[str]`).
3. Extend `filter_rows()` in `ingest.py` to apply the new predicate.
4. Add a unit test in `tests/unit/test_ingest.py`.

---

## Adding a new output backend

The only write abstraction is `store.py`. To add PostgreSQL or another backend:
1. Create `store_pg.py` implementing the same interface:
   `open_db`, `ensure_schema`, `get_existing_ids`, `upsert_rows`.
2. Select via a `--backend` CLI flag (default `duckdb`).
3. Do not modify `embed.py` or `ingest.py`.

---

## Git workflow

- Feature branches off `main`; do not push directly to `main`.
- Commit messages: imperative mood, explain the why.
- Do not include model identifiers or assistant attribution in commits or code
  comments.
