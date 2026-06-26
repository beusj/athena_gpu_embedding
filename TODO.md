# TODO — gpu-embedder implementation plan

Phases are ordered by dependency. Complete each phase before starting the next.
Tick items off as they are merged.

---

## Phase 1 — Project scaffolding

- [ ] Create `pyproject.toml` with `uv` project metadata
  - Python `>=3.12`
  - Dependencies: `duckdb`, `polars` or `pandas`, `typer`, `pydantic`,
    `pydantic-settings`, `python-dotenv`, `transformers`, `torch`, `tqdm`
  - Dev group: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`
  - Entry point: `gpu-embed = "gpu_embedder.cli:app"`
  - Coverage gate `fail_under = 80` in `[tool.pytest.ini_options]`
- [ ] Create `.env.example` with all `EmbedConfig` fields documented
  (committed); create `.gitignore` excluding `.env`, `athena_vocab/`, `*.duckdb`
- [ ] Create `src/gpu_embedder/__init__.py` (version string only)
- [ ] Create `tests/__init__.py`, `tests/unit/__init__.py`,
  `tests/integration/__init__.py`
- [ ] Create `tests/conftest.py` with `gpu` mark registration and
  auto-skip logic (`torch.cuda.is_available()`)
- [ ] Create `tests/fixtures/CONCEPT_mini.tsv` — 10-row Athena-format TSV
  covering at least 3 vocabulary IDs, 2 domain IDs, standard+non-standard,
  valid+invalid rows
- [ ] Verify `uv sync --group dev` installs cleanly
- [ ] Verify `uv run ruff check src tests` passes on empty stubs

---

## Phase 2 — Config & models

- [ ] `src/gpu_embedder/config.py`
  - `EmbedConfig(BaseSettings)`: `vocab_dir`, `db`, `batch_size`, `model`,
    `model_revision`, `device`, `force`, `text_fields`, `separator`, `max_length`
  - `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`
  - Env prefix `GPU_EMBED_` for all fields except `UMLS_API_KEY` / `CPT4_JAR`
    (those are read directly in the `cpt4` subcommand, not via `EmbedConfig`)
  - `device` auto-detection: prefer `cuda` → `mps` → `cpu`
  - `vocab_dir` defaults to `Path("athena_vocab")`
- [ ] `src/gpu_embedder/models.py`
  - `ConceptRow`: Pydantic model mirroring Athena columns
    (`concept_id: int`, `concept_name: str`, `domain_id: str | None`, …)
  - `FilterSpec`: dataclass/model holding per-column include-lists
  - `SCHEMA_DDL`: module-level string with `CREATE TABLE IF NOT EXISTS
    concept_embeddings (…)` — see README for column list
- [ ] Unit tests for `ConceptRow` coercion (empty → `None` for nullable fields)

---

## Phase 3 — CSV ingest & filtering

- [ ] `src/gpu_embedder/ingest.py`
  - `read_csv(path: Path) -> list[ConceptRow]` — TSV, `dtype=str`,
    coerce types, map empty/`"NULL"` to `None`
  - `filter_rows(rows, spec: FilterSpec) -> list[ConceptRow]`
    - OR within each column's include-list, AND across columns
    - `invalid_reason="valid"` shorthand → keep rows where field is `None`
  - Both functions are **pure** (no I/O side effects in `filter_rows`)
- [ ] Unit tests (`tests/unit/test_ingest.py`)
  - Round-trip the fixture TSV
  - Filter: single vocabulary, multiple vocabularies (OR), combined with
    domain (AND), `--invalid-reason valid` shorthand
  - Empty result (all rows filtered out) does not raise
- [ ] `read_csv` handles multi-file input (called once per path, results
  concatenated by caller)

---

## Phase 4 — SapBERT embedding

- [ ] `src/gpu_embedder/embed.py`
  - `compute_model_version(model_or_path: str | Path) -> str`
    - Resolve HF cache path → locate `model.safetensors` or
      `pytorch_model.bin` → SHA-256 digest (first 16 hex chars for
      readability, full 64 chars stored)
  - `load_model(model_id: str, device: str) -> tuple[model, tokenizer]`
    - `AutoModel.from_pretrained(...).float().to(device).eval()`
    - Never call `.half()` or set `torch_dtype`
  - `embed_batch(texts: list[str], model, tokenizer, device: str) -> np.ndarray`
    - Tokenize: `max_length=128`, `truncation=True`, `padding=True`
    - Forward with `torch.no_grad()`
    - CLS pooling: `last_hidden_state[:, 0, :]`
    - L2-normalize each vector
    - Return `(N, 768)` float32 numpy array; move off GPU before returning
  - `embed_all(rows, model, tokenizer, device, batch_size, text_fields,
    separator) -> list[tuple[ConceptRow, list[float]]]`
    - Iterates in batches, shows `tqdm` progress bar, yields `(row, vector)`
    - On exception within a batch: log error, re-raise (no partial writes)
- [ ] Unit tests (`tests/unit/test_embed.py`)
  - `compute_model_version` returns a 64-char hex string
  - `embed_batch` with a `FakeModel` / `FakeTokenizer` returns correct shape
    `(N, 768)`, float32, L2-normalized
  - `embed_all` respects `text_fields` and `separator`
  - GPU tests (`@pytest.mark.gpu`): load real SapBERT, embed 10 concepts,
    assert shape and norm ≈ 1.0

---

## Phase 5 — DuckDB store

- [ ] `src/gpu_embedder/store.py`
  - `open_db(path: Path) -> duckdb.DuckDBPyConnection`
  - `ensure_schema(conn)` — runs `SCHEMA_DDL` from `models.py`
  - `get_existing_ids(conn, model_version: str) -> set[int]`
    — `SELECT concept_id FROM concept_embeddings WHERE model_version = ?`
  - `upsert_rows(conn, rows: list[EmbeddedRow])` — single `executemany`
    with `INSERT OR REPLACE`; `EmbeddedRow` = `ConceptRow` + `embedding` +
    `embed_text` + `model_version` + `embedded_at`
  - `count_rows(conn, model_version: str) -> int` — for progress reporting
- [ ] Unit tests (`tests/unit/test_store.py`)
  - Use an in-memory DuckDB (`":memory:"`)
  - `ensure_schema` is idempotent (call twice, no error)
  - `upsert_rows` then `get_existing_ids` returns the inserted IDs
  - Re-upsert same row with `OR REPLACE` does not duplicate
  - `get_existing_ids` is scoped to `model_version` (different version →
    empty set)

---

## Phase 6 — CLI wiring

- [ ] `src/gpu_embedder/cli.py`
  - `app = typer.Typer()`; entry point `gpu-embed`
  - **`embed` subcommand** (default when no subcommand given):
    - Accepts all `embed` flags documented in README
    - Repeatable multi-value flags via `list[str]` + `typer.Option`
    - When no `CSV_PATH` args given, defaults to `<vocab_dir>/CONCEPT.csv`
    - Flow:
      1. Build `EmbedConfig` (merge CLI flags + env)
      2. Open DuckDB, `ensure_schema`
      3. `get_existing_ids` (skip set, empty if `--force`)
      4. For each CSV path: `read_csv` → `filter_rows`
      5. Remove skip-set from filtered rows (unless `--force`)
      6. `load_model` + `compute_model_version` (once, before batching)
      7. `embed_all` → `upsert_rows` in batches
      8. Print summary: N input, N skipped, N embedded, N errors
  - **`cpt4` subcommand**:
    - Reads `UMLS_API_KEY` and `CPT4_JAR` from env / `.env` (not `EmbedConfig`)
    - Accepts `--vocab-dir`, `--jar`, `--api-key` overrides
    - Checks Java is on `PATH`; emits clear error if not
    - Checks `cpt4.jar` exists; emits clear error if not
    - Checks `UMLS_API_KEY` is non-empty; emits clear error (without printing
      the key) if not
    - Runs `java -Dumls-apikey=<key> -jar <jar> <vocab_dir>` via
      `subprocess.run(..., check=True)`
    - Streams stdout/stderr in real time; exits non-zero on Java failure
    - Never includes the API key in log messages or exception text
  - `typer.echo` for user-facing output; `logging` for debug detail
- [ ] Smoke test: `uv run gpu-embed --help` and `uv run gpu-embed embed --help`
  and `uv run gpu-embed cpt4 --help` all exit 0
- [ ] Integration test (`tests/integration/test_cli.py`)
  - Use `typer.testing.CliRunner` + fixture TSV + in-memory (or tmp) DuckDB
  - First run: all rows embedded, stored
  - Second run (no `--force`): 0 rows re-embedded
  - Second run with `--force`: all rows re-embedded (timestamps updated)
  - `--vocabulary-id` filter: only matching rows in DB
  - `cpt4` subcommand with missing `UMLS_API_KEY` exits non-zero with a clear
    message (does not leak the key)
  - `cpt4` subcommand with missing `cpt4.jar` exits non-zero with a clear
    message

---

## Phase 7 — Polish & CI

- [ ] `ruff.toml` or `[tool.ruff]` in `pyproject.toml`: `line-length = 100`,
  `target-version = "py312"`, `select = ["E","W","F","I","UP"]`
- [ ] `mypy` config: `strict = true`, ignore missing stubs for `duckdb` and
  `transformers`
- [ ] Add `--version` flag printing package version from `__init__.py`
- [ ] Logging: structured output at `INFO` by default; `DEBUG` with
  `--verbose` flag
- [ ] Progress bar (`tqdm`) reports both batch progress and total concept count
- [ ] README accuracy pass: verify all examples work against the final CLI
- [ ] Confirm `uv run pytest --cov=gpu_embedder --cov-fail-under=80` passes

---

## Stretch / future

- [ ] `--backend postgres` option via `store_pg.py` (same interface as
  `store.py`)
- [ ] `--output-format parquet` to write vectors to Parquet instead of DuckDB
- [ ] Streaming CSV reads (chunked) for very large Athena files (>10M rows)
- [ ] Multi-GPU batching with `DataParallel`
- [ ] A `query` subcommand: `gpu-embed query "acute MI"` → cosine top-K from
  DuckDB
