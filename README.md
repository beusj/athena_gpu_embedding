# gpu-embedder

Batch-embed OHDSI Athena concept CSVs with a biomedical embedding model
(SapBERT by default; configurable — see
[Choosing an embedding model](#choosing-an-embedding-model)) in FP32 on a GPU,
persisting the vectors to a local Lance store (`embeddings.lance` by default; a
`.duckdb` path selects the native DuckDB table backend instead).
Already-embedded concepts are skipped unless `--force` is passed, so runs are
safe to restart or extend incrementally.

---

## Runbooks

- `docs/runbooks/aws_interaction_guide.md` — AWS Batch and infrastructure interaction guide
- `docs/runbooks/s3_to_snowflake_load.md` — export parquet, copy to S3, and load to Snowflake

---

## What it does

1. **Reads** `CONCEPT.csv` from the `athena_vocab/` directory (configurable via
   `GPU_EMBED_VOCAB_DIR` in `.env` or `--vocab-dir` on the CLI), or from
   explicit paths passed as arguments.
2. **Filters** rows by any combination of Athena columns
   (`vocabulary_id`, `domain_id`, `concept_class_id`, `standard_concept`,
   `invalid_reason`) so you only embed what you need. CSV scan + filter are
   pushed through **DuckDB by default** before Pydantic validation.
3. **Embeds** the filtered concept names with a biomedical embedding model —
   [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
   by default, configurable via `--model` (see
   [Choosing an embedding model](#choosing-an-embedding-model)) — running in FP32
   on a CUDA GPU (falls back to CPU when no GPU is available, but will be slow).
4. **Writes** each concept's vector plus its metadata into a local Lance store
  (`embeddings.lance` by default). A `.duckdb` path instead selects the native
  DuckDB table backend; a directory path selects the parquet-sharded store, a
  migration/export artifact rather than a live store (see
  [Storage model and migration](#storage-model-and-migration)).
5. **Skips** rows whose `(namespace, concept_id, model_version)` key already
   exists in the store. Pass `--force` to re-embed unconditionally.

---

## Project layout

```
gpu_embedding/
├── .env.example            # config template — copy to .env and fill in
├── .env                    # gitignored; holds secrets + local overrides
├── pyproject.toml          # uv project; Python ≥ 3.12
├── athena_vocab/           # Athena download dir (gitignored); place CSVs here
│   ├── CONCEPT.csv
│   ├── cpt4.jar            # Athena-provided; required for CPT-4 population
│   └── ...                 # other Athena vocab files
├── src/
│   └── gpu_embedder/
│       ├── cli.py          # Typer app: `embed`, `export`, `status`, `coverage`, `cpt4`
│       ├── config.py       # Settings (Pydantic BaseSettings + env)
│       ├── models.py       # Pydantic row models + contracts
│       ├── ingest.py       # DuckDB-backed CSV scan + filter → ConceptRow records
│       ├── embed.py        # FP32 tokenize + forward pass (CLS-pooled, L2-normalized)
│       └── store.py        # Lance (default) / DuckDB / parquet store backends + query view
└── tests/
    ├── unit/
    └── integration/
```

---

## Setup

```bash
# 1. Copy and edit the config template
cp .env.example .env
# edit .env — set GPU_EMBED_VOCAB_DIR, GPU_EMBED_DEVICE, etc.

# 2. Install project + dev deps (requires uv ≥ 0.4). Pick a torch backend:
uv sync --group dev --extra gpu   # CUDA 13.0 (Linux/Windows); CPU/MPS on macOS
#   or, for a GPU-free environment (CI, laptops, running the tests):
uv sync --group dev --extra cpu   # CPU-only torch wheels — small and fast

# 3. Place Athena vocabulary files in athena_vocab/
#    (or set GPU_EMBED_VOCAB_DIR in .env to your download location)
```

`cpu` and `gpu` are mutually exclusive extras (enforced by `[tool.uv].conflicts`);
choose exactly one. The `cpu` extra is the right choice for running the test
suite, which mocks the model and never needs CUDA.

The embedding model is downloaded from Hugging Face on first run and cached
locally via the normal `~/.cache/huggingface` path.

DuckDB is also the default engine for reading and filtering Athena TSVs, so
large source files are narrowed before they are validated or embedded.

If you need a compatibility fallback or want to debug parsing differences,
use `--ingest-engine python` to switch the ingest path.

### GPU setup (CUDA)

> **Python version constraint:** PyTorch CUDA wheels are not published for
> Python 3.14+. `pyproject.toml` already constrains `requires-python = ">=3.12,<3.14"`
> so `uv` will not create a 3.14 environment. If you already have a 3.14 venv,
> recreate it:
> ```bash
> uv venv --python 3.13
> uv sync --group dev --extra gpu
> ```

`pyproject.toml` routes `torch` to a backend-specific index via the `gpu` / `cpu`
extras (see `[tool.uv.sources]`): `uv sync --extra gpu` installs the CUDA 13.0
build on Linux/Windows (CPU/MPS on macOS), while `uv sync --extra cpu` installs
small CPU-only wheels — ideal for CI and for running the test suite without a GPU.

To install or upgrade torch independently (e.g. into a fresh venv or when
testing a different backend), use uv's built-in `--torch-backend` flag:

```bash
# auto-detects your CUDA driver version and picks the right build
uv pip install torch --torch-backend=auto

# or pin explicitly (cu118 / cu126 / cu128 / cu130 / rocm6.4 / cpu)
uv pip install torch --torch-backend=cu130
```

Verify GPU is detected before running a full embed:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Expected output (example): `2.6.0+cu130 True NVIDIA GeForce RTX 5060 Ti`

If `cuda.is_available()` is still `False` after syncing:
- Confirm the driver supports the installed CUDA toolkit (`nvidia-smi` shows
  the maximum supported CUDA version in the top-right corner).
- Run `uv pip show torch` and verify `Location:` is inside your active venv.
- `--torch-backend=auto` resolves the right backend for your installed driver
  automatically; use it to sanity-check what uv would pick.

### Testing without a GPU (CI, sandboxes, Claude Code on the web)

**The unit tests never need a GPU or CUDA** — they mock the model, so there is no
real forward pass. The `cpu` extra exists specifically for these GPU-free
environments. This is the recommended setup for running the suite.

**Open-network CI or a local machine** — use the `cpu` extra, which pulls small
CPU-only wheels from the PyTorch index:

```bash
uv sync --group dev --extra cpu
uv run pytest
```

**Network-restricted sandboxes** (e.g. Claude Code on the web, some CI runners)
often allow `pypi.org` but **block `download.pytorch.org`**. There the `cpu`
extra can't fetch its wheels, so install a CPU-capable torch straight from PyPI
instead (larger, bundles CUDA libs, but runs fine on CPU):

```bash
uv venv
uv pip install "torch>=2.3"          # CPU-capable build from PyPI
uv pip install --no-deps -e .        # project without re-resolving the torch index
uv pip install duckdb typer pydantic pydantic-settings python-dotenv \
  transformers tqdm numpy pyarrow pytest pytest-cov pytest-asyncio ruff mypy
uv run pytest
```

For **Claude Code on the web**, this is automated: `.claude/hooks/session-start.sh`
runs the PyPI-based setup above on session start so `uv run pytest` works
out of the box (see `.claude/settings.json`). The hook only runs in the remote
environment (`CLAUDE_CODE_REMOTE=true`); local development uses the `uv sync`
commands above.

---

## Choosing an embedding model

SapBERT is the default, but any Hugging Face biomedical encoder that produces a
768-dimensional vector can be selected with `--model` (or `GPU_EMBED_MODEL`).
Two strong choices for OMOP/Athena concept text:

| Model | Hugging Face | Paper |
|-------|--------------|-------|
| **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) | [model card](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | [Self-Alignment Pretraining for Biomedical Entity Representations (NAACL 2021) — arXiv:2010.11784](https://arxiv.org/abs/2010.11784) |
| **BioLORD-2023** (`FremyCompany/BioLORD-2023`) | [model card](https://huggingface.co/FremyCompany/BioLORD-2023) | [BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights — arXiv:2311.16075](https://arxiv.org/abs/2311.16075) |

- **SapBERT** is tuned for biomedical *entity* representations (synonym/alias
  matching) and is the default. It works directly with this tool's default
  CLS-token pooling (`--pooling cls`).
- **BioLORD-2023** targets clinical *sentence* similarity and often performs
  better on longer, descriptive concept names. It is a `sentence-transformers`
  model trained with **mean** pooling, so run it with `--pooling mean` to
  reproduce its published behaviour:
  `gpu-embed embed --model FremyCompany/BioLORD-2023 --pooling mean`.

Whichever model you choose, the `model_version` digest (a SHA-256 of the
weights, with non-default pooling folded in) keeps each model's embeddings from
being silently mixed with another's — or with a different pooling of the same
weights — in the same store. Pin `--model-revision` for reproducible downloads.

---

## CPT-4 population (requires UMLS license)

Athena ships CPT-4 concepts as a stub; the actual concept names are populated
by a Java tool (`cpt4.jar`) that calls the UMLS API using your license key.
This step must be run **before** embedding whenever you include CPT-4 concepts.

1. Register for a free UMLS account and generate an API key at
   <https://uts.nlm.nih.gov/uts/profile>.
2. Set `UMLS_API_KEY` in your `.env`.
3. Run:

```bash
uv run gpu-embed cpt4
# or, to point at a non-default vocab directory:
uv run gpu-embed cpt4 --vocab-dir /path/to/vocab
```

This invokes `cpt4.jar` (located via `CPT4_JAR` in `.env`) with your UMLS
API key and populates CPT-4 names directly into the Athena CSV files in place.
Java (JRE ≥ 11) must be available on `PATH`, discoverable via `JAVA_HOME`, or
installed in a standard Windows location.

The `cpt4` subcommand will:
- Verify `cpt4.jar` exists at the configured path
- Verify `UMLS_API_KEY` is set
- Resolve Java from `PATH`, `JAVA_HOME`, or common Windows install locations
- Run `java -Dumls-apikey=<key> -jar cpt4.jar 5` from the vocab directory
- Stream stdout/stderr so you see progress in real time
- Exit non-zero if the Java process fails

---

## CLI usage

The tool has eight subcommands:

```
gpu-embed embed     [OPTIONS] [CSV_PATH...]   — batch embed concepts
gpu-embed export    [OPTIONS] OUTPUT_DIR      — export DB rows to sharded parquet
gpu-embed status    [OPTIONS]                — show what is stored in the DB
gpu-embed model-registry [OPTIONS]           — show hash -> model/revision mappings
gpu-embed coverage  [OPTIONS] [CSV_PATH...]   — identify unembedded concepts
gpu-embed cleanup   [OPTIONS]                — delete embeddings for a model/vocabularies
gpu-embed migrate-store [OPTIONS]            — materialize/initialize the parquet store
gpu-embed migrate-lance [OPTIONS]            — migrate a legacy .duckdb store into a Lance store
gpu-embed compact   [OPTIONS]                — compact a Lance store + prune old versions
gpu-embed cpt4      [OPTIONS]                — populate CPT-4 names via Java
```

Running `gpu-embed` without a subcommand is equivalent to `gpu-embed embed`.

### Storage model and migration

The backend is chosen by the `--db` path suffix:

- **`.lance` (default, `embeddings.lance`):** an embedded, ACID, versioned Lance
  store — the recommended live store (details in the Lance section below).
- **`.duckdb` (for example `embeddings.duckdb`):** the native DuckDB
  single-writer table; fast local upserts but no cross-process concurrency. The
  prior default, still fully supported — migrate one into Lance with
  `gpu-embed migrate-lance` (see [Migrating an existing DuckDB store](#migrating-an-existing-duckdb-store-to-lance)).
- **a directory (for example `embeddings/`):** parquet-sharded storage under
  `model_version=<digest>/vocabulary_id=<value>/part-*.parquet` — a
  migration/export artifact, not a live write store.

- Model provenance is stored alongside the Lance/parquet data under
  `_meta/model_registry/part-*.parquet` with one deduplicated row per
  `model_version` (`model_id`, `model_revision`, `recorded_at`).
- `concept_embeddings` is exposed as a DuckDB relation (a real table for
  `.duckdb`; a view over the dataset for Lance/parquet) for all reads and exports.
- New shard writes default to Snappy compression for faster write throughput;
  existing shards with other codecs (for example ZSTD) remain readable and do
  not require conversion.
- Use `gpu-embed export ...` as the standard Snowflake handoff path.
- Use `gpu-embed migrate-store --db embeddings.duckdb` only when you need a
  full parquet mirror of a DuckDB store (`embeddings/`).

#### Lance backend (default)

Lance is the default store: an embedded, ACID, versioned store whose
`merge_insert` upserts are **O(changes)** (deletion vectors), so a scattered
re-embed rewrites only the changed rows, not the whole partition. It also lets
`embed` (intermittent writes) and `export` / `status` run **concurrently across
processes** — which a single `.duckdb` file cannot, since it holds an exclusive
lock. This is the adopted live store for the ACID + concurrency requirement (see
`docs/adr_lance_store_proposal.md`). `pylance` is a base dependency, so a normal
install (`uv sync`) already has it.

- **A `.lance` store is a container directory:**
  - `<store>.lance/concept_embeddings.lance/` — the Lance dataset (the vectors).
  - `<store>.lance/_meta/model_registry/` — model provenance (parquet, same
    layout as the parquet store).
  - `<store>.lance/_meta/meta.duckdb` — a small DuckDB **sidecar** holding the
    CSV-fingerprint and weight-hash caches. They let `embed` skip unchanged CSVs
    and avoid re-hashing the ~440 MB weights file each run; the Lance store's own
    query connection is in-memory, so the sidecar is what persists them across
    runs. Lance maintenance (compact/cleanup) never touches it.
  - `concept_embeddings` is exposed as a DuckDB view for all reads/exports.
- **Maintenance:** `gpu-embed compact --db embeddings.lance` bin-packs fragments
  and prunes old versions. Reads are correct *without* compaction (deletion
  vectors already dedupe), so it is optional — but Lance retains old versions
  until pruned, so schedule it (or run it after an `embed`) to bound disk. It is
  a writer: never run it concurrently with a live `embed`.
- **Snowflake handoff unchanged:** `gpu-embed export` still emits plain sharded
  parquet regardless of backend.

##### Migrating an existing DuckDB store to Lance

If you have an existing `embeddings.duckdb` from before Lance became the default,
note that the default now points new runs at a fresh, empty `embeddings.lance`.
Carry the data over once with a deliberate cutover:

```bash
# 1. Stream the legacy table into the new default store (re-runnable; --reset
#    moves an existing .lance aside before re-migrating).
gpu-embed migrate-lance --db embeddings.lance --from embeddings.duckdb

# 2. From here on, embed/status/export default to embeddings.lance.
gpu-embed status --db embeddings.lance
```

The migration copies the embeddings, not the fingerprint/weight-hash caches, so
the **first** post-migration `embed` re-hashes the weights once and re-reads the
source CSVs once — but it re-embeds nothing (the migrated rows already satisfy
the change check) and it populates the sidecar, so subsequent runs skip again.
To keep using DuckDB instead, set `GPU_EMBED_DB=embeddings.duckdb` (or pass
`--db embeddings.duckdb`); the backend is fully supported, just no longer the
default.

#### Migration runtime notes

- Migration throughput is often non-linear: it may slow as larger
  vocabulary/model partitions are processed.
- This is expected as long as progress logs continue to advance
  (`partitions`, `rows`, `files`, `eta_minutes`).
- `migrate-lance` uses a conservative default batch size (`--batch-rows 25000`)
  to reduce peak RAM/SSD pressure on local machines. If you have headroom and
  want higher throughput, raise it (for example `--batch-rows 50000` or
  `--batch-rows 100000`).
- If progress appears stalled, prefer `gpu-embed migrate-store --db ...`
  over `status` for migration-only workflows to avoid extra summary queries.

#### Upgrading an existing store (`namespace` column)

The embedding identity key is `(namespace, concept_id, model_version)`, where
`namespace` defaults to `athena`. Upgrading a store created before `namespace`
existed needs **no manual DDL**:

- `ensure_schema` (run automatically by `embed`) adds the `namespace` column to
  an existing table via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` and backfills
  existing rows to `athena`. The same idempotent migration adds the nullable
  `source_id` / `mapping_wave` provenance columns (left NULL on existing rows).
  Just re-run `gpu-embed embed` as usual.
- The first post-upgrade run re-reads and re-hashes each input CSV once (the
  change-detection fingerprint now includes the namespace). Nothing is
  re-embedded — already-stored concepts are detected as unchanged — and
  subsequent runs skip unchanged files again. `model_version` is unchanged for
  FP32 runs, so no re-embedding is triggered by the upgrade.
- **Caveat — mixing namespaces in an existing file.** DuckDB cannot widen a
  table's PRIMARY KEY in place, so a pre-existing DB keeps its old
  `(concept_id, model_version)` key. That is fully correct for Athena-only use
  (every row is `namespace=athena`). But if you want to ingest **source
  concepts under a different `--namespace`** alongside existing data, start a
  **fresh DB file** (e.g. `--db embeddings_v2.duckdb`) so it gets the
  three-column key — otherwise a source `concept_id` that collides with an
  Athena one is not kept separate. Athena-only workflows need no action.

### `embed` — batch embed concepts

```
gpu-embed embed [OPTIONS] [CSV_PATH...]
```

When no `CSV_PATH` arguments are given, reads `CONCEPT.csv` from
`GPU_EMBED_VOCAB_DIR` (default `athena_vocab/`).

To embed source-side query concepts instead, pass `--source-parquet` pointing
at a Stage 0 `source_concepts` parquet file (or a directory of parquet files).
That path uses a source adapter rather than the Athena `CONCEPT.csv` ingest.

#### Positional

| Argument | Description |
|----------|-------------|
| `CSV_PATH` | Zero or more explicit paths to Athena `CONCEPT.csv` files |

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-dir` | `athena_vocab` | Directory containing `CONCEPT.csv` (used when no explicit path given) |
| `--source-parquet` | _(none)_ | Source-concept parquet file or directory to embed instead of Athena CSV |
| `--db` | `embeddings.lance` | Embedding store path. `.lance` = Lance store (ACID + cross-process readers, default); `.duckdb` = native DuckDB single-writer file; a directory = parquet store mode |
| `--batch-size` | `256` | Rows per GPU forward pass |
| `--model` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | HF model ID or local path |
| `--model-revision` | _(default branch)_ | HuggingFace commit hash, branch, or tag to pin the exact model revision |
| `--max-length` | `128` | Tokenizer max sequence length |
| `--pooling` | `cls` | Token pooling: `cls` (SapBERT default) or `mean` (e.g. BioLORD-2023). Non-default pooling is folded into `model_version` |
| `--upsert-every-batches` | `250` | Checkpoint writes every N embedding batches |
| `--ingest-engine` | `duckdb` | CSV ingest engine: `duckdb` (default) or `python` fallback |
| `--device` | auto | `cuda`, `cpu`, or `mps` |
| `--verbose` | false | Enable detailed logging and progress visibility |
| `--force` | false | Re-embed rows that already exist in the store |
| `--vocabulary-id` | _(highest-yield set)_ | Keep only these vocabulary IDs (repeatable or comma-delimited). When omitted, defaults to the curated highest-yield vocabularies (see below). Pass `--vocabulary-id all` to embed every vocabulary instead. |
| `--domain-id` | _(all)_ | Keep only these domain IDs (repeatable) |
| `--concept-class-id` | _(all)_ | Keep only these concept class IDs (repeatable) |
| `--standard-concept` | _(all)_ | Keep only `S`, `C`, or _(blank)_ rows (repeatable) |
| `--invalid-reason` | _(all)_ | Keep only rows with this invalid_reason; use `valid` as a shorthand for NULL/empty (repeatable) |
| `--text-field` | `concept_name` | Column(s) to concatenate as embedding input (repeatable) |
| `--source-text-field` | `source_name` | Source parquet field(s) to concatenate as embedding input (repeatable) |
| `--separator` | `" "` | Separator between concatenated text fields |
| `--namespace` | `athena` | Identity namespace; use a distinct value for source-concept datasets so their `concept_id`s don't collide with Athena |
| `--source-namespace` | `source` | Default namespace used by `--source-parquet` runs unless `--namespace` is passed |

Source-parquet mode does not use the Athena filter flags (`--vocabulary-id`,
`--domain-id`, `--concept-class-id`, `--standard-concept`, `--invalid-reason`).
Instead, adapt the source parquet and choose the embed text with
`--source-text-field`.

#### Round-tripping source provenance

A Stage 0 `source_concepts` parquet is keyed by `(mapping_wave, source_id)` in
concept-mapper. When `embed` adapts it, the string `source_id` is hashed into a
BIGINT `concept_id` surrogate (so it fits the embedding identity key and cannot
collide with Athena `concept_id`s under a distinct `--namespace`). That hash is
one-way, so the original `source_id` — and the `mapping_wave` — would otherwise
be lost, and the resulting vectors could not be rejoined to `source_concepts`.

To prevent that, `embed` carries both keys through unchanged as the nullable
`source_id` / `mapping_wave` columns (NULL for Athena concepts, populated for
source rows). They survive ingest, the DuckDB store, and the parquet `export`,
so the downstream load can `MERGE` embedded source vectors back into
concept-mapper's `source_concepts` on `(mapping_wave, source_id)`. The
Snowflake target table and MERGE that do this are in
[`docs/runbooks/s3_to_snowflake_load.md`](docs/runbooks/s3_to_snowflake_load.md).
If a source parquet predates the `mapping_wave` column, ingest substitutes NULL
rather than failing.

#### Default vocabularies (highest-yield set)

A full Athena `CONCEPT.csv` spans dozens of vocabularies, many of which add
little value for downstream concept mapping while inflating embedding time and
store size. So when **no** `--vocabulary-id` is given, `embed` does **not**
embed everything — it defaults to a curated set of the highest-yield Athena
vocabularies:

| Domain | `vocabulary_id`(s) |
|--------|--------------------|
| Conditions | `SNOMED`, `ICD9CM`, `ICD10CM` |
| Procedures | `CPT4`, `ICD9Proc`, `ICD10PCS` |
| Drugs | `RxNorm`, `RxNorm Extension`, `NDC` |
| Labs / measurements | `LOINC` |
| Demographics | `Race`, `Ethnicity` |
| Providers | `ABMS`, `NUCC`, `Medicare Specialty` |

To override:

```bash
# Embed every vocabulary present in the CSV (reserved sentinel "all"):
gpu-embed embed --vocabulary-id all

# Restrict to an explicit subset (disables the default):
gpu-embed embed --vocabulary-id SNOMED,RxNorm
```

> **CPT-4 note:** Athena ships CPT-4 with blank concept names; run
> `gpu-embed cpt4` (UMLS license required) to populate them *before* embedding,
> otherwise the CPT4 rows embed degenerate text.

The default list is the `DEFAULT_VOCABULARY_IDS` constant in
`src/gpu_embedder/models.py`; edit it there to change the curated set.

### `cpt4` — populate CPT-4 names via Athena Java tool

```
gpu-embed cpt4 [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-dir` | `GPU_EMBED_VOCAB_DIR` / `athena_vocab` | Directory containing `cpt4.jar` and the Athena CSVs |
| `--jar` | `CPT4_JAR` / `athena_vocab/cpt4.jar` | Explicit path to `cpt4.jar` |
| `--api-key` | `UMLS_API_KEY` | UMLS API key (prefer setting in `.env`) |

### `status` — show what is stored in the embeddings store

```
gpu-embed status [OPTIONS]
```

Prints the model versions stored in the embeddings store and a per-(vocabulary, domain)
breakdown of embedded concept counts. No source CSV is required.

Use `--backfill-from-logs` to infer missing model metadata from prior embed logs
before rendering status (helpful after legacy store migrations).

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `embeddings.lance` | Embedding store path to inspect |
| `--model-version` | _(most recent)_ | Show breakdown for the version starting with this prefix |
| `--backfill-from-logs` | false | Parse `GPU_EMBED_LOG_DIR` logs for model/revision lines and upsert derived hash mappings before status output |
| `--log-dir` | `logs` | Directory containing `gpu-embed` log files |

### `model-registry` — inspect model hash provenance

```
gpu-embed model-registry [OPTIONS]
```

Shows mappings between `model_version` hashes and model metadata, including
`model_id`, `model_revision`, `precision`, and `quantization_scheme`.

- In `.duckdb` store mode, metadata is stored in the `model_registry` table.
- In parquet store mode, metadata is stored in `_meta/model_registry/*.parquet`.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `embeddings.lance` | Embedding store path to inspect |
| `--backfill-from-logs` | false | Parse `GPU_EMBED_LOG_DIR` logs for model/revision lines and upsert derived hash mappings |
| `--log-dir` | `logs` | Directory containing `gpu-embed` log files |

### `export` — write Hive-partitioned parquet by model and vocabulary

```
gpu-embed export [OPTIONS] OUTPUT_DIR
```

Exports rows from `concept_embeddings` into Hive-partitioned parquet files:

`OUTPUT_DIR/model_version=<digest>/vocabulary_id=<value>/part-00000.parquet`

This mirrors the parquet store / `migrate-store` layout, so a single uniform
Hive-partitioned layout is used everywhere (S3, Snowflake external stages).
Because the path includes `model_version=<digest>`, exporting more than one
model version into the same `OUTPUT_DIR` is safe — different versions land in
separate partitions instead of colliding on `part-*.parquet` filenames.

Pooling is folded into `model_version` (see "Choosing an embedding model"), so a
`cls` and a `mean` export of the same weights automatically land under distinct
`model_version=` partitions. When a `--model-version` prefix (or the default
"most recent") matches **both** poolings, the selection is ambiguous: the command
lists the candidates and exits, asking you to add `--pooling {cls,mean}`. Use
`gpu-embed model-registry` to see which versions are `cls` vs `mean` first.

Within each partition, sharding is controlled by `--shard-rows` (rows per file).

| Flag | Default | Description |
|------|---------|-------------|
| `OUTPUT_DIR` | _(required)_ | Destination directory for parquet output |
| `--db` | `embeddings.lance` | Embedding store path to export from |
| `--model-version` | _(most recent)_ | Export only the model version starting with this prefix |
| `--pooling` | _(none)_ | Disambiguate by pooling (`cls` or `mean`); required when the selection matches both |
| `--vocabulary-id` | _(all)_ | Export only these vocabulary IDs (repeatable or comma-delimited) |
| `--namespace` | _(all)_ | Export only this identity namespace |
| `--shard-rows` | `50000` | Max rows per parquet shard |
| `--compression` | `snappy` | Parquet codec: `zstd`, `snappy`, `gzip`, `brotli`, `lz4`, `uncompressed` |
| `--overwrite` | false | Replace existing shard files if present |

### `coverage` — identify unembedded concepts

```
gpu-embed coverage [OPTIONS] [CSV_PATH...]
```

Scans a `CONCEPT.csv` and compares every (vocabulary, domain) group against the
embeddings store, reporting how many concepts have been embedded and how many
remain.

By default the output is split into two sections:
- **Groups With Gaps** (needs embedding work)
- **Fully Embedded Groups** (already complete)

Use `--gaps-only` to hide the fully-embedded section.

When no `CSV_PATH` arguments are given, reads from `GPU_EMBED_VOCAB_DIR`.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `embeddings.lance` | Embedding store path to compare against |
| `--vocab-dir` | `athena_vocab` | Directory containing `CONCEPT.csv` |
| `--model-version` | _(most recent)_ | Limit comparison to the version starting with this prefix |
| `--show-complete` / `--gaps-only` | `show-complete` | Include or hide fully-embedded groups |
| `--csv`, `-o` | _(none)_ | Write the aggregated coverage report to a CSV file |

### `cleanup` — delete embeddings for a model / vocabularies

```
gpu-embed cleanup [OPTIONS]
```

Permanently deletes stored embeddings for **one model version**, restricted to
either **all** of its vocabularies or a chosen subset. It is cautious by design:

- It previews exactly what will be removed (model, vocabularies, and the count)
  before touching anything.
- It requires confirmation. Deleting a *subset* of vocabularies asks a `y/N`
  question; deleting an *entire* model version requires retyping the model
  version's 16-character hash prefix.
- `--dry-run` shows the plan and stops without deleting.
- `--yes` / `-y` skips the prompt for scripted use.

Run it with no options for a guided flow: it lists stored model versions (with
their `model_id`, e.g. `FremyCompany/BioLORD-2023`), then lists that model's
vocabularies with counts so you can pick numbers or choose `A` for all.

After a delete it also invalidates the affected `csv_fingerprints` so a later
`embed` run re-reads the source CSVs (the fingerprints otherwise claim the file
was fully ingested). When a model version is left with zero embeddings, its
`model_registry` and weight-hash cache entries are removed too.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `embeddings.lance` | Embedding store path to clean up |
| `--model-version` | _(interactive)_ | Model version hash prefix to delete (must match exactly one) |
| `--vocabulary-id` | _(interactive)_ | Vocabulary IDs to delete (repeatable or comma-delimited) |
| `--all-vocabularies` | `false` | Delete every vocabulary for the model version |
| `--dry-run` | `false` | Show what would be deleted, then stop |
| `--yes`, `-y` | `false` | Skip the confirmation prompt |

```bash
# Guided, fully interactive: choose model, then vocabularies
gpu-embed cleanup

# Delete just the BioLORD SNOMED + LOINC embeddings, no prompt
gpu-embed cleanup --model-version 3f2a9c --vocabulary-id SNOMED,LOINC --yes

# Preview removing an entire model version without deleting
gpu-embed cleanup --model-version 3f2a9c --all-vocabularies --dry-run
```

### Examples

```bash
# Populate CPT-4 names first (only needed once per Athena download)
gpu-embed cpt4

# Embed UCUM concepts only
gpu-embed embed --vocabulary-id UCUM

# Same, but with detailed logging and progress visibility
gpu-embed embed --vocabulary-id UCUM --verbose

# Force the Python ingest fallback if you need to debug DuckDB parsing
gpu-embed embed --vocabulary-id UCUM --ingest-engine python

# Embed all standard valid concepts from athena_vocab/ (default dir)
gpu-embed embed --standard-concept S --invalid-reason valid

# Embed all LOINC standard concepts (explicit CSV path)
gpu-embed embed CONCEPT.csv \
  --vocabulary-id LOINC \
  --standard-concept S \
  --invalid-reason valid

# Embed SNOMED + RxNorm, force re-embedding even if present
gpu-embed embed \
  --vocabulary-id SNOMED \
  --vocabulary-id RxNorm \
  --force

# Equivalent comma-delimited form
gpu-embed embed --vocabulary-id SNOMED,RxNorm --force

# Embed with concept_code prepended to name
gpu-embed embed \
  --text-field concept_code \
  --text-field concept_name \
  --separator ": "

# Point at a specific store path and use CPU
gpu-embed embed /data/vocab/CONCEPT.csv \
  --db /data/embeddings \
  --device cpu \
  --batch-size 64

# Show what model versions are stored and concept counts by vocabulary/domain
gpu-embed status

# Export most recent model version into Hive-partitioned parquet
# (model_version=<digest>/vocabulary_id=<value>/part-*.parquet)
# Errors and lists candidates if the selection matches both cls and mean.
gpu-embed export exports/parquet --shard-rows 50000

# Export the mean-pooled BioLORD-2023 embeddings (disambiguate by pooling)
gpu-embed export exports/parquet --pooling mean

# Export only SNOMED and LOINC for a specific model version prefix
gpu-embed export exports/parquet \
  --model-version abc12345 \
  --vocabulary-id SNOMED,LOINC \
  --shard-rows 50000

# Show only the breakdown for a specific model version (prefix match)
gpu-embed status --model-version abc12345

# Find all vocabularies/domains with unembedded concepts (default CONCEPT.csv)
gpu-embed coverage

# Hide fully-embedded groups and show only gaps
gpu-embed coverage --gaps-only

# Coverage against an explicit CSV and specific store path
gpu-embed coverage /data/vocab/CONCEPT.csv --db /data/embeddings

# Write coverage results to CSV for follow-up workflows
gpu-embed coverage --csv coverage_report.csv
```

---

## Deployment runbook

Operational steps for export → AWS auth → S3 copy/sync → Snowflake `COPY INTO`
and idempotent `MERGE` are documented in:

- `docs/runbooks/s3_to_snowflake_load.md`

Quick start:

```bash
uv run gpu-embed export exports/parquet --db embeddings.duckdb --shard-rows 50000
AWS_PAGER="" aws s3 sync exports/parquet s3://<your-bucket>/concept_embeddings/
# Optional named profile: add --profile <aws-profile> or set AWS_PROFILE
```

---

## Logical schema

All backends expose the same `concept_embeddings` relation — a DuckDB view over
the dataset for the default Lance store and the parquet store, and a real DuckDB
table for the `.duckdb` backend. Logical columns:

```sql
CREATE TABLE concept_embeddings (
    namespace           TEXT      NOT NULL DEFAULT 'athena',
    concept_id          BIGINT    NOT NULL,
    concept_name        TEXT      NOT NULL,
    domain_id           TEXT,
    vocabulary_id       TEXT,
    concept_class_id    TEXT,
    standard_concept    TEXT,
    concept_code        TEXT,
    invalid_reason      TEXT,
    embedding           FLOAT[768] NOT NULL,   -- model embedding (CLS-pooled, L2-normalized)
    embed_text          TEXT      NOT NULL,    -- exact string that was embedded
    model_version       TEXT      NOT NULL,    -- SHA-256 digest of model weights
    embedded_at         TIMESTAMP NOT NULL,
    source_id           TEXT,                  -- source-dataset key (NULL for Athena)
    mapping_wave        TEXT,                  -- concept-mapper wave (NULL for Athena)
    PRIMARY KEY (namespace, concept_id, model_version)
);
```

The `model_version` digest ensures that embeddings from different model
checkpoints (or different models entirely) are never silently mixed.

The nullable `source_id` / `mapping_wave` columns are populated only for
`--source-parquet` runs (NULL for Athena concepts). They carry the
concept-mapper `source_concepts` natural key through embedding so the vectors
can be rejoined on `(mapping_wave, source_id)` — see
[Round-tripping source provenance](#round-tripping-source-provenance) below.
They are intentionally **not** part of the primary key: the hashed `concept_id`
surrogate already disambiguates source rows within their namespace.

---

## Environment variables / `.env`

Copy `.env.example` to `.env` and edit. All settings have sensible defaults
so an empty `.env` is valid for local GPU runs against `athena_vocab/`.

```dotenv
# Paths
GPU_EMBED_VOCAB_DIR=athena_vocab
GPU_EMBED_SOURCE_PARQUET=source_concepts/
GPU_EMBED_DB=embeddings.lance    # Lance store (default); .duckdb = native DuckDB table; a directory = parquet store

# Model (see "Choosing an embedding model"; e.g. FremyCompany/BioLORD-2023)
GPU_EMBED_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
GPU_EMBED_MODEL_REVISION=       # HF commit hash / branch / tag; blank = default branch
GPU_EMBED_DEVICE=auto
GPU_EMBED_BATCH_SIZE=256
GPU_EMBED_MAX_LENGTH=128
GPU_EMBED_UPSERT_EVERY_BATCHES=250
GPU_EMBED_TEXT_FIELDS=concept_name
GPU_EMBED_SOURCE_TEXT_FIELDS=source_name,source_description
GPU_EMBED_SEPARATOR=" "
GPU_EMBED_NAMESPACE=athena
GPU_EMBED_SOURCE_NAMESPACE=source

# CPT-4 / UMLS
UMLS_API_KEY=your-key-here
CPT4_JAR=athena_vocab/cpt4.jar
```

CLI flags always take precedence over `.env` values.

---

## Running tests

```bash
uv run pytest                  # full suite
uv run pytest tests/unit/      # fast unit tests only (no GPU required)
```

Integration tests require a CUDA GPU and are skipped automatically when none is
available (`pytest.mark.gpu`).
