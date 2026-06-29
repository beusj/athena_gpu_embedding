# gpu-embedder

Batch-embed OHDSI Athena concept CSVs with a biomedical embedding model
(SapBERT by default; configurable — see
[Choosing an embedding model](#choosing-an-embedding-model)) in FP32 on a GPU,
persisting the vectors to a local DuckDB table (`embeddings.duckdb` by default).
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
4. **Writes** each concept's vector plus its metadata into a local DuckDB table
  (`embeddings.duckdb` by default). A directory path instead selects the opt-in
  parquet-sharded store, a migration/export artifact rather than the recommended
  live store (see [Storage model and migration](#storage-model-and-migration)).
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
│       └── store.py        # native DuckDB table store (+ optional parquet shards & query view)
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

# 2. Install project + dev deps (requires uv ≥ 0.4)
uv sync --group dev

# 3. Place Athena vocabulary files in athena_vocab/
#    (or set GPU_EMBED_VOCAB_DIR in .env to your download location)
```

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
> uv sync --group dev
> ```

`pyproject.toml` includes a `[tool.uv.sources]` entry that routes `torch` to
the CUDA 13.0 index on Linux and Windows, so a plain `uv sync` automatically
installs the CUDA build — no extra step needed after the initial sync.

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
  matching) and is the default. It works directly with this tool's CLS-token
  pooling.
- **BioLORD-2023** targets clinical *sentence* similarity and often performs
  better on longer, descriptive concept names. It is a `sentence-transformers`
  model trained with **mean** pooling, whereas this pipeline currently pools the
  CLS token — so treat it as a candidate to evaluate (and adjust pooling to
  reproduce its published behaviour) rather than a zero-change swap.

Whichever model you choose, the `model_version` digest (a SHA-256 of the
weights) keeps each model's embeddings from being silently mixed with another's
in the same store. Pin `--model-revision` for reproducible downloads.

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
gpu-embed cpt4      [OPTIONS]                — populate CPT-4 names via Java
```

Running `gpu-embed` without a subcommand is equivalent to `gpu-embed embed`.

### Storage model and migration

- Default store path is `embeddings.duckdb` (file) for fast local upserts.
- Directory paths (for example `embeddings/`) use parquet-sharded storage under
  `model_version=<digest>/vocabulary_id=<value>/part-*.parquet`.
- Model provenance is stored alongside shards under
  `_meta/model_registry/part-*.parquet` with one deduplicated row per
  `model_version` (`model_id`, `model_revision`, `recorded_at`).
- `concept_embeddings` is exposed as a DuckDB view for all reads and exports.
- New shard writes default to Snappy compression for faster write throughput;
  existing shards with other codecs (for example ZSTD) remain readable and do
  not require conversion.
- Use `gpu-embed export ...` as the standard Snowflake handoff path.
- Use `gpu-embed migrate-store --db embeddings.duckdb` only when you need a
  full parquet mirror of the local store (`embeddings/`).
- Local embedding runs can continue to use `embeddings.duckdb` directly.

#### Migration runtime notes

- Migration throughput is often non-linear: it may slow as larger
  vocabulary/model partitions are processed.
- This is expected as long as progress logs continue to advance
  (`partitions`, `rows`, `files`, `eta_minutes`).
- If progress appears stalled, prefer `gpu-embed migrate-store --db ...`
  over `status` for migration-only workflows to avoid extra summary queries.

#### Upgrading an existing store (`namespace` column)

The embedding identity key is `(namespace, concept_id, model_version)`, where
`namespace` defaults to `athena`. Upgrading a store created before `namespace`
existed needs **no manual DDL**:

- `ensure_schema` (run automatically by `embed`) adds the `namespace` column to
  an existing table via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` and backfills
  existing rows to `athena`. Just re-run `gpu-embed embed` as usual.
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

#### Positional

| Argument | Description |
|----------|-------------|
| `CSV_PATH` | Zero or more explicit paths to Athena `CONCEPT.csv` files |

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-dir` | `athena_vocab` | Directory containing `CONCEPT.csv` (used when no explicit path given) |
| `--db` | `embeddings.duckdb` | Embedding store path (default fast local DuckDB file; directory enables parquet store mode) |
| `--batch-size` | `256` | Rows per GPU forward pass |
| `--model` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | HF model ID or local path |
| `--model-revision` | _(default branch)_ | HuggingFace commit hash, branch, or tag to pin the exact model revision |
| `--max-length` | `128` | Tokenizer max sequence length |
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
| `--separator` | `" "` | Separator between concatenated text fields |
| `--namespace` | `athena` | Identity namespace; use a distinct value for source-concept datasets so their `concept_id`s don't collide with Athena |

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

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `embeddings.duckdb` | Embedding store path to inspect |
| `--model-version` | _(most recent)_ | Show breakdown for the version starting with this prefix |

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
| `--db` | `embeddings.duckdb` | Embedding store path to inspect |
| `--backfill-from-logs` | false | Parse `GPU_EMBED_LOG_DIR` logs for model/revision lines and upsert derived hash mappings |
| `--log-dir` | `logs` | Directory containing `gpu-embed` log files |

### `export` — write sharded parquet by vocabulary directory

```
gpu-embed export [OPTIONS] OUTPUT_DIR
```

Exports rows from `concept_embeddings` into parquet files under:

`OUTPUT_DIR/<vocabulary_id>/part-00000.parquet`

Sharding is controlled by `--shard-rows` (rows per file).

| Flag | Default | Description |
|------|---------|-------------|
| `OUTPUT_DIR` | _(required)_ | Destination directory for parquet output |
| `--db` | `embeddings.duckdb` | Embedding store path to export from |
| `--model-version` | _(most recent)_ | Export only the model version starting with this prefix |
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
| `--db` | `embeddings.duckdb` | Embedding store path to compare against |
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
| `--db` | `embeddings.duckdb` | Embedding store path to clean up |
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

# Export most recent model version into sharded parquet by vocabulary directory
gpu-embed export exports/parquet --shard-rows 50000

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

The default store is a native DuckDB table named `concept_embeddings` (in the
opt-in parquet store mode the same columns are exposed through a DuckDB view of
that name). Logical columns:

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
    PRIMARY KEY (namespace, concept_id, model_version)
);
```

The `model_version` digest ensures that embeddings from different model
checkpoints (or different models entirely) are never silently mixed.

---

## Environment variables / `.env`

Copy `.env.example` to `.env` and edit. All settings have sensible defaults
so an empty `.env` is valid for local GPU runs against `athena_vocab/`.

```dotenv
# Paths
GPU_EMBED_VOCAB_DIR=athena_vocab
GPU_EMBED_DB=embeddings.duckdb   # native DuckDB table (default); a directory path selects the parquet store

# Model (see "Choosing an embedding model"; e.g. FremyCompany/BioLORD-2023)
GPU_EMBED_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
GPU_EMBED_MODEL_REVISION=       # HF commit hash / branch / tag; blank = default branch
GPU_EMBED_DEVICE=auto
GPU_EMBED_BATCH_SIZE=256
GPU_EMBED_MAX_LENGTH=128
GPU_EMBED_UPSERT_EVERY_BATCHES=250
GPU_EMBED_TEXT_FIELDS=concept_name
GPU_EMBED_SEPARATOR=" "

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
