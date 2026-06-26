# gpu-embedder

Batch-embed OHDSI Athena concept CSVs with SapBERT (FP32, GPU) and persist the
vectors to DuckDB. Already-embedded concepts are skipped unless `--force` is
passed, so runs are safe to restart or extend incrementally.

---

## What it does

1. **Reads** `CONCEPT.csv` from the `athena_vocab/` directory (configurable via
   `GPU_EMBED_VOCAB_DIR` in `.env` or `--vocab-dir` on the CLI), or from
   explicit paths passed as arguments.
2. **Filters** rows by any combination of Athena columns
   (`vocabulary_id`, `domain_id`, `concept_class_id`, `standard_concept`,
   `invalid_reason`) so you only embed what you need. CSV scan + filter are
   pushed through **DuckDB by default** before Pydantic validation.
3. **Embeds** the filtered concept names with
   [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
   running in FP32 on a CUDA GPU (falls back to CPU when no GPU is available,
   but will be slow).
4. **Writes** each concept's vector plus its metadata into a local DuckDB
   database (`embeddings.duckdb` by default). Embedding runs are stamped with a
   `model_version` digest so mixed-version stores are detected.
5. **Skips** rows whose `(concept_id, model_version)` pair already exists in
   the store. Pass `--force` to re-embed unconditionally.

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
│       ├── cli.py          # Typer app: `embed` + `cpt4` subcommands
│       ├── config.py       # Settings (Pydantic BaseSettings + env)
│       ├── models.py       # Pydantic row models + DuckDB schema DDL
│       ├── ingest.py       # DuckDB-backed CSV scan + filter → ConceptRow records
│       ├── embed.py        # SapBERT FP32 tokenize + forward pass
│       └── store.py        # DuckDB upsert / existence checks
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

# GPU support — install PyTorch with CUDA before syncing if needed:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
# then:
uv sync --group dev

# 3. Place Athena vocabulary files in athena_vocab/
#    (or set GPU_EMBED_VOCAB_DIR in .env to your download location)
```

The SapBERT model is downloaded from Hugging Face on first run and cached
locally via the normal `~/.cache/huggingface` path.

DuckDB is also the default engine for reading and filtering Athena TSVs, so
large source files are narrowed before they are validated or embedded.

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

The tool has two subcommands:

```
gpu-embed embed [OPTIONS] [CSV_PATH...]
gpu-embed cpt4  [OPTIONS]
```

Running `gpu-embed` without a subcommand is equivalent to `gpu-embed embed`.

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
| `--db` | `embeddings.duckdb` | DuckDB file path |
| `--batch-size` | `256` | Rows per GPU forward pass |
| `--model` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | HF model ID or local path |
| `--model-revision` | _(default branch)_ | HuggingFace commit hash, branch, or tag to pin the exact model revision |
| `--max-length` | `128` | Tokenizer max sequence length |
| `--ingest-engine` | `duckdb` | CSV ingest engine: `duckdb` (default) or `python` fallback |
| `--device` | auto | `cuda`, `cpu`, or `mps` |
| `--verbose` | false | Enable detailed logging and progress visibility |
| `--force` | false | Re-embed rows that already exist in the store |
| `--vocabulary-id` | _(all)_ | Keep only these vocabulary IDs (repeatable) |
| `--domain-id` | _(all)_ | Keep only these domain IDs (repeatable) |
| `--concept-class-id` | _(all)_ | Keep only these concept class IDs (repeatable) |
| `--standard-concept` | _(all)_ | Keep only `S`, `C`, or _(blank)_ rows (repeatable) |
| `--invalid-reason` | _(all)_ | Keep only rows with this invalid_reason; use `valid` as a shorthand for NULL/empty (repeatable) |
| `--text-field` | `concept_name` | Column(s) to concatenate as embedding input (repeatable) |
| `--separator` | `" "` | Separator between concatenated text fields |

### `cpt4` — populate CPT-4 names via Athena Java tool

```
gpu-embed cpt4 [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-dir` | `GPU_EMBED_VOCAB_DIR` / `athena_vocab` | Directory containing `cpt4.jar` and the Athena CSVs |
| `--jar` | `CPT4_JAR` / `athena_vocab/cpt4.jar` | Explicit path to `cpt4.jar` |
| `--api-key` | `UMLS_API_KEY` | UMLS API key (prefer setting in `.env`) |

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

# Embed with concept_code prepended to name
gpu-embed embed \
  --text-field concept_code \
  --text-field concept_name \
  --separator ": "

# Point at a specific DuckDB file and use CPU
gpu-embed embed /data/vocab/CONCEPT.csv \
  --db /data/embeddings/omop.duckdb \
  --device cpu \
  --batch-size 64
```

---

## DuckDB schema

```sql
CREATE TABLE concept_embeddings (
    concept_id          BIGINT    NOT NULL,
    concept_name        TEXT      NOT NULL,
    domain_id           TEXT,
    vocabulary_id       TEXT,
    concept_class_id    TEXT,
    standard_concept    TEXT,
    concept_code        TEXT,
    invalid_reason      TEXT,
    embedding           FLOAT[768] NOT NULL,   -- SapBERT CLS vector
    embed_text          TEXT      NOT NULL,    -- exact string that was embedded
    model_version       TEXT      NOT NULL,    -- SHA-256 digest of model weights
    embedded_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (concept_id, model_version)
);
```

The `model_version` digest ensures that embeddings from different model
checkpoints are never silently mixed.

---

## Environment variables / `.env`

Copy `.env.example` to `.env` and edit. All settings have sensible defaults
so an empty `.env` is valid for local GPU runs against `athena_vocab/`.

```dotenv
# Paths
GPU_EMBED_VOCAB_DIR=athena_vocab
GPU_EMBED_DB=embeddings.duckdb

# Model
GPU_EMBED_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
GPU_EMBED_MODEL_REVISION=       # HF commit hash / branch / tag; blank = default branch
GPU_EMBED_DEVICE=auto
GPU_EMBED_BATCH_SIZE=256
GPU_EMBED_MAX_LENGTH=128
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
