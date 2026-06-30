# Cross-Repo Alignment: Embedding → Mapping → dbt

**Status:** Shared contract. This file is committed **identically** to all three
repositories and is the single source of truth for how they interoperate. It
describes the **target aligned state** (the normative contract in §4) and the
**current gaps** against it (§5). Where current code diverges from §4, §4 wins
and §5/§7 track the work to converge.

**Repositories**

| Repo | Package / profile | Role |
|------|-------------------|------|
| [`athena_gpu_embedding`](https://github.com/beusj/athena_gpu_embedding) | `gpu-embedder` (`gpu-embed`) | Batch-embed OHDSI **Athena** concepts (and concept-mapper source concepts) with SapBERT on GPU; persist to Lance/DuckDB; export parquet for the warehouse. |
| [`llm_concept_mapping`](https://github.com/beusj/llm_concept_mapping) | `concept-mapper` | Map local Epic/EHR codes to OMOP standard concepts: hybrid retrieval (incl. semantic) → LLM rerank → human review → promote. |
| [`dbt_omop_clean`](https://github.com/beusj/dbt_omop_clean) | `dbt_omop_choa` | dbt OMOP v5.4 transformation of Epic Clarity; overlays the human-reviewed mappings into its STCM and surfaces unmapped gaps. |

A separate `vocabulary_loader` repo loads `OMOP_VOCAB` (`concept`,
`concept_synonym`, `concept_ancestor`, …) consumed by both `concept-mapper`
(`vocab_schema`) and dbt (`source('omop', …)`); it is the same vocabulary
`gpu-embedder` embeds from `CONCEPT.csv`. Keeping all four on the **same Athena
release** is a precondition for everything below.

---

## 1. End-to-end flow (the loop)

```
                ┌───────────── OMOP_VOCAB  (loaded by vocabulary_loader) ─────────────┐
                │   concept / concept_synonym / concept_ancestor  +  Athena CONCEPT.csv │
                └──────────────┬───────────────────────────────────────┬──────────────┘
                  (A) embed Athena standard concepts                    │ consumed as vocab
                               ▼                                        ▼
   ┌──────────────────────┐  export parquet → S3 → Snowflake   ┌──────────────────────────────┐
   │ athena_gpu_embedding │ ─────────────────────────────────▶ │ llm_concept_mapping          │
   │  SapBERT · CUDA      │   concept_embeddings (targets)      │  Stage 0  source_concepts    │
   │  FP32 · CLS          │ ◀───────────────────────────────── │  Stage 3  hybrid retrieval   │
   │  Lance/DuckDB store  │  (B) source_concepts parquet        │  Stage 4–6 rerank + review   │
   └──────────────────────┘   embed → MERGE back               │  promote → concept_mappings  │
                                  on (mapping_wave, source_id)  └───────────────┬──────────────┘
                                                  (C) source('omop_mapping',     │ promote (ACCEPTED/MODIFIED)
                                                      'concept_mappings')        ▼
                                                       ┌──────────────────────────────────────┐
                                                       │ dbt_omop_clean                        │
                                                       │  int_llm_current_mappings_stcm        │
                                                       │  → int_llm_mapped_codes → voc_priority│
                                                       │  → source_to_concept_map → CDM tables │
                                                       └───────────────┬───────────────────────┘
                            (D) source_to_concept_map where             │ unmapped codes (target_concept_id = 0)
                                target_concept_id = 0 feeds the next ────┘ re-enter Stage 0 (STCM mode)
                                mapping wave
```

---

## 2. Intersection contracts

| # | Producer → Consumer | Artifact / table | Identity / join key |
|---|---|---|---|
| **A** | `gpu-embedder` → `concept-mapper` | `concept_embeddings` — Athena **target** vectors | retrieval keys on `(concept_id, embed_model_version)` |
| **B** | `concept-mapper` ↔ `gpu-embedder` | `source_concepts` **query** vectors via parquet round-trip | `(mapping_wave, source_id)` |
| **C** | `concept-mapper` → dbt | `omop_mapping.concept_mappings` | natural `(source_code, source_vocabulary_id)` |
| **D** | dbt → `concept-mapper` | `source_to_concept_map` rows with `target_concept_id = 0` | `source_id = STCM_<UPPER(vocab)>_<md5(lower(trim(code)))>` |

**Key files per contract**

- **A** — produce: `gpu_embedder/cli.py` (`export`), `gpu_embedder/store.py`
  (`_EMBEDDING_COLUMNS`), runbook `docs/runbooks/s3_to_snowflake_load.md`.
  consume: `concept_mapper/sql/retrieval_semantic.sql`,
  `concept_mapper/sql/ddl/concept_embeddings.sql`,
  `concept_mapper/embeddings/build.py`.
- **B** — export/import: `concept_mapper/embeddings/gpu_export.py`
  (`EXPORT_COLUMNS`, `import_source_embeddings`); ingest:
  `gpu_embedder/ingest.py` (`read_source_parquet`, `_SOURCE_PARQUET_COLUMNS`,
  `_stable_source_concept_id`); table `concept_mapper/sql/ddl/source_concepts.sql`.
- **C** — produce: `concept_mapper/promote.py`,
  `sql/promote_select_accepted.sql`, `sql/promote_upsert_mapping.sql`,
  `sql/ddl/concept_mappings.sql`. consume:
  `dbt: models/_llm_mapping_sources.yml`,
  `models/02_intermediate/stcm/int_llm_current_mappings_stcm.sql`,
  `int_llm_mapped_codes.sql`.
- **D** — produce: dbt `source_to_concept_map` (STCM flow). consume:
  `concept_mapper/sql/stage0_select_source_inventory_stcm.sql`.

---

## 3. What is already aligned (preserve)

These are correct today; do not regress them when converging the rest.

1. **Source-parquet schema (B)** matches column-for-column:
   `gpu_export.EXPORT_COLUMNS` ≡ `gpu_embedder._SOURCE_PARQUET_COLUMNS`
   (`source_id, mapping_wave, source_name, source_description, source_domain,
   ehr_codes, sample_units, sample_values, data_type`).
2. **`(mapping_wave, source_id)`** is carried end-to-end and is the PK of both
   `source_concepts` and `concept_mappings`; `gpu-embedder` preserves it through
   the surrogate-hash embedding step specifically so vectors can be rejoined.
3. **STCM overlay (C)** joins on **natural keys** and re-validates the reviewer's
   target fresh against `concept` (standard `'S'`, not invalid, `id <> 0`), so it
   is robust to any future `source_id` hash change and only ever gap-fills.
4. **Gap feedback (D)** is a clean closed loop: dbt emits unmapped STCM rows
   (`target_concept_id = 0`); the mapper ingests exactly those and nothing else.
5. **`namespace`** isolates source-concept `concept_id`s from Athena ones in the
   embedding store, preventing PK collisions.

---

## 4. Canonical embedding contract (NORMATIVE)

There is **one** SapBERT vector space. A stored vector and a query vector are
compared only when they came from the **same artifact**, identified by one
`model_version` string that **both repos compute identically from config**.

### 4.1 Pinned artifact (decision: **FP32 + CLS everywhere**)

| Attribute | Value | Notes |
|---|---|---|
| `model_name` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | 768-dim PubMedBERT backbone. A different 768-dim biomedical encoder is allowed only if changed in lockstep in both repos. |
| `revision` | a pinned HF **commit hash** | Must be set in production and **identical** in both repos. `None`/unpinned is dev-only. |
| `precision` | **`fp32`** | No `fp16`/`bf16`/**int8**. `gpu-embedder` is already FP32-only; `concept-mapper` must set `EMBEDDING_QUANTIZATION=fp32` (it is `int8` by default today). |
| `pooling` | **`cls`** (`last_hidden_state[:, 0, :]`) | `gpu-embedder` default is CLS. `concept-mapper` loads the bare HF model via `SentenceTransformer`, which falls back to **mean** pooling — it **must** force a CLS pooling module to match. |
| `normalize` | `true` (L2) | Already agrees on both sides. |
| `dimension` | `768` | Stored as `VECTOR(FLOAT, 768)` in Snowflake. |

> **Why FP32 + CLS:** retrieval quality is the priority and the comparison must
> be apples-to-apples. int8 query vectors against FP32 document vectors (or CLS
> vs mean pooling) are *different spaces* and silently degrade recall. ONNX-FP32
> and CUDA-FP32 of the same weights are numerically equivalent for cosine, so the
> runtime engine (CUDA PyTorch vs ONNX CPU) may differ; precision and pooling may
> not.

### 4.2 `model_version` (single shared formula)

The **retrieval-facing** version is a pure function of the pinned vector-space
attributes — and **excludes the runtime engine** (CUDA vs ONNX) and throughput
knobs (batch size), because those do not change FP32+CLS vectors:

```
identity      = { model_name, revision, pooling, precision: "fp32",
                  normalize, dimension }
model_version = "sapbert-" + pooling + "-" + precision + "-" + sha256(canonical_json(identity))[:10]
              # e.g. "sapbert-cls-fp32-1a2b3c4d5e"
```

Both repos MUST produce the same string for the same pinned config. The on-disk
SHA-256 of the weights file stays as **provenance** (`gpu-embedder`'s
`model_registry`) — it is *not* the retrieval key.

### 4.3 Warehouse tables (one schema each)

There must be exactly **one** `concept_embeddings` contract that Stage 3 reads:

```sql
-- omop_mapping.concept_embeddings  (target Athena concepts)
concept_id          INT            NOT NULL
vocabulary_id       VARCHAR        NOT NULL
domain_id           VARCHAR        NOT NULL
concept_name        VARCHAR        NOT NULL
standard_concept    VARCHAR(1)
embedding           VECTOR(FLOAT, 768)          -- NOT ARRAY: VECTOR_COSINE_SIMILARITY requires VECTOR
embedded_at         TIMESTAMP_NTZ
embed_model_version VARCHAR                      -- §4.2 string; column name is embed_model_version
PRIMARY KEY (concept_id, embed_model_version)
```

```sql
-- omop_mapping.source_concepts  (query vectors live here)
... query_embedding VECTOR(FLOAT, 768), embed_model_version VARCHAR, embedded_at TIMESTAMP_NTZ ...
PRIMARY KEY (mapping_wave, source_id)
```

**Handoff rules**

- `gpu-embedder` export → `concept_embeddings`: the engine may store
  `model_version` internally, but anything landing in `omop_mapping` MUST present
  the column as **`embed_model_version`** holding the §4.2 string, and the vector
  MUST load as **`VECTOR(FLOAT, 768)`** (cast on load if the parquet column is a
  list/ARRAY). `gpu-embedder`'s extra columns (`namespace`, `concept_class_id`,
  `concept_code`, `invalid_reason`, `embed_text`, `source_id`, `mapping_wave`)
  are dropped or ignored for the target table.
- `gpu-embedder` export of **source** concepts → `source_concepts`: MERGE on
  `(mapping_wave, source_id)`, setting `query_embedding`, `embed_model_version`
  (§4.2), `embedded_at`. This is the same MERGE `concept-mapper`'s
  `import_source_embeddings` performs; the two MUST stay equivalent.

### 4.4 The single invariant

> Stage 3 (`retrieval_semantic.sql`) filters `embed_model_version = <§4.2
> string>` and discards source cache hits whose version differs. Therefore
> **every** producer of either target or query vectors — `concept-mapper` local,
> `concept-mapper` Snowflake UDF, and `gpu-embedder` — must stamp the **same**
> §4.2 string for the same pinned config, and must produce FP32 + CLS vectors.

### 4.5 Embed-text contract (two populations, one space)

There are **two embedding populations**. They embed **different text** but must
live in the **same vector space** so cosine similarity is meaningful:

| Population | Text embedded | Stored in | Produced by |
|---|---|---|---|
| **Target** concepts | bare `concept_name` | `concept_embeddings.embedding` | `gpu-embedder` (Athena CONCEPT.csv) / `concept-mapper` `build.py` |
| **Source** concepts (queries) | bare `source_name` | `source_concepts.query_embedding` | `concept-mapper` Stage 3 / `gpu-embed --source-parquet` |

SapBERT does **symmetric** biomedical name matching, so **both** sides embed the
**bare name** — do *not* concatenate vocabulary/domain/units/codes into the text,
or the vector shifts out of name space and stops matching the other population.
Concretely: keep `gpu-embedder`'s `GPU_EMBED_SOURCE_TEXT_FIELDS=source_name` (its
default) and `concept-mapper`'s `format_concept_text` / source `source_name` bare.

`model_version` (§4.2) intentionally identifies the **artifact**, not the **text**,
so it does *not* protect against an embed-text mismatch: two source runs with
different `SOURCE_TEXT_FIELDS` get the same version but different vectors. The
bare-name rule above is the guard; treat any deviation as a contract change.

---

## 5. Current gaps vs. §4 (inconsistency register)

Ranked; each lists evidence and the convergence action.

> **Implementation status** (branch `claude/align-embedding-mapping-dbt-9rkgnk`):
> items **1–4 are now largely addressed in code** — `concept-mapper` computes the
> §4.2 stamp, defaults to FP32, and forces CLS pooling; `gpu-embedder` exposes the
> same stamp (`gpu-embed retrieval-version`) and the runbook lands vectors in the
> contract tables with the VECTOR cast + `embed_model_version`. A shared golden
> string (`sapbert-cls-fp32-…`) is asserted in **both** repos' unit tests. Items
> **5–7 remain open** (target-ownership cutover, `source_vocabulary_id` casing,
> de-duplicating the `768`/model-name constants). The descriptions below are the
> design rationale and the remaining work.

1. **🔴 `model_version` is three incompatible schemes.**
   - `gpu-embedder`: SHA-256 of the weights file (`models.py` `EmbeddedRow`;
     `.env.example` "on-disk SHA-256 (model_version)").
   - `concept-mapper` local/UDF: `sapbert-{backend}-{quant}-{10hex}` →
     `sapbert-onnx-int8-…` (`embeddings/embedder.py` `model_version_for`).
   - `concept-mapper` GPU offload script: literal `sapbert-cuda-fp32`
     (`embeddings/gpu_export.py` docstring).
   A `gpu-embedder` vector can never satisfy Stage 3's filter.
   **→** adopt §4.2 in both repos; weights-SHA becomes provenance only.

2. **🔴 Vector spaces differ.** `gpu-embedder` = FP32 + CLS; `concept-mapper`
   default = int8 + (effectively) mean pooling (`config.py`
   `embedding_quantization="int8"`; `embedder.py` `SentenceTransformer(model)`
   with no CLS pooling module). **→** set mapper to FP32; add an explicit CLS
   pooling module to the mapper's loader; verify ONNX-FP32/CLS ≈ CUDA-FP32/CLS.

3. **🟠 Two `concept_embeddings` schemas collide.** Runbook target table uses
   `embedding ARRAY` + column `model_version`
   (`docs/runbooks/s3_to_snowflake_load.md`); the mapper DDL uses
   `VECTOR(FLOAT,768)` + `embed_model_version`
   (`sql/ddl/concept_embeddings.sql`). The documented load never reaches the
   table Stage 3 queries. **→** make the handoff land in §4.3 exactly (rename +
   VECTOR cast); update the runbook with a real loader and the source MERGE
   (only referenced today).

4. **✅ Duplicated GPU offload paths — hardened.** Decision: **keep**
   `embeddings/gpu_export.py` as the lightweight option. `import_source_embeddings`
   now **enforces the §4.2 contract version by default** (raises on mismatch;
   `--allow-version-mismatch` / `allow_version_mismatch=True` to override), and the
   inline-script docstring uses the shared stamp + CLS. Fully deleting the path in
   favour of `gpu-embed` remains optional, not required.

5. **✅ Target-embedding ownership — decided.** `gpu-embedder` is **canonical for
   both source and target** embeddings (both were embedded on GPU at FP32+CLS with
   one pinned revision). The mapper's `build.py`/UDF are **slow fallbacks** and now
   emit a warning (`embedding_fallback_path`) reminding the operator to match the
   pinned model/revision/FP32/CLS or it writes a divergent version.

6. **✅ `source_vocabulary_id` case — fixed.** `retrieval_code.sql` now compares the
   **vocabulary case-insensitively** (`UPPER(vocabulary_id)`; caller uppercases only
   the vocab portion). The **code stays exact** — UCUM is case-sensitive
   (`mg` ≠ `MG`). The dbt overlay join is unaffected (raw=raw still holds), and
   re-embedding is not involved (§5 item 9).

7. **✅ `768` / model name duplication — done.** Dimension centralized in
   `gpu_embedder.models.EMBEDDING_DIM` (re-used by `embed`/`store`); the mapper's
   `config` references `embedder.SAPBERT_DIMENSION` / `DEFAULT_MODEL_NAME`.
   Drift-guard tests in both repos + the model-swap checklist in §8. The
   `VECTOR(FLOAT,768)` / `FLOAT[768]` SQL literals are irreducible (SQL can't read a
   Python constant) and are listed in §8.

8. **🟡 OPEN QUESTION — source-domain naming alignment.** concept-mapper's
   `source_domain` taxonomy (`lab_component`, `medication`, `problem`,
   `procedure`, `unit`, `provider_specialty`, `race`, `ethnicity`, `unknown`;
   `stage0_select_source_inventory_stcm.sql`) is an internal **vocab-routing**
   label — distinct from OMOP `domain_id` and from dbt's feed `default_domain_id`.
   It does **not** affect the embedding contract (the vector space is
   domain-agnostic). Open: decide whether the mapper's `source_domain` labels and
   dbt's staging/source *default domain* naming should be reconciled for
   end-to-end consistency. Investigate before changing — dbt domain routing is
   guarded by `event_domain_filter` + the `concept_in_domain` test, so renames
   there are sensitive.

9. **✅ RESOLVED — does item 6 (vocab-id casing) affect re-embedding?** No. The
   embed text is the **bare name** (§4.5), not `ehr_codes`, so vocab-id casing is
   never in the vector; and `source_id` already uppercases the vocab
   (`STCM_<UPPER(vocab)>_<md5(lower(trim(code)))>`), so the `(mapping_wave,
   source_id)` key and the GPU surrogate `concept_id` are casing-stable. Item 6 is
   a retrieval/join-correctness fix only — no re-embed. (Holds only while
   `ehr_codes` stays out of the embed text, per §4.5.)

---

## 6. Per-repo convergence checklists

**`athena_gpu_embedding`**
- [x] Emit the §4.2 stamp as the retrieval-facing version via
      `embed.retrieval_model_version` / `gpu-embed retrieval-version`
      (config-derived, engine-excluded); weights-SHA stays the store identity /
      `model_registry` provenance.
- [x] Keep FP32 + CLS defaults (already the rule); documented as the contract.
- [x] Runbook: stage the parquet, then upsert into `concept_embeddings` with the
      version column as `embed_model_version` and the embedding cast to
      `VECTOR(FLOAT, 768)`.
- [x] Runbook: real `source_concepts` MERGE on `(mapping_wave, source_id)`.

**`llm_concept_mapping`**
- [x] `EMBEDDING_QUANTIZATION=fp32` as the contract default; `.env.example` updated.
- [x] Force CLS pooling in the SapBERT loader (`_force_pooling`) so it stops
      defaulting to mean.
- [x] `model_version_for` uses the §4.2 formula (engine-excluded; `precision` from
      quantization; `pooling` added). Bumps the version → re-embed required.
- [x] GPU offload path kept + hardened: `import_source_embeddings` enforces the
      §4.2 version by default (`--allow-version-mismatch` override) (#4).
- [x] `build.py`/UDF emit a slow-fallback warning; GPU is canonical (#5).
- [x] `retrieval_code.sql` vocab compare is case-insensitive, code stays exact (#6).
- [x] `config` references `embedder` constants for model name + dimension (#7).

**`dbt_omop_clean`**
- [x] No embedding-space changes (dbt consumes only `concept_mappings`).
- [x] `int_llm_current_mappings_stcm` join unaffected: #6 was fixed on the
      retrieval side, so the natural-key (raw=raw) overlay join is unchanged.
- [x] Keep the natural-key overlay + fresh re-validation as-is (§3.3).

---

## 7. Wave / release lifecycle

Each repo is intentionally un-orchestrated; this is the expected ordering.

1. **Athena release** (monthly) → `vocabulary_loader` refreshes `OMOP_VOCAB` →
   `gpu-embedder` (re)embeds new/changed Athena concepts → load `concept_embeddings`.
2. **New mapping wave** → dbt emits unmapped STCM rows → `concept-mapper` Stage 0
   (STCM mode) ingests gaps → embed source concepts (local FP32/CLS, in-DB UDF,
   or `gpu-embedder` round-trip — all §4) → Stages 3–6.
3. **Human review** → `promote` ACCEPTED/MODIFIED → `concept_mappings`.
4. **dbt run** → overlay picks up the latest promoted rows → CDM → remaining
   gaps feed the next wave.

A `model_version` bump (any §4.1 change) invalidates **both** target and query
vectors: re-embed targets *and* clear/re-embed `source_concepts` query vectors
before retrieval is trustworthy.

---

## 8. Changing this contract

§4 is a cross-repo API. To change it:

1. Edit this file in **all three** repos in the same change set (it must stay
   byte-identical).
2. If §4.1 changes, bump `model_version` (§4.2) and plan a coordinated re-embed
   (§7) — never mix versions in one index.
3. Add/adjust a contract check that asserts (a) `gpu-embedder` export schema ⊇
   the mapper's import-required columns, and (b) both repos compute the same
   §4.2 string for the same pinned config. A failing check means the repos have
   drifted.

### Model-swap checklist (sites to change)

Swapping the model or dimension touches both repos. The `model_version` identity
is config-driven, but a few **literals** can't read a constant — change them too:

- **`athena_gpu_embedding`**: `gpu_embedder.models.EMBEDDING_DIM` (the canonical
  constant; `embed`/`store` re-use it) **and** the `FLOAT[768]` literal in
  `models.SCHEMA_DDL`. Default model: `GPU_EMBED_MODEL` / `config.model`.
- **`llm_concept_mapping`**: `embedder.SAPBERT_DIMENSION` + `DEFAULT_MODEL_NAME`
  (config references these) **and** the `VECTOR(FLOAT, 768)` literals in
  `sql/ddl/concept_embeddings.sql`, `sql/ddl/source_concepts.sql`,
  `sql/retrieval_semantic.sql`, `embeddings/build.py`, `embeddings/gpu_export.py`.
- The drift-guard tests (`test_embedder_contract.py`, `test_embed.py`) assert the
  Python constants agree; they do **not** catch a stale SQL literal — grep
  `768` after any change.

---

## 9. Upstream readiness & operational notes

Beyond the §4 vector-space contract, these affect mapping quality/cost. Not
contract inconsistencies — operational state and open work.

- **✅ Semantic retrieval is ON by default** (`SEMANTIC_RETRIEVAL_ENABLED=true`)
  now that the GPU FP32+CLS vectors are loaded. A host without the embeddings
  extra still auto-falls-back via the Stage 3 preflight.
- **✅ Vector search efficiency on Snowflake — clustered, not indexed.** Snowflake
  has no user-managed HNSW index on `VECTOR` columns; `VECTOR_COSINE_SIMILARITY`
  is brute-force over the rows surviving the WHERE. Stage 3 always filters by
  `embed_model_version` + `vocabulary_id` first, so `concept_embeddings` is now
  `CLUSTER BY (embed_model_version, vocabulary_id)` — micro-partition pruning
  limits the cosine to the relevant vocab+version subset. **Existing tables need a
  one-time `ALTER TABLE {mapping_schema}.concept_embeddings CLUSTER BY
  (embed_model_version, vocabulary_id);`** (the DDL only clusters new tables).
  Revisit a native vector index when Snowflake GAs one.
- **✅ Per-term retrieval batched.** `retrieval_lexical.sql` / `retrieval_synonym.sql`
  now score ALL search terms in ONE round trip (cross-join the term array, top_k
  per term via `QUALIFY`), replacing the N-queries-per-source loop. Same candidate
  set, far fewer warehouse round trips. (Scoring math itself was never the
  bottleneck — it is trivial pure-Python over ~15–60 candidates.)
- **🟡 Evaluate the Stage 2 normalize operation.** Normalize (`--with-stage2-
  normalize`) expands cryptic source names into search terms/synonyms and feeds
  the query embedding; it materially affects recall on terse Epic names but is
  opt-in. Evaluate cost/quality before making it default.
- **🟡 Athena release parity across the 4 repos (note for the future).** `gpu-
  embedder` embeds `CONCEPT.csv`, the mapper queries `OMOP_VOCAB`, dbt consumes
  it, `vocabulary_loader` loads it. `retrieval_semantic.sql` filters
  `standard_concept='S'` on both `concept_embeddings` and live `concept`; a
  concept embedded as standard but demoted in a newer load silently drops. Keep
  all consumers on one Athena release. (Currently aligned.)
- **🟡 `unknown` vocab-routing sink — observability + data-driven routing.** A
  source whose vocabulary is not in the stage-0 CASE routes to
  `source_domain='unknown'`; `vocab_routing` (config/scoring.yaml) has no
  `unknown` entry, so it retrieves nothing and lands Tier D (unmappable) for
  review — the intended explicit sink (no silent misrouting). Status: (a) **✅
  observability shipped** — `concept-mapper diagnostics unknown-sink` rolls up
  unknown-domain sources by `source_vocabulary_id` so the sink is a triage
  backlog; (b) **✅ done** — the vocab→domain map is data-driven
  (`config/scoring.yaml` `source_vocab_domains`, rendered into the stage-0 CASE via
  the `{{ source_vocab_domain_case }}` token), so onboarding a vocab is a one-line
  config edit, not a SQL change; a `PipelineConfig` validator fails fast if any
  routed domain lacks a `vocab_routing` entry (the silent-sink guard). The
  remaining future option is a deterministic **feed-of-origin → domain** signal
  (§10 option 1) once source-feed provenance is carried into `source_concepts`;
  (c) **✅ done** — unknown sources now get a broad,
  always-reviewed fallback retrieval that infers the domain (see §10), instead of
  being dropped.

---

## 10. Inferring a domain for `unknown` sources (so they aren't dropped)

> **✅ Implemented: options 1 + 2.**
> - **Option 2 (semantic bootstrap):** Stage 3 runs `retrieval_semantic_unrestricted.sql`
>   (broad cosine, no vocab/class filter) for unknown-vocab sources; Stage 5 flags
>   `DOMAIN_INFERRED` and caps Tier C. Toggle: `infer_unknown_source_domain`.
> - **Option 1 (feed of origin), mapper side:** `source_concepts.source_feed`
>   provenance is captured at Stage 0 (CDM = originating table, e.g. `measurement`;
>   STCM = the OMOP `source_domain_id` dbt exports on `source_to_concept_map`, e.g.
>   `Device`/`Visit`). `config/scoring.yaml` `source_feed_domains` maps feed → domain
>   (OMOP-domain-keyed for STCM, table-keyed for CDM); `_vocab_ids_for` prefers the
>   feed-routed vocab (deterministic) over the semantic bootstrap for unknown
>   sources, and Stage 5 labels feed-resolved picks distinctly from semantic-bootstrap
>   ones. **Implemented + live-verified — see §11.** Needs no dbt schema change and
>   reuses existing embeddings (source_id / source_name / grain unchanged).
> - Also fixed: `SourceConcept.source_domain` now accepts `'unknown'` — previously
>   an unrecognized vocab would have failed Stage 0 validation, so the whole
>   unknown-handling path could not actually ingest. (option 3 — LLM domain
>   classification — remains a future enhancement.)
>
> Known limitation: `import-source-embeddings --upsert-sources` re-inserts source
> rows without `source_feed` (NULL) — the normal embedding round-trip leaves it
> intact; only that reconstruct-from-file path drops it.

Today `unknown` → no candidate vocab → no candidates → Tier D. To salvage these
without reintroducing the silent misrouting the sink was built to prevent, infer
a *retrieval* domain (always human-reviewed), cheapest signal first:

1. **Deterministic (✅ implemented; dbt already exports the signal).** The
   source-feed domain is already in `source_to_concept_map.source_domain_id`
   (`unmapped_codes` sets `ac.domain_id`; `all_source_codes` members set it as a
   literal OMOP domain per feed). So **no dbt companion object is needed** — Stage 0
   STCM reads that column directly into `source_feed` (superseding the old `'stcm'`
   sentinel), and the feed map is an **OMOP-domain → routing-domain map** (§11).
   Reuses existing embeddings: `source_id` (= `STCM_<UPPER(vocab)>_<md5(...)>`),
   `source_name`, and the source grain are unchanged — reading a column already on
   rows already ingested.
2. **Semantic bootstrap (reuses the FP32+CLS vectors).** Run an *unrestricted*
   top-K cosine over `concept_embeddings` (no vocab filter), take the plurality
   OMOP `domain_id` of the hits as the inferred domain, then run the normal
   constrained retrieval there (or just keep the broad hits as candidates). One
   extra cosine query, no LLM — a natural payoff of the dense-retrieval work.
3. **LLM classification.** Fold a "which OMOP domain?" call into Stage 2 (the LLM
   already runs there). Most accurate per item; cost scales with the unknown set
   (small). Classifying a domain — not inventing a concept_id — stays within the
   "LLM proposes, expert decides" safety rule.

Guards: flag inferred-domain proposals (e.g. `DOMAIN_INFERRED`), cap them at Tier
C (never auto-accept), and keep the explicit sink for the truly unmappable.

**Key reason this is low-risk:** the inferred domain only needs to get retrieval
into the right *neighborhood*. dbt's `event_domain_filter` re-derives the FINAL
CDM domain from the chosen **standard concept's** domain (target → source →
default), so a roughly-right inference that surfaces the correct concept still
lands the row in the correct CDM table. Inference drives recall, not final
placement.

---

## 11. Consume `source_domain_id` — implemented, verified

Status: **implemented + live-verified.** The Stage 0 STCM `source_feed` is now the
OMOP `source_domain_id` dbt exports on `source_to_concept_map`, and the feed map is
keyed by OMOP domain. This supersedes the `'stcm'` sentinel + CDM-table-only
`source_feed_domains` that previously shipped.

### 11.1 What shipped (mapper-only; no dbt schema change)
- **Stage 0 STCM SQL** (`stage0_select_source_inventory_stcm.sql`): `source_feed =
  COALESCE(NULLIF(TRIM(stcm.source_domain_id), ''), 'unknown')` instead of the
  `'stcm'` literal. `source_id` / `source_name` / grain unchanged → **existing
  embeddings reused, no re-embed.** The `COALESCE` guards the rare NULL/blank so the
  feed is never empty (it resolves to no domain → semantic bootstrap, §10 opt 2).
- **Feed map** (`source_feed_domains` in `config/scoring.yaml` +
  `DEFAULT_SOURCE_FEED_DOMAINS`): reworked to an **OMOP-domain → routing-domain** map
  (the values `source_domain_id` actually takes), retaining the CDM-table keys whose
  lowercase form differs (`drug_exposure`, `visit_occurrence`, …) so the CDM-mode
  inventory path keeps working. `Measurement`/`Observation` serve both namespaces.
- **New `vocab_routing` entries**: `visit`, `device`, `route`, `meas_value`, `note`
  (+ added to the `SourceConcept.source_domain` Literal so Stage 0 ingest accepts
  them).
- **CDM CASE corrected** (`stage0_select_source_inventory_cdm.sql`):
  `visit_occurrence → visit`, `device_exposure → device` (were `problem` / `procedure`).
- **Stage 5 inference is now two-mode** (`build_proposal`): an unknown-vocab source
  whose feed resolves to a routed domain is **feed-resolved** (deterministic) — it is
  scored against that domain's preferred vocab, the domain-mismatch check runs, and
  the `DOMAIN_INFERRED` flag names the feed of origin. Only a source with *no* feed
  signal falls to the **semantic bootstrap** (domain check suppressed). Both modes
  still cap the tier at C, so a non-standard vocabulary always gets reviewer eyes.

### 11.2 Final OMOP-domain → routing map (locked; verified against live `concept`)
| `source_domain_id` (OMOP) | routing domain | preferred vocab (fallbacks) |
|---|---|---|
| Drug | medication | RxNorm |
| Measurement | lab_component | LOINC |
| Condition | problem | SNOMED |
| Procedure | procedure | CPT4 |
| Observation | flowsheet_categorical | SNOMED |
| Unit | unit | UCUM |
| Provider | provider_specialty | ABMS (NUCC, Medicare Specialty) |
| Race / Ethnicity | race / ethnicity | Race / Ethnicity |
| **Visit** | **visit** | **Visit** (CMS Place of Service, UB04) — *was wrongly → problem* |
| **Device** | **device** | **SNOMED** (HCPCS, NDC) — *was wrongly → procedure* |
| **Route** | **route** | **SNOMED** |
| **Meas Value** | **meas_value** | **SNOMED** |
| **Note** | **note** | **LOINC** |

`visit`/`device` correct genuinely wrong defaults: visits belong in the OMOP **Visit**
domain (the visit kind / `visit_concept_id`), devices in the OMOP **Device** domain
(SNOMED device hierarchy + HCPCS DME/supply + NDC device-class drugs; UDI/GUDID
populates `unique_device_id`, not `device_concept_id`). `Route`/`Meas Value` are
SNOMED (the OMOP standard for administration route and qualitative value-as-concept).
`Note` is low-volume (1 row at verification) but routed so it is never silently
dropped to the `unknown` sink.

> **FUTURE (PCORnet / PEDSnet):** the Visit (and other) routings currently target the
> CHOA OMOP/CDM feeds. As PCORnet and PEDSnet feeds are onboarded we will want to add
> their visit/encounter-type and other domain vocabularies as fallbacks. This is a
> config edit (`vocab_routing[...].fallback_vocab` + `source_*_domains`), not a code
> change. Tracked here so it isn't lost.

### 11.3 Live verification (run on `dev`, aggregate-only, PHI-safe) — RESULTS
- **NULL-domain coverage**: `source_domain_id` is **never NULL/blank** for any
  unmapped vocabulary (`null_domain = 0` across the board). The `COALESCE → 'unknown'`
  guard is therefore belt-and-suspenders, not load-bearing.
- **Observed source domains** (unmapped, `target_concept_id = 0`):
  Device 244,016 · Measurement 241,857 · Procedure 69,027 · Drug 48,334 ·
  Provider 873 · Unit 827 · Visit 374 · Route 259 · Observation 234 · Condition 48 ·
  Ethnicity 33 · Meas Value 30 · Race 21 · Note 1. **All 14 are covered by 11.2.**
- **Visit/Device standard vocabularies** (`concept`, `standard_concept='S'`): Device
  standard concepts live in **SNOMED + NDC + HCPCS**; Visit in **Visit + CMS Place of
  Service + UB04 Typ bill** — confirming the 11.2 fallbacks.

### 11.4 dbt fix shipped alongside (RxNorm false-unmapped)
`rxnorm_brand_priority.sql` left `priority` NULL for standard RxNorm targets with no
brand term-type rank (e.g. Clinical Drug). NULL `priority` both failed the `not_null`
test and silently dropped genuinely-mapped codes from `source_to_concept_map` (the
`mapped` CTE keeps `priority = min_priority`, never true when both are NULL). Fixed
with `coalesce(pr.priority, 9999)` — unranked targets sort after every real brand term
type but still win `min_priority` when they are the only mapping for a code.

### 11.5 Staleness window (unchanged — re-run before trusting non-clinical domains)
`all_source_codes` source-domain/vocab logic was reworked on dev:
- **2026-06-23** `e96ce4a` — ethnicity/race remap (+ `_legacy`/`_pipeline` route variants).
- **2026-06-22** `b34535f`, `e05a2a5` — care-site + provider specialty mapping.
- ≤ 2026-06-10 — clinical domains stable (drug/measurement/condition/procedure/LOINC/UCUM/NDC/ICD).

An STCM snapshot from ~2026-06-23 or earlier may carry **stale race / ethnicity /
provider-specialty / care-site-specialty** domains+vocab. Re-run
`dbt build --select +all_source_codes +source_to_concept_map` (with the intended
`race_mapping_route` / `ethnicity_mapping_route` vars) before trusting those four.
Clinical domains are unaffected.

### 11.6 Remaining live step (operator)
The mapper code is in place; the only outstanding action is operational: **re-run the
STCM** (`dbt build --select +all_source_codes +source_to_concept_map` post-staleness)
and **re-run Stage 0 STCM ingest** so `source_concepts.source_feed` is backfilled with
the real `source_domain_id` (existing tables also need the one-time
`ALTER TABLE … ADD COLUMN source_feed VARCHAR;` noted in the DDL). Embedding reuse is
expected to be ~100% for unchanged codes (same `source_id` hash) — only genuinely new
codes embed.
