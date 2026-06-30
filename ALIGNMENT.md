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

4. **🟠 Duplicated GPU offload paths.** `concept-mapper`'s homegrown
   `embeddings/gpu_export.py` vs the dedicated `athena_gpu_embedding` repo. **→**
   pick one canonical path; if `gpu_export.py` stays, it must emit the §4.2 stamp
   (its source columns already match — gap #B is only the version stamp).

5. **🟡 Target-embedding ownership undefined.** Mapper builds targets locally
   (`embeddings/build.py`); `gpu-embedder` also produces them. **→** designate
   `gpu-embedder` the owner of Athena target embeddings at scale; the mapper's
   `build.py`/UDF remain the fallback and MUST produce identical §4 vectors.

6. **🟡 `source_vocabulary_id` case drift (mapper-internal).** Stage 0 STCM
   **uppercases** `source_vocabulary_id` but stores raw case in `ehr_codes.system`
   (`sql/stage0_select_source_inventory_stcm.sql`); promote derives the dbt key
   from `ehr_codes[0]:system` (raw), which matches
   `all_source_codes.default_vocabulary_id` — so **C works today** but is a latent
   footgun. **→** normalize to one casing rule end-to-end.

7. **🟡 `768` / model name duplicated** with no shared source of truth across
   repos. **→** centralize per repo and cross-check via §6 + the contract test.

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
- [~] GPU offload path reconciled to the §4.2 stamp in `gpu_export.py` (defers to
      the dedicated repo; inline script uses the shared stamp + CLS). Deleting the
      duplicate path entirely is still open (#4).
- [ ] Normalize `source_vocabulary_id` casing (#6).

**`dbt_omop_clean`**
- [ ] No embedding-space changes (dbt consumes only `concept_mappings`).
- [ ] Confirm `int_llm_current_mappings_stcm` join casing stays consistent with
      whatever casing rule the mapper settles on (#6).
- [ ] Keep the natural-key overlay + fresh re-validation as-is (§3.3).

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
