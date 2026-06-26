"""Unit tests for the optional AWS execution path.

These tests never touch the network: S3 and AWS Batch are replaced with
in-memory fakes implementing the :class:`ObjectStore` / :class:`JobScheduler`
protocols, and embedding is replaced with a deterministic fake ``embed_fn``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from gpu_embedder.aws import artifacts, orchestrate, sharding
from gpu_embedder.aws.artifacts import RunManifest
from gpu_embedder.aws.batch import JobScheduler, JobStatus
from gpu_embedder.aws.config import AwsConfig
from gpu_embedder.aws.s3 import ObjectStore
from gpu_embedder.models import ConceptRow, EmbeddedRow

EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeObjectStore:
    """In-memory ObjectStore: keys -> bytes."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local_path: Path, key: str) -> None:
        self.objects[key] = local_path.read_bytes()

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[key])

    def put_text(self, key: str, text: str) -> None:
        self.objects[key] = text.encode("utf-8")

    def get_text(self, key: str) -> str:
        return self.objects[key].decode("utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self.objects if k.startswith(prefix)]


class FakeScheduler:
    """In-memory JobScheduler that records submissions."""

    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def submit_array_job(
        self,
        *,
        job_name: str,
        job_queue: str,
        job_definition: str,
        array_size: int,
        command: list[str],
        environment: dict[str, str],
    ) -> str:
        self.submissions.append(
            {
                "job_name": job_name,
                "job_queue": job_queue,
                "job_definition": job_definition,
                "array_size": array_size,
                "command": command,
                "environment": environment,
            }
        )
        return f"job-{len(self.submissions)}"

    def describe_job(self, job_id: str) -> JobStatus:
        return JobStatus(job_id=job_id, status="SUCCEEDED", array_size=1)


def _concept(concept_id: int, vocab: str = "SNOMED") -> ConceptRow:
    return ConceptRow(
        concept_id=concept_id,
        concept_name=f"Concept {concept_id}",
        domain_id="Condition",
        vocabulary_id=vocab,
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code=str(concept_id),
        invalid_reason=None,
    )


def _fake_embed_fn(rows: list[ConceptRow], manifest: RunManifest) -> list[EmbeddedRow]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    version = manifest.model_version or "computed-v1"
    return [
        EmbeddedRow(
            concept=row,
            embedding=[float(row.concept_id % 7)] * EMBED_DIM,
            embed_text=row.concept_name,
            model_version=version,
            embedded_at=now,
        )
        for row in rows
    ]


def _cfg(**overrides: object) -> AwsConfig:
    base: dict[str, object] = {
        "s3_bucket": "test-bucket",
        "job_queue": "test-queue",
        "job_definition": "test-def",
        "shard_size": 2,
    }
    base.update(overrides)
    return AwsConfig(_env_file=None, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol conformance — fakes satisfy the real protocols
# ---------------------------------------------------------------------------


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeObjectStore(), ObjectStore)
    assert isinstance(FakeScheduler(), JobScheduler)


# ---------------------------------------------------------------------------
# AwsConfig
# ---------------------------------------------------------------------------


class TestAwsConfig:
    def test_defaults(self) -> None:
        cfg = AwsConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.environment == "academic-dev"
        assert cfg.s3_prefix_root == "gpu-embed"
        # Prefixes are derived as <prefix_root>/<environment>/{input,output}.
        assert cfg.s3_input_prefix == "gpu-embed/academic-dev/input"
        assert cfg.s3_output_prefix == "gpu-embed/academic-dev/output"
        assert cfg.shard_size == 50_000
        assert cfg.embedding_dim == 768
        assert cfg.spot_preferred is True

    def test_prefixes_track_environment(self) -> None:
        cfg = AwsConfig(_env_file=None, environment="academic-prod")  # type: ignore[call-arg]
        assert cfg.s3_input_prefix == "gpu-embed/academic-prod/input"
        assert cfg.s3_output_prefix == "gpu-embed/academic-prod/output"

    def test_explicit_prefix_override(self) -> None:
        cfg = AwsConfig(_env_file=None, s3_input_prefix="custom/in")  # type: ignore[call-arg]
        assert cfg.s3_input_prefix == "custom/in"
        # The non-overridden side is still derived.
        assert cfg.s3_output_prefix == "gpu-embed/academic-dev/output"

    def test_key_builders(self) -> None:
        cfg = _cfg()
        base = "gpu-embed/academic-dev"
        assert cfg.input_key("run1", 3) == f"{base}/input/run1/shard-00003.ndjson"
        assert cfg.output_key("run1", 3) == f"{base}/output/run1/shard-00003.ndjson"
        assert cfg.manifest_key("run1") == f"{base}/input/run1/manifest.json"
        assert cfg.output_prefix_for("run1") == f"{base}/output/run1/"

    def test_require_bucket_raises(self) -> None:
        cfg = AwsConfig(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="No S3 bucket"):
            cfg.require_bucket()

    def test_invalid_shard_size(self) -> None:
        with pytest.raises(ValueError, match="shard_size"):
            AwsConfig(_env_file=None, shard_size=0)  # type: ignore[call-arg]

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPU_EMBED_AWS_S3_BUCKET", "from-env")
        cfg = AwsConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.s3_bucket == "from-env"


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


class TestSharding:
    def test_even_split(self) -> None:
        rows = [_concept(i) for i in range(5)]
        shards = sharding.make_shards(rows, 2)
        assert [len(s) for s in shards] == [2, 2, 1]

    def test_empty(self) -> None:
        assert sharding.make_shards([], 10) == []

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            sharding.make_shards([_concept(1)], 0)

    def test_order_preserved(self) -> None:
        rows = [_concept(i) for i in range(4)]
        shards = sharding.make_shards(rows, 2)
        flat = [r.concept_id for s in shards for r in s]
        assert flat == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_concept_rows_roundtrip(self, tmp_path: Path) -> None:
        rows = [_concept(i) for i in range(3)]
        path = tmp_path / "in.ndjson"
        artifacts.write_concept_rows(path, rows)
        back = artifacts.read_concept_rows(path)
        assert [r.concept_id for r in back] == [0, 1, 2]
        assert back[0].vocabulary_id == "SNOMED"

    def test_embedded_rows_roundtrip(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        rows = _fake_embed_fn([_concept(1), _concept(2)], manifest)
        path = tmp_path / "out.ndjson"
        artifacts.write_embedded_rows(path, rows)
        back = artifacts.read_embedded_rows(path)
        assert len(back) == 2
        assert len(back[0].embedding) == EMBED_DIM
        assert back[0].model_version == manifest.model_version
        assert back[0].embedded_at == rows[0].embedded_at

    def test_manifest_roundtrip(self) -> None:
        manifest = _make_manifest()
        back = RunManifest.from_json(manifest.to_json())
        assert back == manifest

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "in.ndjson"
        artifacts.write_concept_rows(path, [_concept(1)])
        with path.open("a") as fh:
            fh.write("\n\n")
        assert len(artifacts.read_concept_rows(path)) == 1

    def test_validate_bad_dimension(self) -> None:
        bad = EmbeddedRow(
            concept=_concept(1),
            embedding=[0.0] * 10,
            embed_text="x",
            model_version="v1",
            embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="dimension"):
            artifacts.validate_embedded_rows([bad], embedding_dim=EMBED_DIM)

    def test_validate_model_version_mismatch(self) -> None:
        rows = _fake_embed_fn([_concept(1)], _make_manifest())
        with pytest.raises(ValueError, match="model_version"):
            artifacts.validate_embedded_rows(
                rows, embedding_dim=EMBED_DIM, expected_model_version="other"
            )

    def test_validate_ok(self) -> None:
        rows = _fake_embed_fn([_concept(1)], _make_manifest())
        artifacts.validate_embedded_rows(
            rows, embedding_dim=EMBED_DIM, expected_model_version="v-test"
        )


def _make_manifest() -> RunManifest:
    return RunManifest(
        run_id="run-test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        model_revision=None,
        model_version="v-test",
        embedding_dim=EMBED_DIM,
        text_fields=["concept_name"],
        separator=" ",
        max_length=128,
        batch_size=256,
        num_shards=1,
        total_rows=1,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestSubmitRun:
    def test_submits_and_uploads(self) -> None:
        store = FakeObjectStore()
        scheduler = FakeScheduler()
        rows = [_concept(i) for i in range(5)]

        sub = orchestrate.submit_run(
            rows,
            cfg=_cfg(),
            store=store,
            scheduler=scheduler,
            model="m",
            model_revision=None,
            model_version=None,
            text_fields=["concept_name"],
            separator=" ",
            max_length=128,
            batch_size=256,
            run_id="run1",
        )

        assert sub.num_shards == 3  # 5 rows / shard_size 2
        assert sub.total_rows == 5
        assert sub.job_id == "job-1"
        # 3 shard inputs + 1 manifest uploaded
        assert store.objects.keys() >= {
            "gpu-embed/academic-dev/input/run1/shard-00000.ndjson",
            "gpu-embed/academic-dev/input/run1/shard-00002.ndjson",
            "gpu-embed/academic-dev/input/run1/manifest.json",
        }
        sched = scheduler.submissions[0]
        assert sched["array_size"] == 3
        assert sched["environment"]["GPU_EMBED_AWS_RUN_ID"] == "run1"

    def test_empty_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="Nothing to submit"):
            orchestrate.submit_run(
                [],
                cfg=_cfg(),
                store=FakeObjectStore(),
                scheduler=FakeScheduler(),
                model="m",
                model_revision=None,
                model_version=None,
                text_fields=["concept_name"],
                separator=" ",
                max_length=128,
                batch_size=256,
            )

    def test_missing_batch_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Batch is not configured"):
            orchestrate.submit_run(
                [_concept(1)],
                cfg=_cfg(job_queue=None),
                store=FakeObjectStore(),
                scheduler=FakeScheduler(),
                model="m",
                model_revision=None,
                model_version=None,
                text_fields=["concept_name"],
                separator=" ",
                max_length=128,
                batch_size=256,
            )

    def test_too_many_shards_raises(self) -> None:
        with pytest.raises(ValueError, match="exceed max_array_size"):
            orchestrate.submit_run(
                [_concept(i) for i in range(10)],
                cfg=_cfg(shard_size=1, max_array_size=3),
                store=FakeObjectStore(),
                scheduler=FakeScheduler(),
                model="m",
                model_revision=None,
                model_version=None,
                text_fields=["concept_name"],
                separator=" ",
                max_length=128,
                batch_size=256,
            )


class TestRunShard:
    def test_run_shard_embeds_and_uploads(self) -> None:
        store = FakeObjectStore()
        scheduler = FakeScheduler()
        cfg = _cfg()
        rows = [_concept(i) for i in range(3)]
        orchestrate.submit_run(
            rows,
            cfg=cfg,
            store=store,
            scheduler=scheduler,
            model="m",
            model_revision=None,
            model_version="v-test",
            text_fields=["concept_name"],
            separator=" ",
            max_length=128,
            batch_size=256,
            run_id="run1",
        )

        count = orchestrate.run_shard(
            run_id="run1",
            shard_index=0,
            cfg=cfg,
            store=store,
            embed_fn=_fake_embed_fn,
        )
        assert count == 2  # shard_size 2
        assert "gpu-embed/academic-dev/output/run1/shard-00000.ndjson" in store.objects


class TestCollectRun:
    def _seed_full_run(
        self, store: FakeObjectStore, cfg: AwsConfig, n: int = 5
    ) -> None:
        scheduler = FakeScheduler()
        rows = [_concept(i) for i in range(n)]
        orchestrate.submit_run(
            rows,
            cfg=cfg,
            store=store,
            scheduler=scheduler,
            model="m",
            model_revision=None,
            model_version="v-test",
            text_fields=["concept_name"],
            separator=" ",
            max_length=128,
            batch_size=256,
            run_id="run1",
        )
        for shard_index in range((n + cfg.shard_size - 1) // cfg.shard_size):
            orchestrate.run_shard(
                run_id="run1",
                shard_index=shard_index,
                cfg=cfg,
                store=store,
                embed_fn=_fake_embed_fn,
            )

    def test_collect_imports_into_duckdb(self) -> None:
        from gpu_embedder.store import count_rows, ensure_schema

        store = FakeObjectStore()
        cfg = _cfg()
        self._seed_full_run(store, cfg, n=5)

        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        summary = orchestrate.collect_run(
            run_id="run1", cfg=cfg, store=store, conn=conn
        )
        assert summary.rows_imported == 5
        assert summary.shards == 3
        assert summary.model_version == "v-test"
        assert count_rows(conn, "v-test") == 5

    def test_collect_no_outputs_raises(self) -> None:
        from gpu_embedder.store import ensure_schema

        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        with pytest.raises(ValueError, match="No output artifacts"):
            orchestrate.collect_run(
                run_id="missing", cfg=_cfg(), store=FakeObjectStore(), conn=conn
            )

    def test_collect_rejects_wrong_model_version(self) -> None:
        from gpu_embedder.store import ensure_schema

        store = FakeObjectStore()
        cfg = _cfg()
        self._seed_full_run(store, cfg, n=3)
        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        with pytest.raises(ValueError, match="model_version"):
            orchestrate.collect_run(
                run_id="run1",
                cfg=cfg,
                store=store,
                conn=conn,
                expected_model_version="not-this-one",
            )

    def test_roundtrip_submit_run_collect(self) -> None:
        """Full move-to-AWS / embed / export-back loop with fakes."""
        from gpu_embedder.store import count_rows, ensure_schema

        store = FakeObjectStore()
        cfg = _cfg(shard_size=2)
        self._seed_full_run(store, cfg, n=4)
        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        summary = orchestrate.collect_run(
            run_id="run1", cfg=cfg, store=store, conn=conn
        )
        assert summary.rows_imported == 4
        assert count_rows(conn, "v-test") == 4


class TestRunId:
    def test_new_run_id_format(self) -> None:
        rid = orchestrate.new_run_id(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
        assert rid.startswith("20260102T030405Z-")
        assert len(rid.split("-")[-1]) == 6

    def test_new_run_ids_unique(self) -> None:
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert orchestrate.new_run_id(now) != orchestrate.new_run_id(now)
