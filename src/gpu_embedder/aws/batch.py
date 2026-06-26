"""AWS Batch job-scheduler interface and a lazy boto3-backed implementation.

As with :mod:`s3`, the :class:`JobScheduler` protocol lets orchestration submit
and poll jobs against an in-memory fake in tests and a real AWS Batch queue in
production. ``boto3`` is imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobStatus:
    """A point-in-time view of a submitted Batch job."""

    job_id: str
    status: str  # SUBMITTED | PENDING | RUNNABLE | STARTING | RUNNING | SUCCEEDED | FAILED
    array_size: int


@runtime_checkable
class JobScheduler(Protocol):
    """Minimal AWS Batch surface used by the orchestration layer."""

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
        """Submit an array job and return its job id."""
        ...

    def describe_job(self, job_id: str) -> JobStatus:
        """Return the current status of a submitted job."""
        ...


class BatchJobScheduler:
    """boto3-backed :class:`JobScheduler`."""

    def __init__(self, region: str | None = None) -> None:
        import boto3  # lazy

        self._client = boto3.client("batch", region_name=region)

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
        env_overrides = [{"name": k, "value": v} for k, v in environment.items()]
        container_overrides: dict[str, object] = {"environment": env_overrides}
        if command:
            container_overrides["command"] = command

        kwargs: dict[str, object] = {
            "jobName": job_name,
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
            "containerOverrides": container_overrides,
        }
        # AWS Batch rejects arrayProperties with size < 2; a single shard is a
        # plain (non-array) job.
        if array_size >= 2:
            kwargs["arrayProperties"] = {"size": array_size}

        logger.info(
            "Submitting Batch job %s (queue=%s, def=%s, array_size=%d)",
            job_name,
            job_queue,
            job_definition,
            array_size,
        )
        response = self._client.submit_job(**kwargs)
        return str(response["jobId"])

    def describe_job(self, job_id: str) -> JobStatus:
        response = self._client.describe_jobs(jobs=[job_id])
        jobs = response.get("jobs", [])
        if not jobs:
            raise ValueError(f"No Batch job found with id {job_id}")
        job = jobs[0]
        array_size = int(job.get("arrayProperties", {}).get("size", 1) or 1)
        return JobStatus(
            job_id=job_id, status=str(job["status"]), array_size=array_size
        )
