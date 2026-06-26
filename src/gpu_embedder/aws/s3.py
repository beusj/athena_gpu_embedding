"""S3 object-store interface and a lazy boto3-backed implementation.

The :class:`ObjectStore` protocol is the seam that keeps the orchestration layer
testable without a live AWS account: unit tests pass an in-memory fake, while
production uses :class:`S3ObjectStore`. ``boto3`` is imported lazily so importing
this module (and the rest of ``gpu_embedder``) never requires the AWS SDK.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal object-store surface used by the AWS orchestration layer."""

    def upload_file(self, local_path: Path, key: str) -> None: ...

    def download_file(self, key: str, local_path: Path) -> None: ...

    def put_text(self, key: str, text: str) -> None: ...

    def get_text(self, key: str) -> str: ...

    def list_keys(self, prefix: str) -> list[str]: ...


class S3ObjectStore:
    """boto3-backed :class:`ObjectStore` for a single bucket."""

    def __init__(self, bucket: str, region: str | None = None) -> None:
        import boto3  # lazy: only required when actually talking to AWS

        self.bucket = bucket
        self._client = boto3.client("s3", region_name=region)

    def upload_file(self, local_path: Path, key: str) -> None:
        logger.info("Uploading %s -> s3://%s/%s", local_path, self.bucket, key)
        self._client.upload_file(str(local_path), self.bucket, key)

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s -> %s", self.bucket, key, local_path)
        self._client.download_file(self.bucket, key, str(local_path))

    def put_text(self, key: str, text: str) -> None:
        logger.info("Putting text object s3://%s/%s", self.bucket, key)
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=text.encode("utf-8")
        )

    def get_text(self, key: str) -> str:
        logger.info("Getting text object s3://%s/%s", self.bucket, key)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        body: bytes = response["Body"].read()
        return body.decode("utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        logger.info("Listing s3://%s/%s", self.bucket, prefix)
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
