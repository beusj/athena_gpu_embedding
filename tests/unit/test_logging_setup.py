"""Unit tests for logging setup utility."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from gpu_embedder.logging_setup import setup_logging


def _touch(path: Path, ts: float) -> None:
    path.write_text("x", encoding="utf-8")
    path.touch()
    os.utime(path, (ts, ts))


def test_setup_logging_creates_dated_file_and_prunes_to_max_files(tmp_path: Path) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    try:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        now = datetime.now().timestamp()
        for i in range(7):
            _touch(log_dir / f"gpu-embed-2026-01-0{i + 1}.log", now - (i + 1) * 100)

        expected_name = f"gpu-embed-{datetime.now().strftime('%Y-%m-%d')}.log"
        created = setup_logging(
            verbose=False,
            log_dir=log_dir,
            max_bytes=120,
            max_files=5,
        )

        logger = logging.getLogger("gpu_embedder.test")
        for _ in range(50):
            logger.info("This is a long enough log line to trigger file rotation quickly")

        for handler in logging.getLogger().handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        files = list(log_dir.glob("gpu-embed-*.log*"))
        assert created.name == expected_name
        assert len(files) <= 5
        assert any(path.name == expected_name for path in files)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.setLevel(original_level)
        for handler in original_handlers:
            root.addHandler(handler)
