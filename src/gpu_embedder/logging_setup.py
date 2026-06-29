"""Application logging configuration for gpu-embedder."""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _prune_log_files(log_dir: Path, max_files: int) -> None:
    log_files = sorted(
        log_dir.glob("gpu-embed-*.log*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_file in log_files[max_files:]:
        old_file.unlink(missing_ok=True)


class _PruningRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args: object, log_dir: Path, max_files: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._log_dir = log_dir
        self._max_files = max_files

    def doRollover(self) -> None:
        super().doRollover()
        _prune_log_files(self._log_dir, self._max_files)


def setup_logging(
    *,
    verbose: bool,
    log_dir: Path,
    max_bytes: int,
    max_files: int,
) -> Path:
    """Configure console + rotating file logging and return the log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"gpu-embed-{date_stamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = _PruningRotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=max(max_files - 1, 0),
        encoding="utf-8",
        log_dir=log_dir,
        max_files=max_files,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _prune_log_files(log_dir, max_files=max_files)
    return log_file
