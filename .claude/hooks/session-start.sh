#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Provisions a GPU-free environment so the unit tests and linters run in these
# sessions. The test suite mocks the model (no real forward pass), so a
# CPU-only torch is sufficient — see "Testing without a GPU" in the README.
#
# Why not `uv sync --extra cpu`? These sandboxes can reach PyPI but NOT
# download.pytorch.org, and the `cpu`/`gpu` extras route torch to a
# download.pytorch.org index (blocked here). So we install a CPU-capable torch
# straight from PyPI, then install the project with --no-deps plus the remaining
# runtime + dev dependencies (all available on PyPI).
set -euo pipefail

# Only run in the remote (Claude web) environment. Local developers should use
# `uv sync --group dev --extra cpu` (or `--extra gpu`) instead.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Create the venv if it doesn't already exist (idempotent across sessions).
[ -d .venv ] || uv venv --python 3.12

# CPU-capable torch from PyPI (reachable). Large, but runs fine on CPU; CUDA is
# never exercised because tests mock the model.
uv pip install "torch>=2.3"

# Install the project without re-resolving deps (avoids the blocked torch index),
# then the runtime + dev dependencies explicitly from PyPI. `uv pip install` is
# idempotent, so re-running on a warm container is cheap.
uv pip install --no-deps -e .
uv pip install \
  duckdb typer pydantic pydantic-settings python-dotenv transformers tqdm numpy pyarrow \
  pytest pytest-cov pytest-asyncio ruff mypy
