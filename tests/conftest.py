"""Root conftest: register the gpu mark and auto-skip GPU tests on CPU machines."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU — auto-skipped when unavailable"
    )


def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
