"""Shared pytest fixtures: loads the official repo test case (offline, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_case  # noqa: E402
from src.inference import get_device  # noqa: E402


@pytest.fixture(scope="session")
def official_case():
    """Case 10032_1000032 shipped with the official model repo (.mha + reference outputs)."""
    return load_case("10032_1000032")


@pytest.fixture(scope="session")
def device():
    name, _note = get_device()
    return name


@pytest.fixture(scope="session")
def loaded_models(device):
    """All 5 folds (the official configuration), loaded once for the test session."""
    from src.inference import load_models
    return load_models(device=device)
