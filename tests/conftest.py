"""Shared fixtures for Phase-2 tests (repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.paths import SPLITS_DIR  # noqa: E402


@pytest.fixture(scope="session")
def selection_manifest_path():
    p = SPLITS_DIR / "case_selection_seed2026.json"
    if not p.exists():
        pytest.skip("selection manifest missing (run scripts/prepare_data.py --select-only)")
    return p
