"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_payload() -> dict:
    """Realistic /predict request body."""
    return json.loads((ROOT / "tests" / "sample_request.json").read_text())
