"""Shared pytest fixtures and reporting hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_payload() -> dict:
    """Realistic /predict request body."""
    return json.loads((ROOT / "tests" / "sample_request.json").read_text())


@pytest.fixture
def preserve_production_model():
    """Back up the real ``heart_pipeline.joblib`` before a test that overwrites
    it (API tests train a tiny toy model for isolation), then restore it
    afterwards so the API keeps serving the genuinely trained model.
    """
    from src.config import MODEL_FILENAME, MODELS_DIR
    from src.models import predict as predict_mod

    path = MODELS_DIR / MODEL_FILENAME
    backup = path.with_suffix(".joblib.bak")
    had_original = path.exists()
    if had_original:
        if backup.exists():
            backup.unlink()
        path.replace(backup)
    try:
        yield
    finally:
        if path.exists():
            path.unlink()
        if had_original and backup.exists():
            backup.replace(path)
        predict_mod.load_model.cache_clear()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a plain-English one-line summary at the very end of the run."""
    tr = terminalreporter
    passed = len(tr.stats.get("passed", []))
    failed = len(tr.stats.get("failed", []))
    errors = len(tr.stats.get("error", []))
    skipped = len(tr.stats.get("skipped", []))
    total = passed + failed + errors + skipped
    duration = tr._sessionstarttime and (__import__("time").time() - tr._sessionstarttime) or 0.0

    tr.write_sep("=", "TEST RESULT SUMMARY", bold=True)
    if failed == 0 and errors == 0:
        tr.write_line(
            f"  RESULT : ALL TESTS PASSED  ({passed} of {total} succeeded in {duration:.2f}s)",
            green=True,
            bold=True,
        )
    else:
        tr.write_line(
            f"  RESULT : {failed} FAILED, {errors} ERRORS, {passed} passed, "
            f"{skipped} skipped  (out of {total} in {duration:.2f}s)",
            red=True,
            bold=True,
        )
    tr.write_sep("=", bold=True)
