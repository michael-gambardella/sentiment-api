"""Shared pytest configuration and fixtures."""
from collections.abc import Generator

import pytest

from config import settings


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_model: skip when model artifacts are not present on disk",
    )


@pytest.fixture(autouse=True, scope="session")
def disable_auth_by_default() -> Generator[None, None, None]:
    """Run the entire test session with auth disabled (api_keys = empty).

    Tests are self-contained and must not depend on whatever API_KEYS happens
    to be set in .env. Tests that specifically exercise auth use the
    auth_enabled fixture in test_auth.py to opt in per test.
    """
    original = settings.api_keys
    settings.api_keys = ""
    yield
    settings.api_keys = original


@pytest.fixture(autouse=True)
def skip_without_artifacts(request: pytest.FixtureRequest) -> None:
    """Auto-skip any test marked requires_model when artifacts directory is absent.

    Keeps CI green: tests that need the fine-tuned weights are skipped rather than
    erroring, while all other tests (pipeline helpers, auth handler unit tests) run.
    """
    if request.node.get_closest_marker("requires_model"):
        if not settings.artifacts_dir.exists():
            pytest.skip(f"Model artifacts not found at '{settings.artifacts_dir}' — skipping")
