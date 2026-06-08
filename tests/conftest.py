"""Shared pytest configuration and fixtures."""
from collections.abc import Generator

import pytest

from config import settings


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_model: skip when model artifacts are not present on disk",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Add skip markers at collection time so module-scoped fixtures are never set up.

    A function-scoped autouse fixture can't intercept module-scoped fixture errors —
    the module fixture (e.g. Predictor()) raises before the function fixture runs.
    Marking at collection time tells pytest to skip the test entirely, including setup.
    """
    if not settings.artifacts_dir.exists():
        skip_marker = pytest.mark.skip(
            reason=f"Model artifacts not found at '{settings.artifacts_dir}' — skipping"
        )
        for item in items:
            if item.get_closest_marker("requires_model"):
                item.add_marker(skip_marker)


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
