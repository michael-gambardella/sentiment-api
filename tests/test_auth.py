"""Tests for API key authentication and rate-limit error response shape."""
import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.errors import rate_limit_exceeded_handler
from api.main import app
from config import settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_enabled():
    """Temporarily enable API key auth with a known test key, then restore."""
    original = settings.api_keys
    settings.api_keys = frozenset({"test-key-abc"})
    yield "test-key-abc"
    settings.api_keys = original


# --- Auth disabled (default) ---

def test_predict_no_key_auth_disabled(client):
    """With empty api_keys (default), /predict works without any key."""
    response = client.post("/predict", json={"text": "Great film!"})
    assert response.status_code == status.HTTP_200_OK


def test_health_no_key_auth_disabled(client):
    """/health is never gated by auth."""
    assert client.get("/health").status_code == status.HTTP_200_OK


def test_metrics_no_key_auth_disabled(client):
    """/metrics is never gated by auth."""
    assert client.get("/metrics").status_code == status.HTTP_200_OK


# --- Auth enabled ---

def test_predict_missing_key_returns_401(client, auth_enabled):
    """No X-API-Key header returns 401 when auth is enabled."""
    response = client.post("/predict", json={"text": "Great film!"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_predict_wrong_key_returns_401(client, auth_enabled):
    """An incorrect X-API-Key returns 401."""
    response = client.post(
        "/predict",
        json={"text": "Great film!"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_predict_valid_key_returns_200(client, auth_enabled):
    """A correct X-API-Key returns 200."""
    response = client.post(
        "/predict",
        json={"text": "Great film!"},
        headers={"X-API-Key": auth_enabled},
    )
    assert response.status_code == status.HTTP_200_OK


def test_auth_error_body_has_error_field(client, auth_enabled):
    """401 response follows the standard {error, message} envelope."""
    data = client.post("/predict", json={"text": "Great film!"}).json()
    assert data["error"] == "AuthenticationError"
    assert "message" in data


def test_auth_error_body_has_no_unexpected_fields(client, auth_enabled):
    """401 response should only contain error and message (no detail)."""
    data = client.post("/predict", json={"text": "Great film!"}).json()
    assert set(data.keys()) == {"error", "message"}


# --- Rate limit error shape (handler unit test) ---

def test_rate_limit_handler_returns_429():
    """rate_limit_exceeded_handler produces a 429 JSONResponse."""
    exc = MagicMock()
    exc.detail = "1 per 1 minute"
    exc.headers = {"Retry-After": "60"}
    request = MagicMock()

    response = asyncio.run(rate_limit_exceeded_handler(request, exc))
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


def test_rate_limit_handler_body_shape():
    """429 response follows the standard {error, message} envelope."""
    exc = MagicMock()
    exc.detail = "1 per 1 minute"
    exc.headers = None
    request = MagicMock()

    response = asyncio.run(rate_limit_exceeded_handler(request, exc))
    body = json.loads(response.body)
    assert body["error"] == "RateLimitExceeded"
    assert "message" in body


def test_rate_limit_handler_propagates_retry_after():
    """Retry-After header from slowapi is forwarded to the client."""
    exc = MagicMock()
    exc.detail = "60 per 1 minute"
    exc.headers = {"Retry-After": "42"}
    request = MagicMock()

    response = asyncio.run(rate_limit_exceeded_handler(request, exc))
    assert response.headers.get("retry-after") == "42"
