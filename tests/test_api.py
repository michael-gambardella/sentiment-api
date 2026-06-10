import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import LABELS

pytestmark = pytest.mark.requires_model


@pytest.fixture(scope="module")
def client():
    """Spin up the full FastAPI app in-process once for all API tests."""
    with TestClient(app) as c:
        yield c


# --- /health ---

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded(client):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# --- /info (model metadata) ---

def test_info_returns_200(client):
    response = client.get("/info")
    assert response.status_code == 200


def test_info_response_shape(client):
    data = client.get("/info").json()
    assert "model_name" in data
    assert "artifact_path" in data
    assert "max_input_length" in data
    assert "labels" in data


def test_info_labels_match_expected(client):
    data = client.get("/info").json()
    assert data["labels"] == LABELS


# --- /metrics (Prometheus exposition format) ---

def test_prometheus_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_prometheus_metrics_content_type(client):
    response = client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_prometheus_metrics_contains_http_counter(client):
    response = client.get("/metrics")
    assert "http_requests_total" in response.text


def test_prometheus_metrics_contains_prediction_counter(client):
    """After at least one prediction, the custom counter must appear in /metrics."""
    client.post("/predict", json={"text": "Great film!"})
    response = client.get("/metrics")
    assert "sentiment_predictions_total" in response.text


# --- /predict ---

def test_predict_returns_200(client):
    response = client.post("/predict", json={"text": "This was a great movie!"})
    assert response.status_code == 200


def test_predict_response_shape(client):
    response = client.post("/predict", json={"text": "Loved it."})
    data = response.json()
    assert "label" in data
    assert "confidence" in data


def test_predict_label_is_valid(client):
    response = client.post("/predict", json={"text": "What a boring film."})
    data = response.json()
    assert data["label"] in LABELS


def test_predict_confidence_in_range(client):
    response = client.post("/predict", json={"text": "Incredible performances."})
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_text_rejected(client):
    """Pydantic should reject empty strings before they reach the model."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text_field_rejected(client):
    """Pydantic should reject payloads missing the required text field."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


# --- error response shape ---

def test_validation_error_body_has_error_field(client):
    """Our custom validation handler should return {error, message, detail}."""
    response = client.post("/predict", json={"text": ""})
    data = response.json()
    assert "error" in data
    assert data["error"] == "ValidationError"
    assert "message" in data
    assert "detail" in data


def test_validation_error_detail_is_list(client):
    """detail should expose the raw Pydantic error list for client debugging."""
    response = client.post("/predict", json={})
    data = response.json()
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0


# --- correlation IDs ---

def test_response_includes_correlation_id_header(client):
    """Every response should carry an X-Correlation-ID header."""
    response = client.get("/health")
    assert "x-correlation-id" in response.headers


def test_client_supplied_correlation_id_is_echoed(client):
    """A caller-supplied X-Correlation-ID must be preserved and returned unchanged."""
    supplied_id = "test-trace-abc-123"
    response = client.get("/health", headers={"X-Correlation-ID": supplied_id})
    assert response.headers["x-correlation-id"] == supplied_id


def test_generated_correlation_id_is_uuid(client):
    """When no ID is supplied, the generated ID should be a valid UUID4 string."""
    import uuid
    response = client.get("/health")
    correlation_id = response.headers["x-correlation-id"]
    uuid.UUID(correlation_id)  # raises ValueError if not a valid UUID


def test_predict_positive_sentiment(client):
    response = client.post("/predict", json={"text": "Absolutely loved this film. A masterpiece."})
    assert response.json()["label"] == "POSITIVE"


def test_predict_negative_sentiment(client):
    response = client.post("/predict", json={"text": "Terrible movie. Dull and painfully slow."})
    assert response.json()["label"] == "NEGATIVE"
