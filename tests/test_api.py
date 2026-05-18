import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import LABELS


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


# --- /metrics ---

def test_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_response_shape(client):
    data = client.get("/metrics").json()
    assert "model_name" in data
    assert "artifact_path" in data
    assert "max_input_length" in data
    assert "labels" in data


def test_metrics_labels_match_expected(client):
    data = client.get("/metrics").json()
    assert data["labels"] == LABELS


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


def test_predict_positive_sentiment(client):
    response = client.post("/predict", json={"text": "Absolutely loved this film. A masterpiece."})
    assert response.json()["label"] == "POSITIVE"


def test_predict_negative_sentiment(client):
    response = client.post("/predict", json={"text": "Terrible movie. Dull and painfully slow."})
    assert response.json()["label"] == "NEGATIVE"
