import pytest
import torch
from api.predictor import Predictor, LABELS


@pytest.fixture(scope="module")
def predictor():
    """Load the fine-tuned model once for all model tests in this module."""
    return Predictor()


def test_predict_returns_valid_label(predictor):
    result = predictor.predict("This film was absolutely wonderful.")
    assert result["label"] in LABELS


def test_predict_confidence_in_range(predictor):
    result = predictor.predict("I hated every minute of this movie.")
    assert 0.0 <= result["confidence"] <= 1.0


def test_positive_review_predicts_positive(predictor):
    result = predictor.predict("One of the best films I have ever seen. A true masterpiece.")
    assert result["label"] == "POSITIVE"


def test_negative_review_predicts_negative(predictor):
    result = predictor.predict("Terrible. Boring, poorly acted, and a complete waste of time.")
    assert result["label"] == "NEGATIVE"


def test_predict_output_keys(predictor):
    result = predictor.predict("Some text.")
    assert "label" in result
    assert "confidence" in result


def test_model_output_logit_shape(predictor):
    """Verify the model produces exactly two logits (one per class)."""
    inputs = predictor.tokenizer(
        "Test input.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to(predictor.device)
    attention_mask = inputs["attention_mask"].to(predictor.device)

    with torch.no_grad():
        outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (1, 2)
