"""Pydantic v2 request and response schemas for the sentiment classification API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="The text to classify.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "This movie was absolutely fantastic!"}]
        }
    }


class PredictResponse(BaseModel):
    label: str = Field(description="Predicted sentiment: POSITIVE or NEGATIVE.")
    confidence: float = Field(description="Model confidence score between 0 and 1.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"label": "POSITIVE", "confidence": 0.97}]
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(description="Service status.")
    model_loaded: bool = Field(description="Whether the model is ready to serve predictions.")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(description="Base model identifier.")
    artifact_path: str = Field(description="Path the model was loaded from.")
    max_input_length: int = Field(description="Maximum token length accepted by the model.")
    labels: list[str] = Field(description="Possible output labels in index order.")
