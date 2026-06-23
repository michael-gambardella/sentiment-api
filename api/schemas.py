"""Pydantic v2 request and response schemas for the sentiment classification API."""
from datetime import datetime

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
    available_versions: list[str] = Field(description="Loaded model versions (pass via X-Model-Version header).")
    default_version: str = Field(description="Version used when X-Model-Version header is absent.")
    max_input_length: int = Field(description="Maximum token length accepted by the model.")
    labels: list[str] = Field(description="Possible output labels in index order.")


class JobResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the queued prediction job.")
    status: str = Field(description="Initial job status (always 'queued').")


class JobStatusResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the prediction job.")
    status: str = Field(description="Job status: queued | processing | completed | failed.")
    result: PredictResponse | None = Field(default=None, description="Prediction result once completed.")
    error: str | None = Field(default=None, description="Error detail if the job failed.")


class PredictionLog(BaseModel):
    id: int = Field(description="Auto-incrementing row identifier.")
    created_at: datetime = Field(description="UTC timestamp when the prediction was made.")
    input_hash: str = Field(description="SHA-256 of the input text.")
    label: str = Field(description="Predicted sentiment label.")
    confidence: float = Field(description="Model confidence score.")
    model_version: str = Field(description="Model version that produced the prediction.")
    latency_ms: float = Field(description="Inference time in milliseconds (0 for cache hits).")
    served_from_cache: bool = Field(description="True when the result was served from the Redis cache.")
    correlation_id: str | None = Field(description="X-Correlation-ID tied to the original request.")
    client_ip: str | None = Field(description="Originating client IP address.")
