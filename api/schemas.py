"""Pydantic v2 request and response schemas for the sentiment classification API."""
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

_TextItem = Annotated[str, Field(min_length=1, max_length=5000)]


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


class PredictBatchRequest(BaseModel):
    texts: list[_TextItem] = Field(
        min_length=1,
        max_length=64,
        description="1–64 texts to classify. Results are returned in the same order.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["Great movie!", "Terrible experience."]}]
        }
    }


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse] = Field(description="Predictions aligned with the input texts list.")
    model_version: str = Field(description="Model version that produced the results.")
    count: int = Field(description="Total number of texts classified.")
    cached_count: int = Field(description="Number of results served from the Redis cache.")


class JobResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the queued prediction job.")
    status: str = Field(description="Initial job status (always 'queued').")


class JobStatusResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the prediction job.")
    status: str = Field(description="Job status: queued | processing | completed | failed.")
    result: PredictResponse | None = Field(default=None, description="Prediction result once completed.")
    error: str | None = Field(default=None, description="Error detail if the job failed.")


class TokenAttribution(BaseModel):
    token: str = Field(description="Surface-form word or punctuation from the input text.")
    score: float = Field(
        description="SHAP value for the predicted label. "
        "Positive → pushes toward the predicted label; negative → pushes away."
    )


class ExplainResponse(BaseModel):
    label: str = Field(description="Predicted sentiment label.")
    confidence: float = Field(description="Model confidence for the predicted label.")
    attributions: list[TokenAttribution] = Field(
        description="Per-token SHAP values in input order. "
        "sum(scores) + base_value ≈ confidence."
    )
    base_value: float = Field(
        description="Model's expected P(label) when all tokens are masked — the SHAP baseline."
    )
    model_version: str = Field(description="Model version that produced the explanation.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "POSITIVE",
                    "confidence": 0.9968,
                    "attributions": [
                        {"token": "Great", "score": 0.4231},
                        {"token": "film", "score": 0.3105},
                        {"token": "!", "score": 0.0198},
                    ],
                    "base_value": 0.2434,
                    "model_version": "final",
                }
            ]
        }
    }


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
