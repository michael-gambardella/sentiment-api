"""FastAPI application exposing /predict, /health, and /metrics endpoints."""
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import verify_api_key
from api.errors import (
    AuthenticationError,
    InvalidInputError,
    ModelNotLoadedError,
    PredictionError,
    authentication_error_handler,
    invalid_input_handler,
    model_not_loaded_handler,
    prediction_error_handler,
    rate_limit_exceeded_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from api.middleware import CorrelationIdMiddleware
from api.predictor import Predictor, ARTIFACTS_DIR, LABELS
from api.schemas import HealthResponse, MetricsResponse, PredictRequest, PredictResponse
from config import settings
from logging_config import configure_logging

configure_logging(settings.environment, settings.log_level)
logger = structlog.get_logger(module=__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release resources on shutdown."""
    logger.info("Starting up", environment=settings.environment, artifacts_dir=str(settings.artifacts_dir))
    try:
        app.state.predictor = Predictor()
        logger.info("Predictor ready")
    except ModelNotLoadedError as exc:
        logger.error("Predictor failed to load", error=str(exc))
        app.state.predictor = None
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Sentiment Classification API",
    description="Classifies text sentiment as POSITIVE or NEGATIVE using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(CorrelationIdMiddleware)

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(InvalidInputError, invalid_input_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(lambda: settings.rate_limit)
async def predict(
    request: Request,
    body: PredictRequest,
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """Classify the sentiment of the provided text.

    Returns a label (POSITIVE or NEGATIVE) and a confidence score between 0 and 1.
    Requires a valid X-API-Key header when API_KEYS is configured.
    """
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise ModelNotLoadedError(
            "Model is not loaded. Artifacts may be missing — run 'make train' first."
        )
    result = predictor.predict(body.text)
    return PredictResponse(label=result["label"], confidence=result["confidence"])


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check whether the service is running and the model is loaded."""
    model_loaded = request.app.state.predictor is not None
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return metadata about the loaded model."""
    return MetricsResponse(
        model_name=settings.model_name,
        artifact_path=str(ARTIFACTS_DIR),
        max_input_length=settings.max_input_length,
        labels=LABELS,
    )
