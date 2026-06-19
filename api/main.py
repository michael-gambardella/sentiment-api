"""FastAPI application exposing /predict, /health, /info, and /metrics endpoints."""
from contextlib import asynccontextmanager

import structlog
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import verify_api_key
from api.celery_app import celery_app
from api.tasks import predict_task
from api.tracing import configure_tracing
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
from api.metrics import predictions_total
from api.middleware import CorrelationIdMiddleware
from api.predictor import LABELS, load_all_versions
from api.schemas import HealthResponse, JobResponse, JobStatusResponse, ModelInfoResponse, PredictRequest, PredictResponse
from config import settings
from logging_config import configure_logging

configure_logging(settings.environment, settings.log_level)
logger = structlog.get_logger(module=__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model versions at startup; store in app.state.predictors dict."""
    logger.info("Starting up", environment=settings.environment, versions_dir=str(settings.model_versions_dir))
    predictors = load_all_versions(settings.model_versions_dir)
    if not predictors:
        logger.error("No model versions loaded", versions_dir=str(settings.model_versions_dir))
    else:
        logger.info("Model versions ready", versions=sorted(predictors), default=settings.default_model_version)
        if settings.default_model_version not in predictors:
            logger.warning(
                "Default version not found in loaded versions",
                default=settings.default_model_version,
                available=sorted(predictors),
            )
    app.state.predictors = predictors
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

Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app)

configure_tracing(app)

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(InvalidInputError, invalid_input_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


def _resolve_version(predictors: dict, x_model_version: str | None) -> tuple[str, object]:
    """Return (version_name, predictor) or raise 404 if the version is unknown."""
    version = x_model_version or settings.default_model_version
    predictor = predictors.get(version)
    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{version}' not found. Available: {sorted(predictors)}",
        )
    return version, predictor


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(lambda: settings.rate_limit)
async def predict(
    request: Request,
    body: PredictRequest,
    _: None = Depends(verify_api_key),
    x_model_version: str | None = Header(default=None),
) -> PredictResponse:
    """Classify the sentiment of the provided text.

    Returns a label (POSITIVE or NEGATIVE) and a confidence score between 0 and 1.
    Pass X-Model-Version to target a specific checkpoint; omit to use the default.
    """
    _, predictor = _resolve_version(request.app.state.predictors, x_model_version)
    result = predictor.predict(body.text)
    predictions_total.labels(label=result["label"]).inc()
    return PredictResponse(label=result["label"], confidence=result["confidence"])


@app.post("/predict/async", response_model=JobResponse, status_code=202)
@limiter.limit(lambda: settings.rate_limit)
async def predict_async(
    request: Request,
    body: PredictRequest,
    _: None = Depends(verify_api_key),
    x_model_version: str | None = Header(default=None),
) -> JobResponse:
    """Submit a prediction job and return a job ID immediately.

    Poll GET /predict/{job_id} to retrieve the result once processing completes.
    Pass X-Model-Version to target a specific checkpoint; omit to use the default.
    Returns 503 if the worker queue (Redis) is unavailable.
    """
    version, _ = _resolve_version(request.app.state.predictors, x_model_version)
    try:
        task = predict_task.delay(body.text, version)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Worker queue unavailable") from exc
    return JobResponse(job_id=task.id, status="queued")


@app.get("/predict/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    _: None = Depends(verify_api_key),
) -> JobStatusResponse:
    """Poll for the result of an async prediction job.

    Status values: queued → processing → completed | failed.
    Returns 503 if the result backend (Redis) is unavailable.
    """
    try:
        result = AsyncResult(job_id, app=celery_app)
        state = result.state
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Worker queue unavailable") from exc

    if state == "STARTED":
        return JobStatusResponse(job_id=job_id, status="processing")
    if state == "SUCCESS":
        prediction = result.result
        return JobStatusResponse(
            job_id=job_id,
            status="completed",
            result=PredictResponse(label=prediction["label"], confidence=prediction["confidence"]),
        )
    if state == "FAILURE":
        return JobStatusResponse(job_id=job_id, status="failed", error=str(result.result))
    return JobStatusResponse(job_id=job_id, status="queued")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check whether the service is running and at least one model version is loaded."""
    return HealthResponse(status="ok", model_loaded=bool(request.app.state.predictors))


@app.get("/info", response_model=ModelInfoResponse)
async def info(request: Request) -> ModelInfoResponse:
    """Return metadata about the loaded model versions."""
    predictors: dict = request.app.state.predictors
    return ModelInfoResponse(
        model_name=settings.model_name,
        available_versions=sorted(predictors),
        default_version=settings.default_model_version,
        max_input_length=settings.max_input_length,
        labels=LABELS,
    )
