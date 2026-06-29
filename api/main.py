"""FastAPI application exposing /predict, /health, /info, and /metrics endpoints."""
import asyncio
import time
from contextlib import asynccontextmanager

import structlog
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from redis.asyncio import Redis as AsyncRedis
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import verify_api_key
from api.cache import get_cached, set_cached
from api.celery_app import celery_app
from api.db import close_db_pool, get_recent_predictions, init_db_pool, log_prediction
from api.explain import build_explainer, compute_attributions
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
from api.metrics import cache_hits_total, cache_misses_total, predictions_total
from api.middleware import CorrelationIdMiddleware
from api.predictor import LABELS, load_all_versions
from api.schemas import (
    ExplainResponse,
    HealthResponse,
    JobResponse,
    JobStatusResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    PredictionLog,
    TokenAttribution,
)
from config import settings
from logging_config import configure_logging

configure_logging(settings.environment, settings.log_level)
logger = structlog.get_logger(module=__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model versions and connect to Redis at startup."""
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

    explainers = {v: build_explainer(p) for v, p in predictors.items()}
    logger.info("SHAP explainers ready", versions=sorted(explainers))
    app.state.explainers = explainers

    redis: AsyncRedis | None = None
    if settings.cache_ttl > 0:
        try:
            redis = AsyncRedis.from_url(settings.redis_url, decode_responses=True)
            await redis.ping()
            logger.info("Redis cache connected", ttl=settings.cache_ttl)
        except Exception as exc:
            logger.warning("Redis unavailable, caching disabled", error=str(exc))
            redis = None
    app.state.redis = redis

    db = None
    if settings.database_url:
        db = await init_db_pool(settings.database_url)
    app.state.db = db

    yield

    if redis is not None:
        await redis.aclose()
    if db is not None:
        await close_db_pool(db)
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
    Identical inputs are served from the Redis cache when available.
    """
    version, predictor = _resolve_version(request.app.state.predictors, x_model_version)
    redis: AsyncRedis | None = request.app.state.redis
    db = request.app.state.db

    t0 = time.perf_counter()

    if redis is not None:
        cached = await get_cached(redis, body.text, version)
        if cached is not None:
            cache_hits_total.inc()
            latency_ms = (time.perf_counter() - t0) * 1000
            if db is not None:
                ctx = structlog.contextvars.get_contextvars()
                asyncio.create_task(log_prediction(
                    db,
                    text=body.text,
                    label=cached["label"],
                    confidence=cached["confidence"],
                    model_version=version,
                    latency_ms=latency_ms,
                    served_from_cache=True,
                    correlation_id=ctx.get("correlation_id"),
                    client_ip=request.client.host if request.client else None,
                ))
            return PredictResponse(**cached)

    cache_misses_total.inc()
    result = predictor.predict(body.text)
    latency_ms = (time.perf_counter() - t0) * 1000
    predictions_total.labels(label=result["label"]).inc()

    if redis is not None:
        await set_cached(redis, body.text, version, result, settings.cache_ttl)

    if db is not None:
        ctx = structlog.contextvars.get_contextvars()
        asyncio.create_task(log_prediction(
            db,
            text=body.text,
            label=result["label"],
            confidence=result["confidence"],
            model_version=version,
            latency_ms=latency_ms,
            served_from_cache=False,
            correlation_id=ctx.get("correlation_id"),
            client_ip=request.client.host if request.client else None,
        ))

    return PredictResponse(label=result["label"], confidence=result["confidence"])


@app.post("/predict/batch", response_model=PredictBatchResponse)
@limiter.limit(lambda: settings.rate_limit)
async def predict_batch(
    request: Request,
    body: PredictBatchRequest,
    _: None = Depends(verify_api_key),
    x_model_version: str | None = Header(default=None),
) -> PredictBatchResponse:
    """Classify the sentiment of multiple texts in a single batched forward pass.

    Only cache-miss texts incur model inference; hits are served from Redis.
    Cache checks and cache writes for all texts run concurrently.
    Results are returned in the same order as the input texts list.
    """
    version, predictor = _resolve_version(request.app.state.predictors, x_model_version)
    redis: AsyncRedis | None = request.app.state.redis
    db = request.app.state.db
    client_ip = request.client.host if request.client else None
    ctx = structlog.contextvars.get_contextvars()
    correlation_id = ctx.get("correlation_id")

    # Phase 1: concurrent cache lookups for every text
    if redis is not None:
        cache_checks: list[dict | None] = list(
            await asyncio.gather(*[get_cached(redis, t, version) for t in body.texts])
        )
    else:
        cache_checks = [None] * len(body.texts)

    uncached_indices = [i for i, r in enumerate(cache_checks) if r is None]
    uncached_texts = [body.texts[i] for i in uncached_indices]
    cached_count = len(body.texts) - len(uncached_indices)

    # Phase 2: single batched forward pass for all cache misses
    results: list[dict | None] = list(cache_checks)
    inference_ms = 0.0

    if uncached_texts:
        t_infer = time.perf_counter()
        batch_results = predictor.predict_batch(uncached_texts)
        inference_ms = (time.perf_counter() - t_infer) * 1000

        for idx, result in zip(uncached_indices, batch_results):
            results[idx] = result
            predictions_total.labels(label=result["label"]).inc()

        # Write all misses to cache concurrently
        if redis is not None:
            await asyncio.gather(*[
                set_cached(redis, uncached_texts[i], version, batch_results[i], settings.cache_ttl)
                for i in range(len(uncached_texts))
            ])

    if cached_count > 0:
        cache_hits_total.inc(cached_count)
    if uncached_texts:
        cache_misses_total.inc(len(uncached_texts))

    # Phase 3: fire-and-forget audit log — one row per text
    if db is not None:
        per_miss_latency = inference_ms / len(uncached_texts) if uncached_texts else 0.0
        for i, (text, result) in enumerate(zip(body.texts, results)):
            from_cache = cache_checks[i] is not None
            asyncio.create_task(log_prediction(
                db,
                text=text,
                label=result["label"],
                confidence=result["confidence"],
                model_version=version,
                latency_ms=0.0 if from_cache else per_miss_latency,
                served_from_cache=from_cache,
                correlation_id=correlation_id,
                client_ip=client_ip,
            ))

    return PredictBatchResponse(
        results=[PredictResponse(**r) for r in results],
        model_version=version,
        count=len(body.texts),
        cached_count=cached_count,
    )


@app.post("/predict/explain", response_model=ExplainResponse)
@limiter.limit(lambda: settings.rate_limit)
async def predict_explain(
    request: Request,
    body: PredictRequest,
    _: None = Depends(verify_api_key),
    x_model_version: str | None = Header(default=None),
) -> ExplainResponse:
    """Classify text and return SHAP token-level attribution scores.

    Attributions are computed at word-level granularity using SHAP's Partition
    explainer. Positive scores push toward the predicted label; negative scores
    push away. The additive property holds: sum(scores) + base_value ≈ confidence.

    This endpoint is significantly slower than /predict (typically 1–10 s depending
    on text length) because SHAP requires many model calls to estimate attributions.
    Suitable for debugging, model auditing, and interactive tooling — not for
    latency-sensitive production traffic.
    """
    version, predictor = _resolve_version(request.app.state.predictors, x_model_version)
    explainer = request.app.state.explainers.get(version)
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail=f"SHAP explainer not available for version '{version}'",
        )

    result = predictor.predict(body.text)

    loop = asyncio.get_running_loop()
    attributions, base_value = await loop.run_in_executor(
        None, compute_attributions, explainer, body.text, result["label"]
    )

    db = request.app.state.db
    if db is not None:
        ctx = structlog.contextvars.get_contextvars()
        asyncio.create_task(log_prediction(
            db,
            text=body.text,
            label=result["label"],
            confidence=result["confidence"],
            model_version=version,
            latency_ms=0.0,
            served_from_cache=False,
            correlation_id=ctx.get("correlation_id"),
            client_ip=request.client.host if request.client else None,
        ))

    return ExplainResponse(
        label=result["label"],
        confidence=result["confidence"],
        attributions=[TokenAttribution(**a) for a in attributions],
        base_value=base_value,
        model_version=version,
    )


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


@app.get("/predictions", response_model=list[PredictionLog])
async def list_predictions(
    request: Request,
    _: None = Depends(verify_api_key),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of rows to return."),
    version: str | None = Query(default=None, description="Filter by model version."),
) -> list[PredictionLog]:
    """Return recent prediction log entries from the audit database.

    Results are ordered newest-first. Use the version query parameter to filter
    by a specific model checkpoint.
    """
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable or not configured")
    rows = await get_recent_predictions(db, limit=limit, version=version)
    return [PredictionLog(**row) for row in rows]
