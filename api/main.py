"""FastAPI application exposing /predict, /health, and /metrics endpoints."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from api.errors import (
    InvalidInputError,
    ModelNotLoadedError,
    PredictionError,
    invalid_input_handler,
    model_not_loaded_handler,
    prediction_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from api.predictor import Predictor, ARTIFACTS_DIR, LABELS
from api.schemas import HealthResponse, MetricsResponse, PredictRequest, PredictResponse
from config import settings
from data.pipeline import MODEL_NAME, MAX_LENGTH

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release resources on shutdown."""
    logger.info("Starting up — loading predictor...")
    try:
        app.state.predictor = Predictor()
        logger.info("Predictor loaded and ready.")
    except ModelNotLoadedError as exc:
        # Start in a degraded state so /health can report model_loaded=false
        # rather than refusing connections entirely.
        logger.error("Predictor failed to load: %s", exc)
        app.state.predictor = None
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Classification API",
    description="Classifies text sentiment as POSITIVE or NEGATIVE using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(InvalidInputError, invalid_input_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Classify the sentiment of the provided text.

    Returns a label (POSITIVE or NEGATIVE) and a confidence score between 0 and 1.
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
        model_name=MODEL_NAME,
        artifact_path=str(ARTIFACTS_DIR),
        max_input_length=MAX_LENGTH,
        labels=LABELS,
    )
