import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.predictor import Predictor, ARTIFACTS_DIR, LABELS
from api.schemas import HealthResponse, MetricsResponse, PredictRequest, PredictResponse
from data.pipeline import MODEL_NAME, MAX_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release resources on shutdown."""
    logger.info("Starting up — loading predictor...")
    app.state.predictor = Predictor()
    logger.info("Predictor loaded and ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Classification API",
    description="Classifies text sentiment as POSITIVE or NEGATIVE using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Classify the sentiment of the provided text.

    Returns a label (POSITIVE or NEGATIVE) and a confidence score between 0 and 1.
    """
    predictor: Predictor = request.app.state.predictor
    result = predictor.predict(body.text)
    return PredictResponse(label=result["label"], confidence=result["confidence"])


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check whether the service is running and the model is loaded."""
    model_loaded = hasattr(request.app.state, "predictor")
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
