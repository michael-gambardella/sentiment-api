import structlog
from celery.signals import worker_process_init

from api.celery_app import celery_app

logger = structlog.get_logger(module=__name__)

_predictor = None


@worker_process_init.connect
def _load_predictor(**kwargs):
    """Load the model once per worker process at startup — not per-task."""
    global _predictor
    from api.predictor import Predictor
    _predictor = Predictor()


@celery_app.task(bind=True, name="sentiment.predict")
def predict_task(self, text: str) -> dict:
    return _predictor.predict(text)
