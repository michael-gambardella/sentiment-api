import structlog
from celery.signals import worker_process_init

from api.celery_app import celery_app

logger = structlog.get_logger(module=__name__)
_predictors: dict[str, object] = {}


@worker_process_init.connect
def _load_predictors(**kwargs):
    """Load all model versions once per forked worker process at startup."""
    global _predictors
    from api.predictor import load_all_versions
    from config import settings
    _predictors = load_all_versions(settings.model_versions_dir)


@celery_app.task(bind=True, name="sentiment.predict")
def predict_task(self, text: str, version: str) -> dict:
    predictor = _predictors.get(version)
    if predictor is None:
        raise ValueError(f"Model version '{version}' not available in worker. Loaded: {list(_predictors)}")
    return predictor.predict(text)
