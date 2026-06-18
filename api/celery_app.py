from celery import Celery

from config import settings

celery_app = Celery(
    "sentiment",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["api.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)
