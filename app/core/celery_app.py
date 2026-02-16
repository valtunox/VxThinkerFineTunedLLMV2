import os
from celery import Celery

# Read broker/backend from env
BROKER_URL = os.environ.get("BROKER_URL", os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)

celery = Celery(
    "infinity_cloud_gateway",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.services.cloud.gateway.tasks"],
)

# Basic configuration - use JSON serializer for portability
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_send_task_events=True,
)
