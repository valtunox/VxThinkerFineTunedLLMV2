"""
Queue Broker Configuration
==========================

Celery + Redis broker/backend configuration.
Single place for broker URLs and Celery app config so queue, notifications,
and messaging share the same conventions.

Uses:
  - BROKER_URL / REDIS_URL for Celery broker and result backend
"""

import os
from typing import List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Broker URLs (env-driven, same pattern as core/settings)
# -----------------------------------------------------------------------------
BROKER_URL = os.environ.get(
    "BROKER_URL",
    os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
)
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)


def get_broker_url() -> str:
    """Return the Celery/Redis broker URL."""
    return BROKER_URL


def get_result_backend() -> str:
    """Return the Celery result backend URL."""
    return RESULT_BACKEND


def get_celery_config(
    task_serializer: str = "json",
    result_serializer: str = "json",
    accept_content: Optional[List[str]] = None,
    task_track_started: bool = True,
    worker_send_task_events: bool = True,
    task_time_limit: Optional[int] = 30 * 60,
    task_soft_time_limit: Optional[int] = 25 * 60,
) -> dict:
    """
    Return Celery app configuration dict.
    Used by app.core.celery_app and by queue tasks.
    """
    accept_content = accept_content or ["json"]
    return {
        "broker_url": BROKER_URL,
        "result_backend": RESULT_BACKEND,
        "task_serializer": task_serializer,
        "result_serializer": result_serializer,
        "accept_content": accept_content,
        "task_track_started": task_track_started,
        "worker_send_task_events": worker_send_task_events,
        "task_time_limit": task_time_limit,
        "task_soft_time_limit": task_soft_time_limit,
    }


def get_celery_include_list() -> List[str]:
    """
    List of task modules to include in Celery worker.
    Centralizes which services register tasks.
    """
    return [
        "app.services.queue.tasks",
        "app.services.notifications.tasks",
        "app.services.messaging.tasks",
        "app.services.seo_tracker.tasks",
    ]
