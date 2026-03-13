"""
Queue Service (generic pipeline)
================================

Generic queue/outbox pipeline: Celery + Redis + Postgres. Fits any use case
(notifications, document processing, verification, financial analysis, etc.); transport can be extended (e.g. Kafka) later.
Notifications (and other domains) provide only handlers; queue owns the pipeline.

Public API:
  - broker: get_broker_url, get_celery_config, get_celery_include_list
  - tasks: process_outbox_batch, run_agent
  - scheduler: run_periodic_outbox_processor, get_beat_schedule
  - outbox_processor: start_outbox_processor, stop_outbox_processor
  - generic_outbox_processor: register_outbox, process_batch_for_queue (generic)
  - outbox_backends: NotificationOutboxBackend (notification_outbox table)
  - queue_pipeline_service: produce_notification, start_notification_consumer (outbox, no Kafka)
  - queue_services_router: FastAPI router at /api/queue
"""

from app.core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "broker",
    "tasks",
    "scheduler",
    "outbox_processor",
    "queue_pipeline_service",
    "queue_services_router",
]
