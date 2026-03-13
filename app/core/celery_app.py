"""
VaLLM Specialist Model - Celery Configuration
Background task processing with Redis broker
"""

import logging
import sys

# Conditionally import Celery to avoid breaking if it's not installed
try:
    from celery import Celery
    from celery.signals import worker_ready, worker_shutdown
    from celery.schedules import crontab
    CELERY_AVAILABLE = True
except ImportError:
    Celery = None
    worker_ready = None
    worker_shutdown = None
    crontab = None
    CELERY_AVAILABLE = False
    import logging
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("Celery is not installed. Background task processing will be unavailable.")

from app.core.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create Celery app only if Celery is available
if CELERY_AVAILABLE:
    celery_app = Celery(
        "vallm_specialist",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=[
            "app.services.queue.tasks",
            "app.services.queue.sync_tasks",
            "app.services.queue.document_ocr_tasks",
            "app.services.queue.data_import_tasks",
        ]
    )
else:
    celery_app = None

# Configure Celery (only if available)
if CELERY_AVAILABLE and celery_app:
    celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    accept_content=settings.celery_accept_content,
    result_serializer=settings.celery_result_serializer,
    timezone=settings.celery_timezone,
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_max_retries=10,
    result_backend_retry_delay=1.0,
    
    # Task routing
    task_routes={
        "app.tasks.embedding_tasks.*": {"queue": "embedding"},
        "app.tasks.matching_tasks.*": {"queue": "matching"},
        "app.tasks.scoring_tasks.*": {"queue": "scoring"},
        "app.tasks.document_tasks.*": {"queue": "documents"},
        "app.tasks.notification_tasks.*": {"queue": "notifications"},
        "app.tasks.analytics_tasks.*": {"queue": "analytics"},
        "document.process_ocr": {"queue": "default"},
        "data.process_import": {"queue": "default"},
        "queue.process_batch": {"queue": "default"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues={
        "default": {
            "exchange": "default",
            "routing_key": "default",
        },
        "embedding": {
            "exchange": "embedding",
            "routing_key": "embedding",
        },
        "matching": {
            "exchange": "matching", 
            "routing_key": "matching",
        },
        "scoring": {
            "exchange": "scoring",
            "routing_key": "scoring",
        },
        "documents": {
            "exchange": "documents",
            "routing_key": "documents",
        },
        "notifications": {
            "exchange": "notifications",
            "routing_key": "notifications",
        },
        "analytics": {
            "exchange": "analytics",
            "routing_key": "analytics",
        },
        "verification": {
            "exchange": "verification",
            "routing_key": "verification",
        },
    },
    
    # Task time limits (soft time limit only works on Linux/POSIX)
    task_soft_time_limit=300 if sys.platform != "win32" else None,  # 5 minutes
    task_time_limit=600,       # 10 minutes

    # Pool: use 'solo' on Windows (prefork/billiard doesn't work), prefork on Linux
    worker_pool="solo" if sys.platform == "win32" else "prefork",

    # Worker settings
    worker_disable_rate_limits=False,
    worker_hijack_root_logger=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    worker_log_color=False,

    # Beat schedule -- periodic tasks (15-min intervals for background sync)
    beat_schedule={
        "process-outbox-every-15min": {
            "task": "queue.process_outbox_batch",
            "schedule": 15 * 60,
            "kwargs": {"limit": 20},
        },
        "health-check-every-5min": {
            "task": "app.core.celery_app.health_check",
            "schedule": 5 * 60,
        },
    },

    # Broker connection retry
    broker_connection_retry_on_startup=True,
    )


# Register signal handlers only if Celery is available
if CELERY_AVAILABLE and worker_ready and worker_shutdown:
    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        """Handle worker ready event"""
        logger.info(f"Celery worker {sender} is ready")


    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        """Handle worker shutdown event"""
        logger.info(f"Celery worker {sender} is shutting down")


# Task decorators for easy usage
def task(*args, **kwargs):
    """Custom task decorator with default settings"""
    if not CELERY_AVAILABLE or celery_app is None:
        raise RuntimeError("Celery is not available. Install celery to use background tasks.")
    return celery_app.task(*args, **kwargs)


def periodic_task(*args, **kwargs):
    """Periodic task decorator"""
    if not CELERY_AVAILABLE or celery_app is None:
        raise RuntimeError("Celery is not available. Install celery to use background tasks.")
    return celery_app.task(*args, **kwargs)


# Health check task (only if Celery is available)
if CELERY_AVAILABLE and celery_app:
    @celery_app.task(bind=True)
    def health_check(self):
        """Health check task for monitoring"""
        return {
            "status": "healthy",
            "task_id": self.request.id,
            "worker": self.request.hostname,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
else:
    def health_check(self):
        """Health check task (stub when Celery is not available)"""
        raise RuntimeError("Celery is not available. Install celery to use background tasks.")


# Get Celery app instance
def get_celery_app():
    """Get Celery app instance"""
    if not CELERY_AVAILABLE:
        logger.warning("Celery is not available. Background task processing will be unavailable.")
        return None
    return celery_app


if __name__ == "__main__":
    if CELERY_AVAILABLE and celery_app:
        celery_app.start()
    else:
        logger.error("Cannot start Celery: Celery is not installed or not available.")
