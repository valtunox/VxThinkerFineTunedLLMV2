"""
Queue Outbox Processor
======================

Generic outbox pipeline: poll Postgres outbox, call domain handlers (notifications, email, etc.).
Uses Celery + Redis + Postgres; no Kafka in this path.

Registered queues:
    - notification  : system notifications (email, WebSocket, push)
    - campaign_email: bulk campaign emails (SMTP / Mailjet / SendGrid)

Key Features:
    - Periodic processing of ALL registered outbox queues
    - Automatic retry and dead letter (in backend)
    - Configurable interval and graceful shutdown
"""

import asyncio
from typing import Dict, Optional

from app.core.logger import get_logger

from app.services.queue.generic_outbox_processor import (
    process_batch_all_queues,
    process_batch_for_queue,
    process_notification_outbox_batch,
    register_outbox,
)
from app.services.queue.outbox_backends.notification_backend import NotificationOutboxBackend
from app.services.queue.outbox_backends.campaign_backend import CampaignOutboxBackend
from app.services.notifications.outbox_service import (
    mark_notification_dead_letter_callback,
    process_notification_payload,
)

logger = get_logger(__name__)

_processor_task: Optional[asyncio.Task] = None
_processor_running = False
_processor_interval = 5
_notification_registered = False
_campaign_registered = False
_campaign_backend: Optional[CampaignOutboxBackend] = None


def _ensure_notification_outbox_registered() -> None:
    global _notification_registered
    if _notification_registered:
        return
    backend = NotificationOutboxBackend(on_dead_letter=mark_notification_dead_letter_callback)
    register_outbox("notification", backend, process_notification_payload)
    _notification_registered = True


def _ensure_campaign_outbox_registered() -> None:
    global _campaign_registered, _campaign_backend
    if _campaign_registered:
        return
    from app.services.sales.campaign_queue_handler import (
        mark_campaign_dead_letter_callback,
        process_campaign_email,
    )
    backend = CampaignOutboxBackend(on_dead_letter=mark_campaign_dead_letter_callback)
    register_outbox("campaign_email", backend, process_campaign_email)
    _campaign_backend = backend
    _campaign_registered = True


def _ensure_all_queues_registered() -> None:
    _ensure_notification_outbox_registered()
    _ensure_campaign_outbox_registered()


_CELERY_ONLY_QUEUES = {"campaign_email"}


async def _process_outbox_loop() -> None:
    """Background polling loop for lightweight queues (notifications, etc.).

    Campaign emails are excluded here — they are processed exclusively by
    the Celery worker (``process_campaign_batch``) to avoid a race where both
    the background loop and Celery grab the same PENDING entries.
    """
    global _processor_running
    _ensure_all_queues_registered()
    logger.info(
        "[OutboxProcessor] Started (queues: notification) interval=%ds "
        "(campaign_email excluded — handled by Celery only)",
        _processor_interval,
    )
    _processor_running = True
    cycle = 0
    while _processor_running:
        cycle += 1
        try:
            all_stats = await process_batch_all_queues(
                limit_per_queue=20,
                exclude_queues=_CELERY_ONLY_QUEUES,
            )
            for queue_name, stats in all_stats.items():
                processed = stats.get("processed", 0)
                completed = stats.get("completed", 0)
                failed = stats.get("failed", 0)
                retries = stats.get("retries", 0)
                if processed > 0 or retries > 0:
                    logger.info(
                        "[OutboxProcessor] cycle=%d queue=%s processed=%d completed=%d failed=%d retries=%d",
                        cycle, queue_name, processed, completed, failed, retries,
                    )
            await asyncio.sleep(_processor_interval)
        except asyncio.CancelledError:
            logger.info("[OutboxProcessor] Cancelled after %d cycles", cycle)
            break
        except Exception as e:
            logger.error(
                "[OutboxProcessor] Error in cycle %d: %s", cycle, e, exc_info=True,
            )
            await asyncio.sleep(_processor_interval * 2)
    logger.info("[OutboxProcessor] Stopped after %d cycles", cycle)
    _processor_running = False


async def start_outbox_processor(interval: int = 5) -> None:
    """Start the outbox processor background task."""
    global _processor_task, _processor_interval, _processor_running
    if _processor_running:
        logger.warning("[OutboxProcessor] Already running, skipping start")
        return
    _processor_interval = interval
    _processor_task = asyncio.create_task(_process_outbox_loop())
    logger.info("[OutboxProcessor] Background task created (interval=%ds)", interval)


async def stop_outbox_processor() -> None:
    """Stop the outbox processor background task."""
    global _processor_task, _processor_running
    if not _processor_running:
        return
    _processor_running = False
    if _processor_task:
        _processor_task.cancel()
        try:
            await _processor_task
        except asyncio.CancelledError:
            pass
    logger.info("[OutboxProcessor] Stopped gracefully")


def is_processor_running() -> bool:
    """Check if the outbox processor is currently running."""
    return _processor_running


async def process_outbox_manually(limit: int = 10) -> dict:
    """Manually trigger ALL registered outbox queues. Returns combined statistics."""
    _ensure_all_queues_registered()
    logger.info("[OutboxProcessor] Manual trigger ALL queues (limit=%d)", limit)
    result = await process_batch_all_queues(limit_per_queue=limit)
    logger.info("[OutboxProcessor] Manual trigger completed: %s", result)
    return result


async def process_campaign_outbox_manually(limit: int = 20, campaign_id: int = None) -> dict:
    """Manually trigger campaign email outbox processing only.

    Args:
        limit: Max entries to process.
        campaign_id: If provided, reset any FAILED entries for this campaign
                     back to PENDING before processing, and restrict processing
                     to entries belonging to this campaign only.
    """
    _ensure_campaign_outbox_registered()
    logger.info(
        "[OutboxProcessor] Manual trigger CAMPAIGN queue (limit=%d, campaign_id=%s)",
        limit, campaign_id,
    )

    result = await process_batch_for_queue(
        "campaign_email",
        limit=limit,
        campaign_id=campaign_id,
        reset_failed=campaign_id is not None,
    )
    logger.info("[OutboxProcessor] Campaign manual trigger completed: %s", result)
    return result
