"""
Queue Scheduler (Optional)
===========================

Periodic task scheduling. Can be used to trigger outbox processing or agent jobs
on a schedule. Integrates with Celery Beat if enabled, or provides a simple
asyncio-based interval runner for in-process scheduling.

Usage:
  - Celery Beat: add schedule to app.core.celery_app.conf.beat_schedule
  - In-process: call run_periodic_outbox_processor(interval_seconds) from lifespan
"""

import asyncio
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# Default interval for periodic outbox processing when run in-process
DEFAULT_OUTBOX_INTERVAL_SEC = 30


async def run_periodic_outbox_processor(
    interval_seconds: int = DEFAULT_OUTBOX_INTERVAL_SEC,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Run outbox processing on a fixed interval (in-process scheduler).
    Use this when not using Celery Beat.

    Args:
        interval_seconds: Seconds between each run.
        stop_event: If set, stop when the event is set.
    """
    from app.services.queue.outbox_processor import process_outbox_manually
    stop = stop_event or asyncio.Event()
    while not stop.is_set():
        try:
            await process_outbox_manually(limit=20)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Scheduled outbox run failed: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=float(interval_seconds))
        except asyncio.TimeoutError:
            pass
    logger.info("Periodic outbox scheduler stopped")


def get_beat_schedule() -> dict:
    """
    Return Celery Beat schedule for queue tasks.
    Merge this into celery_app.conf.beat_schedule if using Beat.

    Example:
        from app.services.queue.broker import get_celery_app
        app = get_celery_app()
        app.conf.beat_schedule.update(get_beat_schedule())
    """
    return {
        "queue-process-outbox": {
            "task": "queue.process_outbox_batch",
            "schedule": float(DEFAULT_OUTBOX_INTERVAL_SEC),
            "options": {"kwargs": {"limit": 20}},
        },
    }
