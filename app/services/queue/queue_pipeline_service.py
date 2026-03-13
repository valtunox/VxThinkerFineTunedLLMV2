"""
Queue Pipeline Service (notification pipeline)
===============================================

Notification pipeline using Celery + Redis + Postgres (outbox). No Kafka.
Enqueue via outbox; process via generic outbox processor (queue) + notification handler.

Flow:
  1. Produce: create notification (writes to notification_outbox) or enqueue payload to outbox.
  2. Consume: outbox processor (Postgres) + notification handler (email, WebSocket, Redis).

Usage:
  from app.services.queue.queue_pipeline_service import (
      produce_notification,
      start_notification_consumer,
      stop_notification_consumer,
      pipeline_health,
  )
  await produce_notification({"username": "alice", "title": "Hi", "body": "Hello", ...})
  await start_notification_consumer()
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

from app.core.logger import get_logger

logger = get_logger(__name__)


async def produce_notification(payload: Dict[str, Any], key: Optional[str] = None) -> bool:
    """
    Enqueue a notification for delivery (outbox + Celery/Redis/Postgres).
    Uses payload dict only (SQL path; no models in queue).
    """
    try:
        from app.services.notifications.service import notification_service

        payload = dict(payload)
        payload.setdefault("id", str(uuid4()))
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat()
        await notification_service.create_notification_from_payload(payload, use_outbox=True)
        return True
    except Exception as e:
        logger.error("produce_notification failed: %s", e, exc_info=True)
        return False


def produce_notification_sync(payload: Dict[str, Any], key: Optional[str] = None) -> bool:
    """Synchronous enqueue: runs create_notification in event loop."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                fut = pool.submit(asyncio.run, produce_notification(payload, key))
                return fut.result(timeout=30)
        return loop.run_until_complete(produce_notification(payload, key))
    except Exception as e:
        logger.error("produce_notification_sync failed: %s", e, exc_info=True)
        return False


async def start_notification_consumer() -> bool:
    """Start the notification outbox processor (generic queue processor)."""
    from app.services.queue.outbox_processor import start_outbox_processor
    await start_outbox_processor()
    return True


async def stop_notification_consumer() -> None:
    """Stop the notification outbox processor."""
    from app.services.queue.outbox_processor import stop_outbox_processor
    await stop_outbox_processor()


async def pipeline_health() -> Dict[str, Any]:
    """Health: Redis optional, outbox processor status. No Kafka."""
    from app.services.queue.outbox_processor import is_processor_running

    health: Dict[str, Any] = {
        "kafka": False,
        "redis": None,
        "consumer_running": is_processor_running(),
    }
    try:
        from redis import asyncio as aioredis
        import os
        url = os.getenv("REDIS_URL", "redis://3.80.77.66:6379/0")
        r = aioredis.from_url(url)
        await r.ping()
        await r.close()
        health["redis"] = True
    except Exception as e:
        logger.debug("Redis health check failed: %s", e)
        health["redis"] = False
    return health
