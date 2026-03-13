"""
Generic outbox processor.
=========================

Queue-owned pipeline: poll outbox backends, call domain handlers, update status.
Fits any use case (notifications, email, sales campaigns, resumes, staffing, etc.); transport is Celery + Redis + Postgres.
Other transports (e.g. Kafka) can be added later as alternative backends without changing this.
"""

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

from app.services.queue.outbox_backends.base import OutboxBackend, OutboxEntry, OutboxHandler

logger = get_logger(__name__)

# Registry: queue_name -> (backend, handler)
_backend_registry: Dict[str, tuple[OutboxBackend, OutboxHandler]] = {}


def register_outbox(queue_name: str, backend: OutboxBackend, handler: OutboxHandler) -> None:
    """Register an outbox backend and its domain handler. Call at app startup."""
    _backend_registry[queue_name] = (backend, handler)
    logger.info("Registered outbox queue=%s", queue_name)


def get_registered_queues() -> List[str]:
    """Return list of registered queue names."""
    return list(_backend_registry.keys())


async def process_entry(
    backend: OutboxBackend,
    handler: OutboxHandler,
    entry: OutboxEntry,
    already_claimed: bool = False,
) -> bool:
    """Process one outbox entry: mark processing, run handler, mark completed or failed.

    Args:
        already_claimed: If True, skip the mark_processing step (the entry
                         was already atomically set to PROCESSING during fetch).
    """
    outbox_id = entry["id"]
    payload = entry.get("payload") or entry
    try:
        if not already_claimed:
            ok = await backend.mark_processing(outbox_id)
            if not ok:
                logger.warning("[OutboxEntry] mark_processing returned False for outbox_id=%s (already claimed?)", outbox_id)
                return False
        success = await handler(payload, entry)
        if success:
            await backend.mark_completed(outbox_id)
            return True
        logger.warning("[OutboxEntry] Handler returned False for outbox_id=%s, marking FAILED", outbox_id)
        await backend.mark_failed(
            outbox_id,
            "Handler returned False",
            error_code="PROCESSING_FAILED",
            retry=True,
        )
        return False
    except Exception as e:
        logger.error(
            "[OutboxEntry] EXCEPTION outbox_id=%s error=%s",
            outbox_id, e, exc_info=True,
        )
        try:
            await backend.mark_failed(
                outbox_id,
                str(e),
                error_code=type(e).__name__,
                retry=True,
            )
        except Exception as mark_err:
            logger.error("[OutboxEntry] mark_failed also failed for outbox_id=%s: %s", outbox_id, mark_err)
        return False


async def process_batch_for_queue(
    queue_name: str,
    limit: int = 10,
    campaign_id: int = None,
    reset_failed: bool = False,
) -> Dict[str, int]:
    """Process a batch of pending and retry entries for one queue. Returns stats.

    Args:
        queue_name: Registered outbox queue name.
        limit: Max entries to fetch per phase (pending / retry).
        campaign_id: If provided and the backend supports it, restrict
                     to entries for this campaign (aggregate_id).
        reset_failed: If True, reset FAILED entries to PENDING atomically
                      on the same connection before fetching (requires
                      backend support).
    """
    stats = {"processed": 0, "completed": 0, "failed": 0, "retries": 0}
    reg = _backend_registry.get(queue_name)
    if not reg:
        logger.warning("No outbox registered for queue=%s", queue_name)
        return stats

    backend, handler = reg

    get_kwargs: Dict[str, Any] = {"limit": limit}
    import inspect
    sig = inspect.signature(backend.get_pending_entries)
    pre_claimed = False
    if campaign_id is not None and "campaign_id" in sig.parameters:
        get_kwargs["campaign_id"] = campaign_id
    if reset_failed and "reset_failed" in sig.parameters:
        get_kwargs["reset_failed"] = True
        pre_claimed = True

    pending = await backend.get_pending_entries(**get_kwargs)
    stats["processed"] = len(pending)
    if pending:
        logger.info(
            "[OutboxBatch] queue=%s fetched %d entries (limit=%d, pre_claimed=%s)",
            queue_name, len(pending), limit, pre_claimed,
        )
    for idx, entry in enumerate(pending, 1):
        outbox_id = entry.get("id", "?")
        payload = entry.get("payload") or {}
        recipient = payload.get("recipient_email", "N/A")
        logger.info(
            "[OutboxBatch] queue=%s processing %d/%d outbox_id=%s recipient=%s",
            queue_name, idx, len(pending), outbox_id, recipient,
        )
        success = await process_entry(backend, handler, entry, already_claimed=pre_claimed)
        if success:
            stats["completed"] += 1
        else:
            stats["failed"] += 1

    retries = await backend.get_failed_retry_entries(limit=limit)
    stats["retries"] = len(retries)
    if retries:
        logger.info(
            "[OutboxBatch] queue=%s fetched %d retry entries",
            queue_name, len(retries),
        )
    for idx, entry in enumerate(retries, 1):
        outbox_id = entry.get("id", "?")
        logger.info(
            "[OutboxBatch] queue=%s retrying %d/%d outbox_id=%s",
            queue_name, idx, len(retries), outbox_id,
        )
        await backend.reset_to_pending(entry["id"])
        success = await process_entry(backend, handler, entry)
        if success:
            stats["completed"] += 1
        else:
            stats["failed"] += 1

    if stats["processed"] > 0 or stats["retries"] > 0:
        logger.info(
            "[OutboxBatch] queue=%s BATCH_DONE processed=%d completed=%d failed=%d retries=%d",
            queue_name, stats["processed"], stats["completed"], stats["failed"], stats["retries"],
        )

    return stats


async def process_batch_all_queues(
    limit_per_queue: int = 10,
    exclude_queues: Optional[set] = None,
) -> Dict[str, Dict[str, int]]:
    """Process a batch for every registered queue. Returns { queue_name: stats }.

    Args:
        limit_per_queue: Max entries to process per queue.
        exclude_queues: Optional set of queue names to skip (e.g. queues
                        owned exclusively by Celery workers).
    """
    result = {}
    for queue_name in _backend_registry:
        if exclude_queues and queue_name in exclude_queues:
            continue
        result[queue_name] = await process_batch_for_queue(queue_name, limit=limit_per_queue)
    return result


async def process_notification_outbox_batch(limit: int = 10) -> Dict[str, int]:
    """
    Process the notification outbox only. Convenience for backward compatibility.
    Uses the generic processor with the registered 'notification' queue.
    """
    return await process_batch_for_queue("notification", limit=limit)
