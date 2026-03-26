"""
Queue Background Tasks
======================

Background jobs for agent runs, outbox processing, and other queue-driven work.
Uses the shared Celery app from core; broker config from queue.broker.

Tasks:
  - process_outbox_batch: Process a batch of outbox entries (all registered queues)
  - process_campaign_batch: Process campaign email outbox entries only
  - run_agent: Enqueue an agent run for async execution (optional)
"""

import asyncio
import threading
from typing import Any, Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# Use core Celery app so one worker can run queue + notifications + messaging tasks
try:
    from app.core.celery_app import celery_app
except ImportError:
    celery_app = None

# One persistent event loop per worker thread.
# asyncpg connection pools are bound to the loop they were created on, so we
# must NOT create-then-close a new loop per task (that kills the pool).
_thread_local = threading.local()


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop for the current worker thread."""
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    return loop


def process_outbox_batch(limit: int = 20) -> Dict[str, Any]:
    """
    Process a batch of outbox entries (e.g. notification outbox).
    Calls the outbox processor service; typically the outbox_processor loop runs
    in-app, but this task allows on-demand or scheduled batch processing.

    Args:
        limit: Max number of entries to process per batch.

    Returns:
        Dict with processed, completed, failed, retries counts.
    """
    if limit <= 0:
        return {"processed": 0, "completed": 0, "failed": 0, "retries": 0}
    try:
        from app.services.queue.outbox_processor import process_outbox_manually
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(process_outbox_manually(limit=limit))
        return result
    except Exception as e:
        logger.error("queue.process_outbox_batch failed: %s", e, exc_info=True)
        return {"processed": 0, "completed": 0, "failed": 1, "retries": 0, "error": str(e)}


def run_agent(
    agent_key: str,
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run an AI agent in the background (async job).
    Dispatches to the agent handler and returns result metadata.

    Args:
        agent_key: Handler key (e.g. networking, marketing, code).
        query: User query.
        user_id: Optional user id.
        session_id: Optional session id.
        model: Optional model override.
        context: Optional context dict.

    Returns:
        Dict with status, agent_type, response excerpt, or error.
    """
    try:
        from app.services.ai.ai_services_router import _get_agent_handler
        handler = _get_agent_handler(agent_key)
        if not handler:
            return {"status": "error", "error": f"Agent {agent_key} not available"}
        event = {
            "query": query,
            "user_id": user_id or "anonymous",
            "session_id": session_id,
            "model": model,
            "context": context or {},
        }
        result = handler(event)
        status_code = result.get("statusCode", 200)
        body = result.get("body", "{}")
        if isinstance(body, str):
            import json
            try:
                body = json.loads(body)
            except Exception:
                body = {"response": body}
        return {
            "status": "success" if status_code == 200 else "error",
            "status_code": status_code,
            "agent_type": body.get("agent_type", agent_key),
            "response_excerpt": (body.get("response") or "")[:500],
        }
    except Exception as e:
        logger.error("queue.run_agent failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


_CAMPAIGN_MAX_RETRIES = 3
_CAMPAIGN_RETRY_DELAY = 1.0


def _is_transient_db_error(exc: Exception) -> bool:
    """Return True for errors caused by stale / reset connections."""
    msg = str(exc).lower()
    transient_markers = (
        "connection was closed",
        "connection reset by peer",
        "connection refused",
        "broken pipe",
        "connection does not exist",
    )
    return any(m in msg for m in transient_markers)


def process_campaign_batch(campaign_id: Optional[int] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Process a batch of campaign email outbox entries.

    If campaign_id is provided, only entries for that campaign are processed
    (via aggregate_id filter).  Otherwise processes all pending campaign emails.

    Retries automatically on transient database connection errors (stale pool).

    Args:
        campaign_id: Optional campaign ID to restrict processing.
        limit: Max entries to process per batch.

    Returns:
        Dict with processed, completed, failed, retries counts.
    """
    import time as _t
    _start = _t.time()
    if limit <= 0:
        return {"processed": 0, "completed": 0, "failed": 0, "retries": 0}

    logger.info(
        "[Campaivaltunoxatch] START campaign_id=%s limit=%d",
        campaign_id, limit,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, _CAMPAIGN_MAX_RETRIES + 1):
        try:
            from app.services.queue.outbox_processor import process_campaign_outbox_manually
            loop = _get_or_create_event_loop()

            if attempt > 1:
                from app.core.db import _ensure_healthy_pool
                loop.run_until_complete(_ensure_healthy_pool())

            result = loop.run_until_complete(
                process_campaign_outbox_manually(limit=limit, campaign_id=campaign_id)
            )

            _check_campaign_completion(campaign_id)

            elapsed = _t.time() - _start
            processed = result.get("processed", 0)
            completed = result.get("completed", 0)
            failed = result.get("failed", 0)
            retries = result.get("retries", 0)

            logger.info(
                "[Campaivaltunoxatch] DONE campaign_id=%s processed=%s completed=%s "
                "failed=%s retries=%s elapsed=%.2fs",
                campaign_id, processed, completed, failed, retries, elapsed,
            )

            _log_campaign_final_stats(campaign_id)
            return result

        except Exception as e:
            last_err = e
            if _is_transient_db_error(e) and attempt < _CAMPAIGN_MAX_RETRIES:
                logger.warning(
                    "[Campaivaltunoxatch] Transient DB error on attempt %d/%d, retrying in %.1fs: %s",
                    attempt, _CAMPAIGN_MAX_RETRIES, _CAMPAIGN_RETRY_DELAY * attempt, e,
                )
                import time
                time.sleep(_CAMPAIGN_RETRY_DELAY * attempt)
                continue
            break

    elapsed = _t.time() - _start
    logger.error(
        "[Campaivaltunoxatch] FAILED campaign_id=%s error=%s elapsed=%.2fs",
        campaign_id, last_err, elapsed,
        exc_info=True,
    )
    return {"processed": 0, "completed": 0, "failed": 1, "retries": 0, "error": str(last_err)}


def _check_campaign_completion(campaign_id: Optional[int]):
    """Mark campaign as completed when no pending/processing entries remain."""
    if not campaign_id:
        return
    try:
        from app.core.db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM notification_outbox
            WHERE aggregate_id = %s
              AND event_type = 'campaign_email'
            GROUP BY status
            """,
            (str(campaign_id),),
        )
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        logger.info(
            "[Campaivaltunoxatch] OUTBOX_STATUS campaign_id=%s breakdown=%s",
            campaign_id, status_counts,
        )

        remaining = sum(
            status_counts.get(s, 0) for s in ('PENDING', 'PROCESSING', 'FAILED')
        )
        if remaining == 0:
            cursor.execute(
                """
                UPDATE campaign
                SET status = 'completed', completed_at = NOW(), updated_at = NOW()
                WHERE id = %s AND status IN ('queued', 'active')
                """,
                (campaign_id,),
            )
            conn.commit()
            logger.info(
                "[Campaivaltunoxatch] COMPLETED campaign_id=%s (all outbox entries processed) "
                "completed=%s dead_letter=%s",
                campaign_id,
                status_counts.get('COMPLETED', 0),
                status_counts.get('DEAD_LETTER', 0),
            )
        else:
            logger.info(
                "[Campaivaltunoxatch] campaign_id=%s still has %d remaining entries "
                "(pending=%d processing=%d failed=%d)",
                campaign_id, remaining,
                status_counts.get('PENDING', 0),
                status_counts.get('PROCESSING', 0),
                status_counts.get('FAILED', 0),
            )
        conn.close()
    except Exception as e:
        logger.error(
            "[Campaivaltunoxatch] _check_campaign_completion error campaign_id=%s: %s",
            campaign_id, e,
        )


def _log_campaign_final_stats(campaign_id: Optional[int]):
    """Log the final campaign stats from the DB after batch processing."""
    if not campaign_id:
        return
    try:
        from app.core.db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT campaign_name, status, total_recipients,
                   COALESCE(emails_sent, 0) as emails_sent,
                   COALESCE(emails_failed, 0) as emails_failed,
                   provider
            FROM campaign
            WHERE id = %s
            """,
            (campaign_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            name, status, total, sent, failed, provider = row
            accounted = sent + failed
            remaining = max(total - accounted, 0) if total else 0
            logger.info(
                "[Campaivaltunoxatch] CAMPAIGN_SUMMARY campaign_id=%s name='%s' status=%s "
                "provider=%s total_recipients=%d emails_sent=%d emails_failed=%d "
                "emails_remaining=%d",
                campaign_id, name, status, provider, total, sent, failed, remaining,
            )
            if remaining > 0:
                logger.info(
                    "[Campaivaltunoxatch] %d out of %d emails still pending for campaign '%s'. "
                    "Possible causes: outbox entries in FAILED/PENDING state awaiting retry, "
                    "leads already marked sent=TRUE (check allow_already_sent), or entries "
                    "skipped by SKIP_ALREADY_SENT / SKIP_CLAIM_FAILED.",
                    remaining, total, name,
                )
    except Exception as e:
        logger.error(
            "[Campaivaltunoxatch] _log_campaign_final_stats error campaign_id=%s: %s",
            campaign_id, e,
        )


# Register with Celery when app is available (assign back so .delay() works)
if celery_app is not None:
    process_outbox_batch = celery_app.task(name="queue.process_outbox_batch")(process_outbox_batch)
    process_campaign_batch = celery_app.task(
        name="queue.process_campaign_batch",
        soft_time_limit=1800,
        time_limit=3600,
    )(process_campaign_batch)
    run_agent = celery_app.task(name="queue.run_agent")(run_agent)
