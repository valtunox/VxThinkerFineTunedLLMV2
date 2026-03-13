"""
Campaign email outbox backend.
===============================

Implements the generic OutboxBackend for campaign emails stored in the
notification_outbox table with event_type='campaign_email'.  Reuses the
same table, retry logic, and dead-letter handling as notifications --
only the SELECT filters differ.

The campaign handler registered with this backend is responsible for
actually sending emails via SMTP / Mailjet / SendGrid.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

EVENT_TYPE = "campaign_email"

_MAX_CONN_RETRIES = 3
_CONN_RETRY_DELAY = 0.5


class CampaignOutboxBackend:
    """Outbox backend for campaign emails in the notification_outbox table."""

    queue_name = "campaign_email"

    def __init__(
        self,
        get_connection=None,
        release_connection=None,
        on_dead_letter: Optional[Callable] = None,
    ):
        self._get_connection = get_connection
        self._release_connection = release_connection
        self._on_dead_letter = on_dead_letter

    async def _get_conn(self):
        """Acquire a connection, retrying on transient failures (stale pool)."""
        last_err = None
        for attempt in range(1, _MAX_CONN_RETRIES + 1):
            try:
                if self._get_connection:
                    conn = await self._get_connection()
                else:
                    from app.core.db import async_postgres_connection
                    conn = await async_postgres_connection()
                await conn.execute("SELECT 1")
                return conn
            except (
                ConnectionResetError,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                last_err = exc
                logger.warning(
                    "[CampaignOutbox] connection attempt %d/%d failed: %s",
                    attempt, _MAX_CONN_RETRIES, exc,
                )
                from app.core.db import async_pool as _pool
                if _pool is not None:
                    try:
                        from app.core.db import _ensure_healthy_pool
                        await _ensure_healthy_pool()
                    except Exception:
                        pass
                if attempt < _MAX_CONN_RETRIES:
                    await asyncio.sleep(_CONN_RETRY_DELAY * attempt)
            except Exception as exc:
                if "connection was closed" in str(exc).lower():
                    last_err = exc
                    logger.warning(
                        "[CampaignOutbox] stale connection attempt %d/%d: %s",
                        attempt, _MAX_CONN_RETRIES, exc,
                    )
                    from app.core.db import _ensure_healthy_pool
                    await _ensure_healthy_pool()
                    if attempt < _MAX_CONN_RETRIES:
                        await asyncio.sleep(_CONN_RETRY_DELAY * attempt)
                else:
                    raise
        raise last_err  # type: ignore[misc]

    async def _release_conn(self, conn):
        if self._release_connection:
            await self._release_connection(conn)
        else:
            from app.core.db import release_async_connection
            await release_async_connection(conn)

    @staticmethod
    def _row_to_entry(row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "notification_id": row["notification_id"],
            "username": row["username"],
            "organization": row["organization"],
            "event_type": row["event_type"],
            "aggregate_id": row["aggregate_id"],
            "payload": row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"] or "{}"),
            "priority": row["priority"],
            "status": row["status"],
            "retry_count": row["retry_count"],
            "max_retries": row["max_retries"],
            "scheduled_at": row["scheduled_at"],
            "next_retry_at": row["next_retry_at"],
            "created_at": row["created_at"],
            "last_error": row["last_error"],
            "error_code": row["error_code"],
        }

    async def get_pending_entries(
        self,
        limit: int = 10,
        priority_min: int = 1,
        campaign_id: Optional[int] = None,
        reset_failed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch PENDING outbox entries and atomically claim them as PROCESSING.

        When *campaign_id* and *reset_failed* are both set, any FAILED
        entries for that campaign are first reset to PENDING inside the
        same explicit transaction, then all PENDING entries are claimed
        in a single UPDATE ... RETURNING, eliminating races between
        fetch and mark_processing.
        """
        conn = await self._get_conn()
        try:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                "notification_outbox",
            )
            if not exists:
                logger.error("[CampaignOutbox] notification_outbox table does not exist")
                return []

            async with conn.transaction():
                if reset_failed and campaign_id is not None:
                    result = await conn.execute(
                        """
                        UPDATE notification_outbox
                        SET status = 'PENDING',
                            next_retry_at = NULL,
                            updated_at = NOW()
                        WHERE aggregate_id = $1
                          AND event_type = $2
                          AND status IN ('FAILED', 'PROCESSING')
                        """,
                        str(campaign_id),
                        EVENT_TYPE,
                    )
                    reset_count = int(result.split()[-1]) if result else 0
                    if reset_count > 0:
                        logger.info(
                            "[CampaignOutbox] RESET_FAILED campaign_id=%s reset %d FAILED/PROCESSING entries to PENDING",
                            campaign_id, reset_count,
                        )

                if campaign_id is not None:
                    rows = await conn.fetch(
                        """
                        UPDATE notification_outbox
                        SET status = 'PROCESSING', updated_at = NOW()
                        WHERE id IN (
                            SELECT id FROM notification_outbox
                            WHERE status = 'PENDING'
                              AND event_type = $1
                              AND aggregate_id = $2
                              AND priority >= $3
                              AND (scheduled_at <= NOW() OR scheduled_at IS NULL)
                            ORDER BY priority DESC, created_at ASC
                            LIMIT $4
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING id, notification_id, username, organization, event_type,
                                  aggregate_id, payload, priority, status, retry_count,
                                  max_retries, scheduled_at, next_retry_at, created_at,
                                  last_error, error_code
                        """,
                        EVENT_TYPE,
                        str(campaign_id),
                        priority_min,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        UPDATE notification_outbox
                        SET status = 'PROCESSING', updated_at = NOW()
                        WHERE id IN (
                            SELECT id FROM notification_outbox
                            WHERE status = 'PENDING'
                              AND event_type = $1
                              AND priority >= $2
                              AND (scheduled_at <= NOW() OR scheduled_at IS NULL)
                            ORDER BY priority DESC, created_at ASC
                            LIMIT $3
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING id, notification_id, username, organization, event_type,
                                  aggregate_id, payload, priority, status, retry_count,
                                  max_retries, scheduled_at, next_retry_at, created_at,
                                  last_error, error_code
                        """,
                        EVENT_TYPE,
                        priority_min,
                        limit,
                    )

            entries = [self._row_to_entry(r) for r in rows]
            if entries:
                campaign_ids = set(e.get("aggregate_id") for e in entries)
                logger.info(
                    "[CampaignOutbox] Fetched and claimed %d entries for campaigns: %s",
                    len(entries), campaign_ids,
                )
            else:
                logger.info(
                    "[CampaignOutbox] No pending entries found (campaign_id=%s, limit=%d)",
                    campaign_id, limit,
                )
            return entries
        except Exception as e:
            logger.error("[CampaignOutbox] Failed to get pending entries: %s", e, exc_info=True)
            raise
        finally:
            await self._release_conn(conn)

    async def mark_processing(self, outbox_id: Any) -> bool:
        conn = await self._get_conn()
        try:
            result = await conn.execute(
                """
                UPDATE notification_outbox
                SET status = 'PROCESSING', updated_at = NOW()
                WHERE id = $1 AND status IN ('PENDING', 'PROCESSING')
                """,
                outbox_id,
            )
            ok = result == "UPDATE 1"
            if ok:
                logger.debug("[CampaignOutbox] PROCESSING outbox_id=%s", outbox_id)
            return ok
        finally:
            await self._release_conn(conn)

    async def mark_completed(self, outbox_id: Any) -> bool:
        conn = await self._get_conn()
        try:
            result = await conn.execute(
                """
                UPDATE notification_outbox
                SET status = 'COMPLETED', processed_at = NOW(), updated_at = NOW()
                WHERE id = $1
                """,
                outbox_id,
            )
            ok = result == "UPDATE 1"
            if ok:
                logger.info("[CampaignOutbox] COMPLETED outbox_id=%s", outbox_id)
            return ok
        finally:
            await self._release_conn(conn)

    async def mark_failed(
        self,
        outbox_id: Any,
        error_message: str,
        error_code: Optional[str] = None,
        retry: bool = True,
    ) -> bool:
        conn = await self._get_conn()
        try:
            row = await conn.fetchrow(
                "SELECT retry_count, max_retries FROM notification_outbox WHERE id = $1",
                outbox_id,
            )
            if not row:
                return False
            retry_count = row["retry_count"]
            max_retries = row["max_retries"]

            if retry and retry_count < max_retries:
                delay = min(60 * (2 ** retry_count), 1800)
                next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
                await conn.execute(
                    """
                    UPDATE notification_outbox
                    SET status = 'FAILED',
                        retry_count = retry_count + 1,
                        next_retry_at = $1,
                        last_error = $2,
                        error_code = $3,
                        updated_at = NOW()
                    WHERE id = $4
                    """,
                    next_retry_at,
                    error_message,
                    error_code,
                    outbox_id,
                )
                logger.info(
                    "Campaign outbox %s failed, retry %s/%s at %s",
                    outbox_id, retry_count + 1, max_retries, next_retry_at,
                )
                return True
            else:
                await conn.execute(
                    """
                    UPDATE notification_outbox
                    SET status = 'DEAD_LETTER',
                        last_error = $1,
                        error_code = $2,
                        updated_at = NOW()
                    WHERE id = $3
                    """,
                    error_message,
                    error_code,
                    outbox_id,
                )
                if self._on_dead_letter:
                    try:
                        await self._on_dead_letter(conn, outbox_id, error_message)
                    except Exception as e:
                        logger.error("Campaign on_dead_letter callback error: %s", e, exc_info=True)
                logger.warning(
                    "Campaign outbox %s moved to dead letter after %s retries",
                    outbox_id, retry_count,
                )
                return False
        finally:
            await self._release_conn(conn)

    async def get_failed_retry_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = await self._get_conn()
        try:
            rows = await conn.fetch(
                """
                SELECT id, notification_id, username, organization, event_type,
                       aggregate_id, payload, priority, status, retry_count,
                       max_retries, scheduled_at, next_retry_at, created_at,
                       last_error, error_code
                FROM notification_outbox
                WHERE status = 'FAILED'
                  AND event_type = $1
                  AND retry_count < max_retries
                  AND next_retry_at <= NOW()
                ORDER BY priority DESC, next_retry_at ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
                """,
                EVENT_TYPE,
                limit,
            )
            return [self._row_to_entry(r) for r in rows]
        finally:
            await self._release_conn(conn)

    async def reset_to_pending(self, outbox_id: Any) -> bool:
        conn = await self._get_conn()
        try:
            result = await conn.execute(
                """
                UPDATE notification_outbox
                SET status = 'PENDING', updated_at = NOW()
                WHERE id = $1
                """,
                outbox_id,
            )
            ok = result == "UPDATE 1"
            if ok:
                logger.info("[CampaignOutbox] RESET_TO_PENDING outbox_id=%s", outbox_id)
            return ok
        finally:
            await self._release_conn(conn)
