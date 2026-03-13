"""
Notification outbox backend.
============================

Implements the generic OutboxBackend for the notification_outbox table.
Used by the queue pipeline to poll and update entries; the actual
notification processing (email, WebSocket, Redis) is done by the
notification handler registered in the queue (notifications domain).
"""

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from app.core.logger import get_logger

logger = get_logger(__name__)


class NotificationOutboxBackend:
    """Outbox backend for notification_outbox table. Queue owns pipeline; notifications own handler."""

    queue_name = "notification"

    def __init__(
        self,
        get_connection=None,
        release_connection=None,
        on_dead_letter: Optional[Callable] = None,
    ):
        """
        get_connection: async () -> conn
        release_connection: async (conn) -> None
        on_dead_letter: async (conn, outbox_id, reason) -> None, called when moving to DEAD_LETTER
        """
        self._get_connection = get_connection
        self._release_connection = release_connection
        self._on_dead_letter = on_dead_letter

    async def _get_conn(self):
        if self._get_connection:
            return await self._get_connection()
        from app.core.db import async_postgres_connection
        return await async_postgres_connection()

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
    ) -> List[Dict[str, Any]]:
        conn = await self._get_conn()
        try:
            # SQL-only: check table exists
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
                logger.error("notification_outbox table does not exist. Run table creation/migrations first.")
                return []
            query = """
                SELECT id, notification_id, username, organization, event_type,
                       payload, priority, status, retry_count, max_retries,
                       scheduled_at, next_retry_at, created_at, last_error, error_code
                FROM notification_outbox
                WHERE status = 'PENDING'
                  AND priority >= $1
                  AND (scheduled_at <= NOW() OR scheduled_at IS NULL)
                ORDER BY priority DESC, scheduled_at ASC, created_at ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            """
            rows = await conn.fetch(query, priority_min, limit)
            return [self._row_to_entry(r) for r in rows]
        except Exception as e:
            logger.error("Failed to get pending notification outbox entries: %s", e, exc_info=True)
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
                WHERE id = $1 AND status = 'PENDING'
                """,
                outbox_id,
            )
            return result == "UPDATE 1"
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
            return result == "UPDATE 1"
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
                delay = min(300 * (2 ** retry_count), 3600)
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
                    "Outbox %s failed, retry %s/%s at %s",
                    outbox_id,
                    retry_count + 1,
                    max_retries,
                    next_retry_at,
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
                        logger.error("on_dead_letter callback error: %s", e, exc_info=True)
                logger.warning("Outbox %s moved to dead letter after %s retries", outbox_id, retry_count)
                return False
        finally:
            await self._release_conn(conn)

    async def get_failed_retry_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = await self._get_conn()
        try:
            rows = await conn.fetch(
                """
                SELECT id, notification_id, username, organization, event_type,
                       payload, priority, status, retry_count, max_retries,
                       scheduled_at, next_retry_at, created_at, last_error, error_code
                FROM notification_outbox
                WHERE status = 'FAILED'
                  AND retry_count < max_retries
                  AND next_retry_at <= NOW()
                ORDER BY priority DESC, next_retry_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
                """,
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
            return result == "UPDATE 1"
        finally:
            await self._release_conn(conn)
