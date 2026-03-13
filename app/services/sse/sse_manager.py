"""
SSE Manager — Server-Sent Events via Redis Pub/Sub
===================================================

Provides real-time progress streaming for all agent operations (document processing,
verification, financial analysis, business recommendations, etc.).  Works alongside existing polling
endpoints — nothing is replaced.

Architecture:
  Producer side:  call ``await sse_manager.publish_progress(task_id, data)``
                  from any agent handler, Kafka consumer, or Celery task.
  Consumer side:  ``GET /sse/progress/{task_id}`` returns a text/event-stream
                  that yields events until the task completes or the client
                  disconnects.

Redis channels:  ``sse:progress:{task_id}``  (one channel per task)

When Redis is unavailable the manager falls back to an in-process asyncio.Queue
so the feature still works in dev / single-process mode.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://3.80.77.66:6379/0")


def _channel(task_id: str) -> str:
    return f"sse:progress:{task_id}"


class SSEManager:
    """Publish / subscribe progress events over Redis Pub/Sub (or in-memory fallback)."""

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self._redis = None
        self._pubsub_redis = None
        self._use_redis = True
        self._queues: Dict[str, asyncio.Queue] = {}
        self._last_events: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _get_redis(self):
        """Lazy-init Redis publisher connection."""
        if self._redis is not None:
            return self._redis
        try:
            from redis import asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                max_connections=10,
            )
            await self._redis.ping()
            logger.info("SSEManager: Redis publisher connected (%s)", self.redis_url)
            return self._redis
        except Exception as e:
            logger.warning("SSEManager: Redis unavailable (%s), using in-memory fallback", e)
            self._use_redis = False
            self._redis = None
            return None

    async def _get_pubsub_redis(self):
        """Separate Redis connection for pub/sub subscriber."""
        if self._pubsub_redis is not None:
            return self._pubsub_redis
        try:
            from redis import asyncio as aioredis
            self._pubsub_redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
            )
            await self._pubsub_redis.ping()
            return self._pubsub_redis
        except Exception:
            self._use_redis = False
            return None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish_progress(
        self,
        task_id: str,
        *,
        phase: str = "",
        progress: int = 0,
        total: int = 100,
        message: str = "",
        status: str = "in_progress",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish a progress event for *task_id*.

        Parameters
        ----------
        task_id : str       Unique task / workflow identifier.
        phase   : str       Current phase label (e.g. "Searching job boards…").
        progress: int       Completed steps (0-based).
        total   : int       Total expected steps.
        message : str       Human-readable status text.
        status  : str       One of ``in_progress | completed | failed | cancelled``.
        data    : dict      Arbitrary extra payload (e.g. partial results).
        """
        event = {
            "task_id": task_id,
            "phase": phase,
            "progress": progress,
            "total": total,
            "percent": round(progress / total * 100) if total else 0,
            "message": message,
            "status": status,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload = json.dumps(event, default=str)

        if status in ("completed", "failed", "cancelled"):
            self._last_events[task_id] = payload
            if self._use_redis:
                redis = await self._get_redis()
                if redis:
                    try:
                        await redis.setex(f"sse:result:{task_id}", 600, payload)
                    except Exception:
                        pass

        if self._use_redis:
            redis = await self._get_redis()
            if redis:
                try:
                    await redis.publish(_channel(task_id), payload)
                    return
                except Exception as e:
                    logger.warning("SSEManager: Redis publish failed (%s), using fallback", e)

        q = self._queues.get(task_id)
        if q is not None:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Subscribe (yields SSE-formatted strings)
    # ------------------------------------------------------------------

    async def subscribe(
        self, task_id: str, timeout_seconds: int = 600
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE-formatted events for *task_id*.

        Stops when ``status`` is ``completed``, ``failed``, or ``cancelled``,
        or when *timeout_seconds* elapses.

        If the task already completed before the subscriber connected, the
        cached terminal event is replayed immediately.
        """
        cached = self._last_events.get(task_id)
        if not cached and self._use_redis:
            try:
                redis = await self._get_redis()
                if redis:
                    cached = await redis.get(f"sse:result:{task_id}")
            except Exception:
                pass
        if cached:
            yield f"data: {cached}\n\n"
            return

        if self._use_redis:
            async for chunk in self._subscribe_redis(task_id, timeout_seconds):
                yield chunk
        else:
            async for chunk in self._subscribe_memory(task_id, timeout_seconds):
                yield chunk

    async def _subscribe_redis(
        self, task_id: str, timeout: int
    ) -> AsyncGenerator[str, None]:
        sub_redis = await self._get_pubsub_redis()
        if not sub_redis:
            async for chunk in self._subscribe_memory(task_id, timeout):
                yield chunk
            return

        pubsub = sub_redis.pubsub()
        channel = _channel(task_id)
        await pubsub.subscribe(channel)
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=2.0
                )
                if msg and msg["type"] == "message":
                    data_str = msg["data"]
                    yield f"data: {data_str}\n\n"
                    try:
                        parsed = json.loads(data_str)
                        if parsed.get("status") in ("completed", "failed", "cancelled"):
                            return
                    except json.JSONDecodeError:
                        pass
                else:
                    yield ": heartbeat\n\n"
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def _subscribe_memory(
        self, task_id: str, timeout: int
    ) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = self._queues.setdefault(task_id, asyncio.Queue(maxsize=500))
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                try:
                    data_str = await asyncio.wait_for(q.get(), timeout=3.0)
                    yield f"data: {data_str}\n\n"
                    try:
                        parsed = json.loads(data_str)
                        if parsed.get("status") in ("completed", "failed", "cancelled"):
                            return
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            self._queues.pop(task_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_task_id(prefix: str = "task") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    async def close(self):
        if self._redis:
            await self._redis.close()
            self._redis = None
        if self._pubsub_redis:
            await self._pubsub_redis.close()
            self._pubsub_redis = None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
sse_manager = SSEManager()
