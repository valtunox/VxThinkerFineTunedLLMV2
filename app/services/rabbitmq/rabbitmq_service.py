"""
RabbitMQ Service for message queue operations.
Provides connect, publish, declare queue, and health check.
Mirrors the pattern used by Redis, Celery, and Kafka services.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pika
    from pika.exceptions import AMQPConnectionError, AMQPChannelError
    RABBITMQ_AVAILABLE = True
except ImportError:
    pika = None  # type: ignore
    RABBITMQ_AVAILABLE = False
    logger.warning("pika not installed. Install with: pip install pika")


class RabbitMQService:
    """
    RabbitMQ service for publishing messages and managing queues.
    Uses pika (sync); blocking calls are intended to be run in executor from async routes.
    """

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self._connection: Optional[Any] = None
        self._channel: Optional[Any] = None

    def connect(self) -> bool:
        """Establish connection and channel to RabbitMQ."""
        if not RABBITMQ_AVAILABLE:
            return False
        try:
            params = pika.URLParameters(self.url)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            logger.info("RabbitMQ connection established")
            return True
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ: %s", e)
            return False

    def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection and channel."""
        if not RABBITMQ_AVAILABLE:
            return False
        try:
            if self._channel is None or self._connection is None or self._connection.is_closed:
                return self.connect()
            return True
        except Exception:
            self._connection = None
            self._channel = None
            return self.connect()

    def close(self) -> None:
        """Close connection and channel."""
        try:
            if self._channel:
                self._channel.close()
            if self._connection and not self._connection.is_closed:
                self._connection.close()
            logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.debug("Error closing RabbitMQ connection: %s", e)
        finally:
            self._channel = None
            self._connection = None

    def declare_queue(self, queue: str, durable: bool = True) -> bool:
        """
        Declare a queue (idempotent).
        Args:
            queue: Queue name.
            durable: If True, queue survives broker restart.
        Returns:
            True if successful.
        """
        if not self._ensure_connection():
            return False
        try:
            self._channel.queue_declare(queue=queue, durable=durable)
            logger.debug("Queue declared: %s", queue)
            return True
        except Exception as e:
            logger.error("Failed to declare queue %s: %s", queue, e)
            return False

    def publish(self, queue: str, body: str | bytes | Dict[str, Any], durable: bool = True) -> bool:
        """
        Publish a message to a queue.
        Args:
            queue: Queue name.
            body: Message body (str, bytes, or dict; dict is JSON-serialized).
            durable: Declare queue as durable if not already declared.
        Returns:
            True if published successfully.
        """
        if not self._ensure_connection():
            return False
        try:
            self._channel.queue_declare(queue=queue, durable=durable)
            if isinstance(body, dict):
                body = json.dumps(body, default=str).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self._channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=body,
                properties=pika.BasicProperties(delivery_mode=2),  # persistent
            )
            logger.debug("Published message to queue: %s", queue)
            return True
        except Exception as e:
            logger.error("Failed to publish to queue %s: %s", queue, e)
            return False

    def get_queue_info(self, queue: str) -> Optional[Dict[str, Any]]:
        """
        Get queue info (message count, etc.) if the broker supports it.
        Requires management API or passive queue_declare. We use passive queue_declare
        to get message count when possible.
        """
        if not self._ensure_connection():
            return None
        try:
            method = self._channel.queue_declare(queue=queue, passive=True)
            return {"queue": queue, "message_count": method.method.message_count}
        except Exception as e:
            logger.debug("Could not get queue info for %s: %s", queue, e)
            return None

    def health_check(self) -> bool:
        """Check if RabbitMQ connection is healthy."""
        return self._ensure_connection()


# Singleton for dependency injection (optional)
_rabbitmq_service: Optional[RabbitMQService] = None


def get_rabbitmq_service() -> RabbitMQService:
    """Get or create the global RabbitMQ service instance."""
    global _rabbitmq_service
    if _rabbitmq_service is None:
        _rabbitmq_service = RabbitMQService()
    return _rabbitmq_service
