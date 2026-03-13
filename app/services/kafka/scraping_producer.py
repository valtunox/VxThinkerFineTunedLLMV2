"""
Kafka Scraping Producer
=======================

Publishes job_scraping and websearch tasks to the ``scraping-tasks`` Kafka topic.
The scraping consumer picks them up, runs the agent, and streams SSE progress.

Falls back gracefully: returns False when Kafka is unavailable so the orchestrator
can use direct execution instead.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict

from app.core.logger import get_logger

logger = get_logger(__name__)

SCRAPING_TOPIC = "scraping-tasks"


def publish_scraping_task(payload: Dict[str, Any]) -> bool:
    """
    Publish a scraping task to Kafka.

    Parameters
    ----------
    payload : dict
        Must include ``task_id``, ``agent_type``, ``message``.
        May include ``provider``, ``model``, ``sources``, ``country``, etc.

    Returns
    -------
    bool  True if published, False if Kafka unavailable.
    """
    try:
        from app.services.kafka.kafka_service import kafka_service, KAFKA_AVAILABLE

        if not KAFKA_AVAILABLE:
            logger.warning("scraping_producer: kafka-python not available")
            return False

        if not kafka_service.producer:
            if not kafka_service.create_producer():
                logger.warning("scraping_producer: could not create Kafka producer")
                return False

        task_id = payload.get("task_id", f"task_{uuid.uuid4().hex[:12]}")
        agent_type = payload.get("agent_type", "job_scraping")

        message = {
            "task_id": task_id,
            "agent_type": agent_type,
            "message": payload.get("message", ""),
            "provider": payload.get("provider", "gemini"),
            "model": payload.get("model"),
            "session_id": payload.get("session_id"),
            "tenant_id": payload.get("tenant_id"),
            "user_id": payload.get("user_id"),
            "sources": payload.get("sources"),
            "country": payload.get("country"),
            "experience": payload.get("experience"),
            "industry": payload.get("industry"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        key = f"{agent_type}_{task_id}"

        future = kafka_service.producer.send(
            SCRAPING_TOPIC,
            key=key,
            value=message,
            headers=[
                ("agent_type", agent_type.encode()),
                ("task_id", task_id.encode()),
            ],
        )
        record = future.get(timeout=10)
        logger.info(
            "scraping_producer: published %s to %s partition=%s offset=%s",
            task_id, SCRAPING_TOPIC, record.partition, record.offset,
        )
        return True

    except Exception as e:
        logger.error("scraping_producer: failed to publish: %s", e)
        return False
