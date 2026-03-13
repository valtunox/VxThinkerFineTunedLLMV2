"""
Kafka Scraping Consumer
=======================

Consumes tasks from the ``scraping-tasks`` Kafka topic, runs the appropriate
agent (job_scraping or websearch), and publishes SSE progress events throughout
the execution so the frontend can stream real-time updates.

Start this consumer as a standalone process or call ``start_scraping_consumer()``
from the FastAPI lifespan in a background task.

Usage (standalone)::

    python -m app.services.kafka.scraping_consumer

Usage (background task in app.py lifespan)::

    asyncio.create_task(start_scraping_consumer_async())
"""

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

# Ensure app root is importable
_this_dir = Path(__file__).parent
_app_root = _this_dir.parent.parent
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))

from app.core.logger import get_logger

logger = get_logger(__name__)

SCRAPING_TOPIC = "scraping-tasks"
CONSUMER_GROUP = "scraping-consumer-group"


def _run_job_scraping(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run the job_scraping agent synchronously and return the response dict."""
    from app.services.ai.agents.job_scraping.agent import (
        handle_job_scraping,
        JobScrapingRequest,
    )

    req = JobScrapingRequest(
        query=task.get("message", ""),
        sources=task.get("sources"),
        country=task.get("country"),
        experience=task.get("experience"),
        industry=task.get("industry"),
        provider=task.get("provider", "gemini"),
        model=task.get("model"),
        tenant_id=task.get("tenant_id"),
        user_id=task.get("user_id"),
    )
    response = handle_job_scraping(req)
    return response.model_dump()


def _run_websearch(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run the websearch agent synchronously and return the response dict."""
    from app.services.ai.agents.websearch.agent import (
        handle_websearch,
        WebSearchRequest,
    )

    req = WebSearchRequest(
        message=task.get("message", ""),
        session_id=task.get("session_id"),
        provider=task.get("provider", "gemini"),
        model=task.get("model"),
        auto_save=True,
        agent_type="websearch",
    )
    response = handle_websearch(req)
    return response.model_dump()


_AGENT_HANDLERS = {
    "job_scraping": _run_job_scraping,
    "websearch": _run_websearch,
}


async def _process_task(task: Dict[str, Any]) -> None:
    """
    Process a single scraping task: publish SSE progress, run agent, publish result.
    """
    from app.services.sse.sse_manager import sse_manager

    task_id = task.get("task_id", "unknown")
    agent_type = task.get("agent_type", "job_scraping")
    handler = _AGENT_HANDLERS.get(agent_type)

    if not handler:
        logger.error("scraping_consumer: unknown agent_type=%s for task=%s", agent_type, task_id)
        await sse_manager.publish_progress(
            task_id,
            phase="error",
            message=f"Unknown agent type: {agent_type}",
            status="failed",
        )
        return

    logger.info("scraping_consumer: processing task=%s agent=%s", task_id, agent_type)

    await sse_manager.publish_progress(
        task_id,
        phase="running",
        progress=10,
        message=f"Starting {agent_type.replace('_', ' ')} agent…",
        status="in_progress",
    )

    try:
        await sse_manager.publish_progress(
            task_id,
            phase="searching",
            progress=30,
            message="Searching the web…",
            status="in_progress",
        )

        result = await asyncio.to_thread(handler, task)

        await sse_manager.publish_progress(
            task_id,
            phase="processing",
            progress=70,
            message="Processing results with AI…",
            status="in_progress",
        )

        await sse_manager.publish_progress(
            task_id,
            phase="completed",
            progress=100,
            total=100,
            message=f"{agent_type.replace('_', ' ')} completed successfully.",
            status="completed",
            data=result,
        )

        logger.info("scraping_consumer: task=%s completed", task_id)

    except Exception as e:
        logger.exception("scraping_consumer: task=%s failed: %s", task_id, e)
        await sse_manager.publish_progress(
            task_id,
            phase="error",
            progress=0,
            message=f"Task failed: {e}",
            status="failed",
        )


async def start_scraping_consumer_async() -> None:
    """
    Async entry-point: poll Kafka for scraping tasks in a background loop.
    Designed to run inside ``asyncio.create_task()`` from the FastAPI lifespan.
    """
    from app.services.kafka.kafka_service import KAFKA_AVAILABLE

    if not KAFKA_AVAILABLE:
        logger.warning("scraping_consumer: kafka-python not available, consumer not started")
        return

    await asyncio.sleep(5)

    from app.services.kafka.kafka_service import KafkaService

    svc = KafkaService()
    created = await asyncio.to_thread(svc.create_consumer, CONSUMER_GROUP, [SCRAPING_TOPIC])
    if not created:
        logger.error("scraping_consumer: failed to create Kafka consumer")
        return

    logger.info("scraping_consumer: listening on topic=%s group=%s", SCRAPING_TOPIC, CONSUMER_GROUP)

    def _poll():
        return svc.consumer.poll(timeout_ms=2000)

    try:
        while True:
            try:
                records = await asyncio.to_thread(_poll)
                for tp, messages in records.items():
                    for message in messages:
                        try:
                            task = message.value
                            if isinstance(task, str):
                                task = json.loads(task)
                            await _process_task(task)
                        except Exception as e:
                            logger.error("scraping_consumer: error processing message: %s", e)
            except Exception as e:
                logger.error("scraping_consumer: poll error: %s", e)
                await asyncio.sleep(5)

    except asyncio.CancelledError:
        logger.info("scraping_consumer: shutting down")
    finally:
        svc.close()


def start_scraping_consumer() -> None:
    """Blocking entry-point for running the consumer as a standalone process."""
    asyncio.run(start_scraping_consumer_async())


if __name__ == "__main__":
    start_scraping_consumer()
